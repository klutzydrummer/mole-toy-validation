"""
Griffin/Hawk-style Real-Gated Linear Recurrent Unit (RG-LRU).

Spec: references/components/causal_recurrence.md
Sources: references/sources/papers/griffin_2402.19427.md,
         references/sources/code/griffin_rglru.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from phase1.components._shared import RMSNorm


def _parallel_scan(
    a_t: torch.Tensor, b_t: torch.Tensor, h0: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Parallel linear recurrence: h_t = a_t * h_{t-1} + b_t.

    a_t, b_t: [B, L, d]   a_t must be in (0, 1] (no zeros).
    h0:       [B, d]       optional initial state (default: zeros)
    Returns:  [B, L, d]

    Closed-form solution via log-space prefix scan (O(1) depth, fully vectorised):
      log_A_t = cumsum(log(a_t), dim=1)          — log of running product A_t
      h_t     = A_t * h0 + A_t * cumsum(b_t / A_t, dim=1)

    Computed in float32 regardless of input dtype to avoid float16 saturation.
    log_A is clamped to [-80, 0]:
      - upper bound 0 is exact (a_t ≤ 1 → log ≤ 0)
      - lower bound -80 caps exp(−80) ≈ 1.8e-35; contributions from steps where
        the cumulative decay is that small are negligible (< 1e-34 × input scale)
      - exp(+80) ≈ 5.5e34 fits comfortably in float32 (max ~3.4e38)

    Replaces _sequential_scan (512 Python-loop iterations × 6 instances = ~12k
    CUDA kernel launches per step). Reduces to ~6 ops per instance, compilable
    by torch.compile into a single fused kernel graph.
    """
    a = a_t.float()
    b = b_t.float()
    log_A = torch.log(a.clamp(min=1e-7)).cumsum(dim=1).clamp(min=-80.0)  # [B, L, d]
    A     = log_A.exp()                                                    # [B, L, d]
    h     = A * (b * (-log_A).exp()).cumsum(dim=1)                        # [B, L, d]
    if h0 is not None:
        h = h + A * h0.float().unsqueeze(1)
    return h.to(a_t.dtype)


class CausalRecurrenceLayer(nn.Module):
    """
    Griffin/Hawk-style Real-Gated Linear Recurrent Unit (RG-LRU).

    Per Griffin (arXiv:2402.19427):
      a_t = sigmoid(log_a)^(8 * sigmoid(W_r · x_t))   — input-dependent decay
      h_t = a_t * h_{t-1} + sqrt(1 - a_t²) * (sigmoid(W_i · x_t) * x_t)

    The sqrt(1 - a_t²) term is NOT optional — it maintains E[||h||²] bounded.
    Without it, hidden state norm grows unboundedly at small d.

    log_a_init controls the initial memory length:
      log_a_init=7.5  sigmoid(7.5)^8 ≈ 0.995, half-life ~289 steps  (too long — gradient
                      through log_a ∝ (1-sigmoid(7.5)) ≈ 0.0006, essentially frozen)
      log_a_init=3.0  sigmoid(3.0)^8 ≈ 0.68,  half-life ~4 steps    (CRLEncoder default)
      log_a_init=0.0  sigmoid(0.0)^8 = 0.004,  half-life <1 step     (not used in Phase 2 outer)

    CRLEncoder uses log_a_init=3.0: encoder captures ~4-token local context, enough contrast
    for boundary detection, gradient (1-sigmoid(3.0)) ≈ 0.047 allows adaptation.
    """

    def __init__(self, d: int, log_a_init: float = 7.5):
        super().__init__()
        self.d = d

        # Causal depthwise conv for local mixing (kernel=4, left-pad manually)
        # groups=d makes it depthwise: each channel convolved independently
        self.conv_weight = nn.Parameter(torch.randn(d, 1, 4) * 0.02)
        self.conv_bias   = nn.Parameter(torch.zeros(d))

        # Recurrence gates
        self.W_r = nn.Linear(d, d, bias=True)   # recurrence gate: controls decay strength
        self.W_i = nn.Linear(d, d, bias=True)   # input gate: controls write strength

        # Per-channel log decay — see docstring for init value rationale.
        self.log_a = nn.Parameter(torch.full((d,), log_a_init))

        # Output
        self.out_proj = nn.Linear(d, d, bias=False)
        self.norm = RMSNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d] → [B, L, d]"""
        B, L, d = x.shape

        # 1. Causal depthwise conv (kernel=4, pad 3 on left only → causal)
        x_t      = x.transpose(1, 2)                          # [B, d, L]
        x_padded = F.pad(x_t, (3, 0))                         # [B, d, L+3]
        x_conv   = F.conv1d(x_padded, self.conv_weight,
                             self.conv_bias, groups=d)         # [B, d, L]
        x_conv   = x_conv.transpose(1, 2)                     # [B, L, d]

        # 2. Per-timestep gates
        r_t = torch.sigmoid(self.W_r(x_conv))                 # [B, L, d] in (0, 1)
        i_t = torch.sigmoid(self.W_i(x_conv))                 # [B, L, d] in (0, 1)

        # Per-channel base decay raised to 8*r_t (input-dependent).
        # Compute sigmoid(log_a) in float32 before casting back to working dtype.
        # float16 max < 1.0 is 0.99951; for large log_a values this is a safe
        # practice to preserve precision in the decay computation.
        a_base = torch.sigmoid(self.log_a.float()).to(x.dtype)  # [d] float32 → dtype
        a_t    = a_base.pow(8.0 * r_t)                          # [B, L, d] in (~0.9, 1)

        # 3. Recurrence scan with norm conservation
        # Pre-compute b_t = sqrt(1 - a_t²) * (i_t * x_conv) for all timesteps,
        # then run parallel scan (log-space cumsum, fully vectorised, compilable).
        b_t = torch.sqrt((1.0 - a_t * a_t).clamp(min=1e-6)) * (i_t * x_conv)
        out = _parallel_scan(a_t, b_t)                           # [B, L, d]

        # 4. Output projection + norm
        return self.norm(self.out_proj(out))
