"""
Phase 2 Model: Outer Encoder Study — pluggable ZoneE + threshold-based BoundaryRouter + SimpleDecoder.

Architecture:
  embed(x) [B, L, d]
    → ZoneE (pluggable encoder) → encoder_out [B, L, d]
    → BoundaryRouter (threshold p > 0.5) → concept_tokens [B, M_max, d] (padded)
                                          → boundary_idx  [B, M_max]    (padded with 0)
                                          → concept_mask  [B, M_max]    (True = valid)
    → SimpleDecoder (H-Net Eq. 5–9):
        → EMA smooth over M concept tokens
        → plug-back via cumsum(boundary_mask) - 1 → [B, L, d]   (H-Net Eq. 8)
        → confidence scoring + STE scaling                        (H-Net Eq. 6–7–9)
        → residual_proj (plain linear, near-zero init)            (H-Net Eq. 3)
        → lm_head → logits [B, L, vocab]

Configs:
  outer_crl           CRL encoder, cosine_rule router. Baseline. Run first.
  outer_crl_learned   CRL encoder, learned_e2e router. Main A study.
  outer_transformer   4-layer baseline transformer encoder, learned_e2e router.
  outer_diff_attn     4-layer diff_attn transformer encoder, learned_e2e router.
  outer_mla           4-layer MLA transformer encoder, learned_e2e router.
  outer_strided       Identity encoder, fixed_stride router. Lower bound baseline.

Key correctness invariants:
  - CausalRecurrenceLayer: sqrt(1 - a_t²) normalization (Griffin arXiv:2402.19427)
  - BoundaryRouter: threshold p > 0.5, NOT topk. Variable M per sequence, padded to M_max.
  - boundary_probs[:, 0] == 1.0 for cosine_rule and learned_e2e
  - SimpleDecoder EMA: p_at_bounds.clamp(min=0.1) prevents EMA collapse
  - STE: ste(c_t) = c_t + stopgradient(1 - c_t) — Eq. 7; = 1.0 in forward pass
  - residual_proj initialized near-zero (weight=0, no bias) — H-Net warm-start (Eq. 3)
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1.model import RMSNorm, TransformerBlock  # noqa: I001


# ============================================================
# Compile-friendly helpers
# ============================================================

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


# ============================================================
# CausalRecurrenceLayer
# ============================================================

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


# ============================================================
# Zone E: Pluggable Encoders
# ============================================================

class CRLEncoder(nn.Module):
    """
    CRL-based encoder: down_proj → 3× CausalRecurrenceLayer → up_proj.
    d → d//4 → d//4 → d//4 → d
    log_a_init=3.0: half-life ~4 steps, trainable gradient.
    """

    def __init__(self, d: int):
        super().__init__()
        d_inner = d // 4
        self.down_proj  = nn.Linear(d, d_inner, bias=False)
        self.recurrence = nn.ModuleList([
            CausalRecurrenceLayer(d_inner, log_a_init=3.0) for _ in range(3)
        ])
        self.up_proj = nn.Linear(d_inner, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d] → [B, L, d]"""
        h = self.down_proj(x)
        for rec in self.recurrence:
            h = rec(h)
        return self.up_proj(h)


class CRLEncoderFull(nn.Module):
    """
    Full-width CRL encoder: 3× CausalRecurrenceLayer directly at d (no bottleneck).
    No down/up projection — operates at full model dimension throughout.

    Param count at d=512: 3 × ~525K ≈ 1.57M params.
    Compare to CRLEncoder (bottlenecked at d//4=128): ~282K params.
    Compare to transformer encoders: ~11–14M params.

    Use this config when you want CRL without the bottleneck as a confound.
    """

    def __init__(self, d: int):
        super().__init__()
        self.recurrence = nn.ModuleList([
            CausalRecurrenceLayer(d, log_a_init=3.0) for _ in range(3)
        ])
        self.norm_out = RMSNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d] → [B, L, d]"""
        for rec in self.recurrence:
            x = rec(x)
        return self.norm_out(x)


class TransformerEncoder(nn.Module):
    """
    Standard causal transformer encoder (baseline attention).
    n_layers_outer causal transformer layers using TransformerBlock with baseline config.
    """

    def __init__(self, d: int, n_heads: int, n_layers_outer: int = 4, max_len: int = 256):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d=d, n_heads=n_heads, layer_idx=i, max_len=max_len)
            for i in range(n_layers_outer)
        ])
        self.norm_out = RMSNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d] → [B, L, d]"""
        for block in self.blocks:
            x = block(x)
        return self.norm_out(x)


class DiffAttnEncoder(nn.Module):
    """
    Differential Attention V2 causal transformer encoder.
    n_layers_outer causal transformer layers using TransformerBlock with diff_attn config.
    """

    def __init__(self, d: int, n_heads: int, n_layers_outer: int = 4, max_len: int = 256):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d=d, n_heads=n_heads, use_diff_attn=True,
                             layer_idx=i, max_len=max_len)
            for i in range(n_layers_outer)
        ])
        self.norm_out = RMSNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d] → [B, L, d]"""
        for block in self.blocks:
            x = block(x)
        return self.norm_out(x)


class MLAEncoder(nn.Module):
    """
    MLA (Multi-head Latent Attention) causal transformer encoder.
    n_layers_outer causal transformer layers using TransformerBlock with mla config.
    """

    def __init__(self, d: int, n_heads: int, n_layers_outer: int = 4, max_len: int = 256):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d=d, n_heads=n_heads, use_mla=True,
                             layer_idx=i, max_len=max_len)
            for i in range(n_layers_outer)
        ])
        self.norm_out = RMSNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d] → [B, L, d]"""
        for block in self.blocks:
            x = block(x)
        return self.norm_out(x)


class IdentityEncoder(nn.Module):
    """
    No-op encoder: returns x unchanged.
    Used for outer_strided (lower bound — no encoder processing).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d] → [B, L, d]"""
        return x


# ============================================================
# Boundary Router (threshold-based)
# ============================================================

class BoundaryRouter(nn.Module):
    """
    Selects concept-token positions from L encoder outputs using threshold p > 0.5.

    Unlike the old topk(M) approach, this produces variable M per sequence, padded
    to M_max = max boundary count in the batch. This is the correct H-Net formulation.

    Three routing modes:

    cosine_rule:
      p_t = (1 - cos_sim(enc_t, enc_{t-1})) / 2, no learned params.
      Most stable. Validates pipeline before adding learned routing.

    learned_e2e:
      p_t = (1 - dot(normalize(W_q·enc_t), normalize(W_k·enc_{t-1}))) / 2
      W_q, W_k initialized to identity (pure cosine similarity at init).
      LM gradients flow through boundary_probs via SimpleDecoder's EMA and gated residual.
      This is the H-Net approach (arXiv:2507.07955).

    fixed_stride (outer_strided):
      Select evenly spaced positions (round(L * target_rate) positions across L).
      No boundary_probs computation; boundary_probs = zeros except at selected positions.

    IMPORTANT: always uses k_{t-1} (adjacent key), NOT k_ema.
    """

    def __init__(self, d: int, routing: str = "cosine_rule", target_rate: float = 0.25):
        super().__init__()
        self.d           = d
        self.routing     = routing
        self.target_rate = target_rate

        if routing == "learned_e2e":
            # Identity init: router starts as pure cosine similarity
            self.W_q = nn.Linear(d, d, bias=False)
            self.W_k = nn.Linear(d, d, bias=False)
            nn.init.eye_(self.W_q.weight)
            nn.init.eye_(self.W_k.weight)

    def forward(self, enc: torch.Tensor):
        """
        enc: [B, L, d] encoder output

        Returns:
          concept_tokens: [B, M_max, d]  gathered encoder outputs at boundaries (padded)
          encoder_out:    [B, L, d]      pass-through (same as enc)
          boundary_probs: [B, L]         soft boundary probability in [0, 1]
          boundary_idx:   [B, M_max]     positions of selected concept tokens (padded with 0)
          concept_mask:   [B, M_max]     True for valid concept tokens, False for padding
        """
        B, L, d = enc.shape

        # ── Fixed stride ──────────────────────────────────────────────────────
        if self.routing == "fixed_stride":
            M = max(1, round(L * self.target_rate))
            # Evenly spaced positions across L
            positions = torch.linspace(0, L - 1, steps=M, device=enc.device).long()
            # All sequences get the same positions; M_max = M (no padding needed)
            boundary_idx   = positions.unsqueeze(0).expand(B, -1).contiguous()  # [B, M]
            boundary_probs = torch.zeros(B, L, device=enc.device, dtype=enc.dtype)
            boundary_probs.scatter_(1, boundary_idx, 1.0)
            concept_mask   = torch.ones(B, M, device=enc.device, dtype=torch.bool)
            concept_tokens = enc.gather(
                1, boundary_idx.unsqueeze(-1).expand(-1, -1, d)
            )  # [B, M, d]
            return concept_tokens, enc, boundary_probs, boundary_idx, concept_mask

        # ── Compute boundary probabilities ────────────────────────────────────
        if self.routing == "cosine_rule":
            enc_n = F.normalize(enc, dim=-1)                   # [B, L, d]
            sim   = (enc_n[:, 1:] * enc_n[:, :-1]).sum(dim=-1) # [B, L-1]
            p     = (1.0 - sim) / 2.0                          # [B, L-1] in [0, 1]
            # Position 0: no predecessor → pad with 1.0 (certain boundary).
            # Must be 1.0, not 0.0: SimpleDecoder's EMA uses p_at_bounds[:,0] directly.
            boundary_probs = F.pad(p, (1, 0), value=1.0)       # [B, L]

        else:  # learned_e2e
            q = F.normalize(self.W_q(enc), dim=-1)             # [B, L, d]
            k = F.normalize(self.W_k(enc), dim=-1)             # [B, L, d]
            # Compare each q_t to k_{t-1} (adjacent, NOT ema) — H-Net Eq. 4
            sim = (q[:, 1:] * k[:, :-1]).sum(dim=-1)           # [B, L-1]
            p   = (1.0 - sim) / 2.0                            # [B, L-1]
            boundary_probs = F.pad(p, (1, 0), value=1.0)       # [B, L], pos 0 = 1.0

        # ── Threshold selection: p > 0.5, variable M per sequence ─────────────
        # Position 0 is always 1.0 → always selected. Subsequent positions selected
        # where cosine dissimilarity is high (tokens differ from predecessor).
        boundary_mask = (boundary_probs > 0.5)                 # [B, L] bool
        counts        = boundary_mask.sum(dim=1)               # [B]
        M_max         = int(counts.max().item())
        M_max         = max(M_max, 1)                          # at least 1 slot

        # Build padded boundary_idx and concept_mask — fully vectorised, no Python loop.
        # Compile-friendly: no data-dependent control flow, no per-element Python ops.
        #
        # Strategy: assign each boundary position a 0-indexed rank within its row.
        #   rank[b, l] = cumsum(b_long)[b, l] * b_long[b, l]
        #              = 1-indexed rank if boundary, 0 if non-boundary.
        # Non-boundary positions (rank=0) must not clobber slot 0.
        # Fix: route non-boundary positions to a dummy slot M_max and discard it.
        #   slot_safe[b, l] = rank[b,l]-1  if boundary   (0-indexed valid slot)
        #                   = M_max        if non-boundary (discarded dummy slot)
        # Allocate M_max+1 slots, scatter, then trim the last slot.
        b_long   = boundary_mask.long()                         # [B, L]
        rank     = b_long.cumsum(dim=1) * b_long                # [B, L] 1-indexed rank, 0 if not boundary
        slot_safe = torch.where(
            boundary_mask,
            rank - 1,                                           # 0-indexed slot for boundaries
            torch.full_like(rank, M_max),                       # dummy slot for non-boundaries
        )                                                        # [B, L]

        pos = torch.arange(L, device=enc.device).unsqueeze(0).expand(B, -1)  # [B, L]
        # Allocate M_max+1 slots; the last (M_max) is the dummy absorbing non-boundary writes.
        _bidx = torch.zeros(B, M_max + 1, device=enc.device, dtype=torch.long)
        _cmsk = torch.zeros(B, M_max + 1, device=enc.device, dtype=torch.bool)
        _bidx.scatter_(1, slot_safe, pos)
        _cmsk.scatter_(1, slot_safe, boundary_mask)
        boundary_idx = _bidx[:, :M_max]                         # [B, M_max] — drop dummy slot
        concept_mask = _cmsk[:, :M_max]                         # [B, M_max]

        # Gather concept tokens at boundary positions
        concept_tokens = enc.gather(
            1, boundary_idx.unsqueeze(-1).expand(-1, -1, d)   # [B, M_max, d]
        )

        return concept_tokens, enc, boundary_probs, boundary_idx, concept_mask


# ============================================================
# SimpleDecoder: EMA smooth → plug-back → gated residual → lm_head
# ============================================================

class SimpleDecoder(nn.Module):
    """
    Reconstructs L-length representations from M_max concept tokens (padded).

    No inner transformer. No CRL decoder layers.

    Implements H-Net Equations 5, 6, 7, 8, 9, 3 in order:

    1. EMA smoothing (H-Net Eq. 5) over M_max concept tokens:
         h̄_i = p_i * concept_i + (1 - p_i) * h̄_{i-1}
         p_i  = boundary_probs at boundary position i, clamped to [0.1, 1.0]
         h0   = concept_tokens[:, 0]  (always present since p_0=1.0)

    2. Plug-back (H-Net Eq. 8) via cumsum(boundary_mask) - 1:
         plug_back_idx = cumsum(boundary_mask.long(), dim=1) - 1  [B, L]
         maps each of L positions to its nearest preceding concept token in smoothed [B, M_max, d]

    3. Confidence scoring + STE (H-Net Eq. 6–7–9):
         c_t     = p_t  if b_t=1,  (1 - p_t) if b_t=0          (Eq. 6)
         ste_c_t = c_t + stopgradient(1 - c_t)  = 1.0 in fwd   (Eq. 7)
         upsampled_t = ste_c_t * z̃_t                             (Eq. 9)
         Incentivizes the router to make confident, decisive decisions.

    4. Residual (H-Net Eq. 3):
         out = upsampled + residual_proj(encoder_out)
         residual_proj is nn.Linear(d, d, bias=False) initialized near-zero (weight=0).
         No sigmoid, no p-modulation — plain linear skip, suppressed at init.

    lm_head is weight-tied to embed — applied in OuterModel.forward.
    """

    def __init__(self, d: int, use_ste: bool = True):
        super().__init__()
        self.use_ste = use_ste
        # Plain linear skip connection (H-Net Eq. 3); initialized near-zero in OuterModel.__init__
        self.residual_proj = nn.Linear(d, d, bias=False)

    def forward(
        self,
        concept_tokens: torch.Tensor,   # [B, M_max, d]  concept token representations
        encoder_out:    torch.Tensor,   # [B, L, d]      Zone E output (skip connection)
        boundary_probs: torch.Tensor,   # [B, L]         soft boundary probabilities
        boundary_idx:   torch.Tensor,   # [B, M_max]     sorted boundary positions (padded)
        concept_mask:   torch.Tensor,   # [B, M_max]     True = valid, False = padding
    ) -> torch.Tensor:
        """Returns [B, L, d] reconstructed token representations."""
        B, L, d = encoder_out.shape
        M_max   = concept_tokens.shape[1]

        # ── Step 1: EMA smoothing over M_max concept tokens (H-Net Eq. 5) ─────
        # EMA recurrence: h_i = p_i * concept_i + (1 - p_i) * h_{i-1}
        # h0 = concept_tokens[:, 0]  (pos 0 always valid since boundary_probs[:,0]=1.0)
        #
        # p_at_bounds: boundary_probs at each concept position.
        # Padding slots get p=1.0 so EMA treats them as fresh boundaries (no bleed).
        p_at_bounds_raw = boundary_probs.gather(1, boundary_idx)        # [B, M_max]
        p_at_bounds = torch.where(concept_mask, p_at_bounds_raw,
                                  torch.ones_like(p_at_bounds_raw))     # [B, M_max]
        p_at_bounds = p_at_bounds.clamp(min=0.1)                        # EMA collapse guard

        h0 = concept_tokens[:, 0]                                       # [B, d]
        if M_max > 1:
            p_rest   = p_at_bounds[:, 1:]                               # [B, M_max-1]
            decay    = (1.0 - p_rest).clamp(min=1e-7)                  # [B, M_max-1]
            b_rest   = p_rest.unsqueeze(-1) * concept_tokens[:, 1:]    # [B, M_max-1, d]
            a_rest   = decay.unsqueeze(-1).expand_as(b_rest)           # [B, M_max-1, d]
            h_rest   = _parallel_scan(a_rest, b_rest, h0)              # [B, M_max-1, d]
            smoothed = torch.cat([h0.unsqueeze(1), h_rest], dim=1)     # [B, M_max, d]
        else:
            smoothed = h0.unsqueeze(1)                                  # [B, 1, d]

        # ── Step 2: Plug-back via cumsum (H-Net Eq. 8) ───────────────────────
        # boundary_mask[b, j] = True iff position j is a boundary in sequence b.
        # cumsum(boundary_mask) - 1 gives the 0-indexed concept bucket for each L position:
        #   positions before the first boundary clamp to 0 (safe since p_0=1.0 always)
        #   each boundary position starts a new bucket
        boundary_mask  = (boundary_probs >= 0.5)                        # [B, L] bool
        plug_back_idx  = torch.cumsum(boundary_mask.long(), dim=1) - 1 # [B, L]
        plug_back_idx  = plug_back_idx.clamp(min=0)                    # guard against -1 at pos 0
        plugback = smoothed.gather(
            1, plug_back_idx.unsqueeze(-1).expand(-1, -1, d),
        )                                                                # [B, L, d]

        # ── Step 3: Confidence scoring + STE (H-Net Eq. 6–7–9) ──────────────
        # c_t = p_t if b_t=1, (1-p_t) if b_t=0 — quantifies router confidence
        # ste(c_t) = c_t + stopgrad(1-c_t) = 1.0 in fwd, gradient flows through c_t
        # Incentivizes confident decisions: uncertain router (p≈0.5) gets output scaled by 0.5
        # use_ste=False: ablation — use plain c_t without STE (no rounding to 1.0 in fwd)
        b_float = boundary_mask.float()                                 # [B, L]
        c_t     = b_float * boundary_probs + (1.0 - b_float) * (1.0 - boundary_probs)
        if self.use_ste:
            ste_c = c_t + (1.0 - c_t).detach()                        # [B, L] = 1.0 in fwd
        else:
            ste_c = c_t                                                 # plain confidence scaling
        upsampled = ste_c.unsqueeze(-1) * plugback                     # [B, L, d]

        # ── Step 4: Residual from Zone E skip (H-Net Eq. 3) ─────────────────
        # out = upsampled + residual_proj(encoder_out)
        # residual_proj weight=0 at init → residual path suppressed at start of training
        out = upsampled + self.residual_proj(encoder_out)              # [B, L, d]

        return out


# ============================================================
# OuterModel
# ============================================================

class OuterModel(nn.Module):
    """
    Outer encoder study model: pluggable ZoneE + threshold BoundaryRouter + SimpleDecoder.

    Forward returns (logits [B, L, vocab], boundary_probs [B, L], compression_ratio scalar).

    Training loop:
      logits, bp, cr = model(x)
      loss_ntp  = F.cross_entropy(logits.view(-1, V), y.view(-1))
      loss_comp = (bp.mean() - target_rate) ** 2
      loss      = loss_ntp + lambda_comp * loss_comp

    Weight tying: lm_head.weight = embed.weight (same as Phase 1).
    """

    CONFIGS = {
        # ── Core study configs ────────────────────────────────────────────────
        "outer_crl":              dict(encoder="crl",          router="cosine_rule"),
        "outer_crl_learned":      dict(encoder="crl",          router="learned_e2e"),
        "outer_crl_full":         dict(encoder="crl_full",     router="cosine_rule"),
        "outer_crl_full_learned": dict(encoder="crl_full",     router="learned_e2e"),
        "outer_transformer":      dict(encoder="transformer",  router="learned_e2e"),
        "outer_diff_attn":        dict(encoder="diff_attn",    router="learned_e2e"),
        "outer_mla":              dict(encoder="mla",          router="learned_e2e"),
        "outer_strided":          dict(encoder="identity",     router="fixed_stride"),
        # ── Ablations ─────────────────────────────────────────────────────────
        "outer_crl_learned_noste": dict(encoder="crl",         router="learned_e2e",
                                        use_ste=False),
        "outer_crl_r2":            dict(encoder="crl",         router="cosine_rule",
                                        target_rate=0.5),
        # Encoder param counts at d=512 (for cross-config interpretation):
        #   outer_crl / outer_crl_learned          CRLEncoder (d//4=128 bottleneck):  ~282K
        #   outer_crl_full / outer_crl_full_learned CRLEncoderFull (d=512, no proj):  ~1.57M
        #   outer_transformer                       TransformerEncoder (4×d=512):     ~12.8M
        #   outer_diff_attn                         DiffAttnEncoder (4×d=512):        ~13.9M
        #   outer_mla                               MLAEncoder (4×d=512):             ~11.5M
        # These are architecture-vs-architecture comparisons, NOT param-matched.
    }

    def __init__(
        self,
        config:         str   = "outer_crl",
        d:              int   = 512,
        n_layers:       int   = 8,       # kept for API compatibility; unused in Phase 2 outer
        n_heads:        int   = 8,
        vocab_size:     int   = 4096,
        seq_len:        int   = 256,
        n_layers_outer: int   = 4,       # transformer encoder depth
        target_rate:    float = 0.25,    # soft sparsity target for loss_comp
    ):
        super().__init__()
        if config not in self.CONFIGS:
            raise ValueError(f"Unknown config '{config}'. Valid: {list(self.CONFIGS)}")

        cfg = self.CONFIGS[config]
        self.config_name = config
        # Config can override target_rate (e.g. outer_crl_r2 uses 0.5)
        self.target_rate = cfg.get("target_rate", target_rate)
        self.use_ste     = cfg.get("use_ste", True)
        self.d           = d

        # Shared embedding (weight-tied to lm_head)
        self.embed = nn.Embedding(vocab_size, d)

        # Pluggable encoder
        encoder_type = cfg["encoder"]
        if encoder_type == "crl":
            self.encoder = CRLEncoder(d=d)
        elif encoder_type == "crl_full":
            self.encoder = CRLEncoderFull(d=d)
        elif encoder_type == "transformer":
            self.encoder = TransformerEncoder(d=d, n_heads=n_heads,
                                              n_layers_outer=n_layers_outer,
                                              max_len=seq_len)
        elif encoder_type == "diff_attn":
            self.encoder = DiffAttnEncoder(d=d, n_heads=n_heads,
                                           n_layers_outer=n_layers_outer,
                                           max_len=seq_len)
        elif encoder_type == "mla":
            self.encoder = MLAEncoder(d=d, n_heads=n_heads,
                                      n_layers_outer=n_layers_outer,
                                      max_len=seq_len)
        elif encoder_type == "identity":
            self.encoder = IdentityEncoder()
        else:
            raise ValueError(f"Unknown encoder type '{encoder_type}'")

        # Threshold-based boundary router
        self.router = BoundaryRouter(d=d, routing=cfg["router"],
                                     target_rate=self.target_rate)

        # Simple decoder (no inner transformer, no CRL decoder)
        self.decoder = SimpleDecoder(d=d, use_ste=self.use_ste)

        # LM head (weight-tied)
        self.lm_head        = nn.Linear(d, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        # Standard init for all nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Re-apply eye init for BoundaryRouter W_q/W_k after _init_weights
        if hasattr(self.router, "W_q"):
            nn.init.eye_(self.router.W_q.weight)
            nn.init.eye_(self.router.W_k.weight)

        # residual_proj near-zero init: H-Net warm-start requirement (Eq. 3).
        # Weight=0 suppresses the skip connection at init, forcing the model to rely
        # on concept token representations before the skip path activates.
        nn.init.zeros_(self.decoder.residual_proj.weight)

    def _init_weights(self, m):
        """Standard init for nn.Linear and nn.Embedding.
        CausalRecurrenceLayer.log_a (nn.Parameter) keeps its role-specific init.
        BoundaryRouter W_q/W_k are re-initialized to eye_ after this pass."""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor):
        """
        x: [B, L] token indices

        Returns: (logits [B, L, vocab_size], boundary_probs [B, L], compression_ratio scalar)

        Training loop:
          logits, bp, cr = model(x)
          loss_ntp  = F.cross_entropy(logits.view(-1, V), y.view(-1))
          loss_comp = (bp.mean() - self.target_rate) ** 2
          loss      = loss_ntp + lambda_comp * loss_comp
        """
        # Embed once — shared between encoder and weight-tied lm_head
        h = self.embed(x)                                               # [B, L, d]

        # Encode: [B, L, d] → [B, L, d]
        encoder_out = self.encoder(h)                                   # [B, L, d]

        # Route: threshold-based selection of concept token positions
        concept_tokens, _, boundary_probs, boundary_idx, concept_mask = \
            self.router(encoder_out)
        # concept_tokens: [B, M_max, d]
        # boundary_idx:   [B, M_max]
        # concept_mask:   [B, M_max]

        # Decode: EMA smooth → plug-back → gated residual
        token_repr = self.decoder(
            concept_tokens, encoder_out, boundary_probs,
            boundary_idx, concept_mask,
        )                                                               # [B, L, d]

        logits = self.lm_head(token_repr)                               # [B, L, vocab_size]

        # Compression ratio: mean fraction of tokens selected as boundaries
        B, L = x.shape
        compression_ratio = concept_mask.float().sum(dim=1).mean() / L  # scalar

        return logits, boundary_probs, compression_ratio
