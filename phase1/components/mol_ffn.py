"""
Mixture-of-LoRAs FFN (MoL) — routing via DeepSeek-V3 pattern.

Spec: references/components/mol_ffn.md
Sources: references/sources/papers/deepseek_v3_2412.19437.md,
         references/sources/code/deepseek_v3_moe.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from phase1.components._shared import SwiGLU  # noqa: F401 — re-exported for TransformerBlock


class LoRAAdapter(nn.Module):
    """Single LoRA: output = (x @ A @ B) * scale. B=0 at init (no-op)."""

    def __init__(self, d_in: int, d_out: int, rank: int):
        super().__init__()
        self.A = nn.Parameter(torch.randn(d_in, rank) * (1.0 / math.sqrt(d_in)))
        self.B = nn.Parameter(torch.zeros(rank, d_out))
        self.scale = 1.0 / rank

    def forward(self, x):
        return (x @ self.A @ self.B) * self.scale


class SingleLoRAFFN(nn.Module):
    """
    Single high-rank LoRA over a SwiGLU base — no routing, no experts.

    Q2 ablation control for MoLFFN: same base FFN + exactly equal total
    LoRA parameter count, but no routing or specialization mechanism.

    MoLFFN has 9 adapter sets (1 shared + 8 experts) × rank 8 = rank-72 equivalent.
    We use rank=72 (exact capacity match) as the Q2 baseline.
    """

    def __init__(self, d: int, rank: int = 72, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            d_ff = int(d * 8 / 3)
            d_ff = ((d_ff + 63) // 64) * 64
        self.d_ff = d_ff

        # Base SwiGLU projections
        self.base_gate = nn.Linear(d, d_ff, bias=False)
        self.base_up   = nn.Linear(d, d_ff, bias=False)
        self.base_down = nn.Linear(d_ff, d, bias=False)

        # Single high-rank LoRA (always active, no routing)
        self.lora_gate = LoRAAdapter(d, d_ff, rank)
        self.lora_up   = LoRAAdapter(d, d_ff, rank)
        self.lora_down = LoRAAdapter(d_ff, d, rank)

    def forward(self, x):
        gate   = self.base_gate(x) + self.lora_gate(x)
        up     = self.base_up(x)   + self.lora_up(x)
        hidden = F.silu(gate) * up
        return self.base_down(hidden) + self.lora_down(hidden)


class MoLFFN(nn.Module):
    """
    Mixture-of-LoRAs FFN with correct weight-space composition.

    W_eff · x = W_base·x + ΔW_shared·x + Σ w_k · (ΔW_k · x)
    For SwiGLU: corrections added BEFORE silu (gate/up) and AFTER (down).

    Routing follows DeepSeek-V3 (deepseek-ai/DeepSeek-V3 model.py):
      - Bias added for TOP-K SELECTION only
      - Forward-pass weights use UNBIASED affinity scores
    """

    def __init__(self, d: int, n_experts: int = 8, top_k: int = 2,
                 rank: int = 8, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            d_ff = int(d * 8 / 3)
            d_ff = ((d_ff + 63) // 64) * 64

        self.n_experts = n_experts
        self.top_k = top_k
        self.d = d
        self.d_ff = d_ff

        # Base projections
        self.base_gate = nn.Linear(d, d_ff, bias=False)
        self.base_up = nn.Linear(d, d_ff, bias=False)
        self.base_down = nn.Linear(d_ff, d, bias=False)

        # Shared LoRA (always active)
        self.shared_gate = LoRAAdapter(d, d_ff, rank)
        self.shared_up = LoRAAdapter(d, d_ff, rank)
        self.shared_down = LoRAAdapter(d_ff, d, rank)

        # Conditional LoRA experts
        self.expert_gate = nn.ModuleList([LoRAAdapter(d, d_ff, rank) for _ in range(n_experts)])
        self.expert_up = nn.ModuleList([LoRAAdapter(d, d_ff, rank) for _ in range(n_experts)])
        self.expert_down = nn.ModuleList([LoRAAdapter(d_ff, d, rank) for _ in range(n_experts)])

        # Router
        self.router = nn.Linear(d, n_experts, bias=False)

        # Aux-loss-free load balancing (not gradient-trained)
        self.register_buffer("expert_bias", torch.zeros(n_experts))
        self.register_buffer("expert_counts", torch.zeros(n_experts))
        self.bias_step = 0.01  # γ, larger than DS-V3's 0.001 for toy scale

    def forward(self, x):
        B, L, D = x.shape

        # ---- Routing (DeepSeek-V3 pattern: bias for selection, unbiased for weights) ----
        logits = self.router(x)                             # [B, L, n_experts]
        scores = torch.sigmoid(logits)                      # unbiased affinity
        biased = scores + self.expert_bias                  # biased for selection only
        _, topk_idx = biased.topk(self.top_k, dim=-1)      # [B, L, k] — select with bias
        topk_scores = scores.gather(2, topk_idx)            # [B, L, k] — weight with UNBIASED
        topk_weights = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-8)

        # Build per-expert weight tensor via one_hot (differentiable through topk_weights)
        one_hot = F.one_hot(topk_idx, self.n_experts).to(x.dtype)  # [B, L, k, n_experts]
        expert_weights = (one_hot * topk_weights.unsqueeze(-1)).sum(dim=2)  # [B, L, n_experts]

        # ---- Batched expert LoRA (replaces per-expert loop) ----
        scale = self.expert_gate[0].scale  # 1/rank, identical for all experts
        w = expert_weights.unsqueeze(-1)   # [B, L, n_experts, 1]

        A_gate = torch.stack([e.A for e in self.expert_gate])  # [E, d,    rank]
        B_gate = torch.stack([e.B for e in self.expert_gate])  # [E, rank, d_ff]
        A_up   = torch.stack([e.A for e in self.expert_up])
        B_up   = torch.stack([e.B for e in self.expert_up])

        # x: [B,L,d] → [B,L,E,rank] → [B,L,E,d_ff], then weighted sum → [B,L,d_ff]
        xA_gate  = torch.einsum("bld,edr->bler", x, A_gate)
        xA_up    = torch.einsum("bld,edr->bler", x, A_up)
        xAB_gate = torch.einsum("bler,erd->bled", xA_gate, B_gate) * scale
        xAB_up   = torch.einsum("bler,erd->bled", xA_up,   B_up)   * scale

        # ---- Gate / Up: base + shared + Σ weighted expert corrections (BEFORE silu) ----
        gate = self.base_gate(x) + self.shared_gate(x) + (xAB_gate * w).sum(dim=2)
        up   = self.base_up(x)   + self.shared_up(x)   + (xAB_up   * w).sum(dim=2)

        # ---- Nonlinearity (ONCE on combined projections) ----
        hidden = F.silu(gate) * up

        # ---- Down: base + shared + Σ weighted expert corrections ----
        A_down = torch.stack([e.A for e in self.expert_down])  # [E, d_ff, rank]
        B_down = torch.stack([e.B for e in self.expert_down])  # [E, rank, d]
        xA_down  = torch.einsum("bld,edr->bler", hidden, A_down)
        xAB_down = torch.einsum("bler,erd->bled", xA_down, B_down) * scale
        output = self.base_down(hidden) + self.shared_down(hidden) + (xAB_down * w).sum(dim=2)

        # ---- Load balance update (no gradient) ----
        if self.training:
            with torch.no_grad():
                counts = torch.zeros(self.n_experts, device=x.device)
                for k in range(self.top_k):
                    counts.scatter_add_(0, topk_idx[:, :, k].reshape(-1),
                                        torch.ones(B * L, device=x.device))
                avg = counts.mean()
                self.expert_bias += self.bias_step * (avg - counts).sign()
                self.expert_counts += counts

        return output

    def get_load_stats(self):
        total = self.expert_counts.sum()
        if total == 0:
            return {}
        probs = self.expert_counts / total
        entropy = -(probs * (probs + 1e-8).log()).sum().item()
        max_entropy = math.log(self.n_experts)
        return {
            "expert_entropy": entropy,
            "expert_balance": entropy / max_entropy,
            "expert_counts": self.expert_counts.tolist(),
        }

    def reset_counts(self):
        self.expert_counts.zero_()
