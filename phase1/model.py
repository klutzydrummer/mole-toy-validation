"""
Phase 1 Model: Character-level transformer with optional mHC and MoL.

Configurations:
  baseline  - Vanilla transformer (SwiGLU FFN, RMSNorm, RoPE)
  mhc       - + KromHC hyper-connections (n=4 streams, exact doubly stochastic)
  mol       - + mixture-of-LoRAs FFN (8 experts, top-2, rank 8)
  compose   - mHC + MoL together

Reference implementations consulted:
  - mHC formula: x_{l+1} = H_res · x_l + H_post^T · F(H_pre · x_l)
    Source: tokenbender/mHC (GitHub), lucidrains/hyper-connections (PyPI),
    DeepSeek arXiv:2512.24880, HC paper arXiv:2409.19606 Figure 2b
  - MoL routing: sigmoid scores, bias for selection only, unbiased for weights
    Source: deepseek-ai/DeepSeek-V3 inference/model.py (Gate class)
  - MoL composition: W_eff · x = W_base·x + ΔW_shared·x + Σ w_k·(ΔW_k·x)
    Source: architecture doc v0.3.1 Sections 7.1, 8
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Building Blocks
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope(d_head: int, max_len: int = 4096, theta: float = 10000.0):
    pos = torch.arange(max_len, dtype=torch.float32)
    freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
    angles = pos.unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    """x: [B, n_heads, L, d_head]"""
    d_half = x.shape[-1] // 2
    cos = cos[:x.shape[2], :d_half].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[2], :d_half].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., :d_half], x[..., d_half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, max_len: int = 4096):
        super().__init__()
        assert d % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        cos, sin = precompute_rope(self.d_head, max_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out(out)


class SwiGLU(nn.Module):
    def __init__(self, d: int, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            d_ff = int(d * 8 / 3)
            d_ff = ((d_ff + 63) // 64) * 64
        self.d_ff = d_ff
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ============================================================
# mHC: Manifold-Constrained Hyper-Connections
# ============================================================
#
# Full mHC layer update (arXiv:2512.24880, tokenbender/mHC):
#
#   x_{l+1} = H_res · x_l + H_post^T · F(H_pre · x_l, W_l)
#
# Three matrices per sub-layer:
#   H_res  [n, n] - doubly stochastic, mixes existing streams
#   H_pre  [n]    - non-negative, combines streams into branch input
#   H_post [n]    - non-negative, distributes branch output per-stream
#
# H_post is the critical piece: different per-stream weights on the
# branch output is what causes streams to diverge even from identical
# initialization. Without it, streams remain identical forever
# (proven: doubly stochastic × identical = identical, uniform add = no divergence).
#
# H_res implementation: KromHC (arXiv:2601.21579) — exact doubly stochastic via
# Kronecker-product factorization. For n=4=2×2: two 2×2 factors, each parameterized
# as a single scalar a ∈ (0,1) via softmax: U = a·I + (1-a)·Swap. The Kronecker
# product of two doubly stochastic matrices is doubly stochastic (Theorem 4.2).
# Replaces the approximate Sinkhorn-Knopp from original mHC. No iterations needed.
# Init: factor_logits = [0, -8] → softmax → a≈1 → U≈I → H_res = I⊗I ≈ I_4.

class KromHCResidual(nn.Module):
    """
    Exact doubly stochastic H_res for n=4 via Kronecker-product factorization.
    References: arXiv:2601.21579 (KromHC), Theorem 4.2 + Eq. 14.
    Factorization: H_res = U1 ⊗ U2, each U_k a 2×2 doubly stochastic matrix.

    For n=2: 2! = 2 permutation matrices (I, Swap).
    Each factor: U_k = a_k·I + (1-a_k)·Swap, a_k = softmax([logit_0, logit_1])[0].
    Kronecker product: (2×2) ⊗ (2×2) → 4×4, exactly doubly stochastic.

    Static variant only (input-independent). Dynamic extension not implemented here.
    """

    # The two 2×2 permutation matrices in the Birkhoff polytope for n=2
    _I2 = torch.eye(2)
    _S2 = torch.tensor([[0., 1.], [1., 0.]])

    def __init__(self):
        super().__init__()
        # Two factor logit pairs — each length-2 vector over {identity, swap}.
        # Init [0, -8]: softmax → [≈1, ≈0] → a≈1 → U≈I → H_res = I⊗I ≈ I_4.
        self.factor1_logits = nn.Parameter(torch.tensor([0.0, -8.0]))
        self.factor2_logits = nn.Parameter(torch.tensor([0.0, -8.0]))

    def forward(self):
        """Returns a 4×4 exactly doubly stochastic matrix."""
        device = self.factor1_logits.device
        dtype  = self.factor1_logits.dtype
        I2 = self._I2.to(device=device, dtype=dtype)
        S2 = self._S2.to(device=device, dtype=dtype)

        a1 = F.softmax(self.factor1_logits, dim=0)[0]  # scalar ∈ (0,1)
        a2 = F.softmax(self.factor2_logits, dim=0)[0]

        U1 = a1 * I2 + (1 - a1) * S2  # 2×2 doubly stochastic
        U2 = a2 * I2 + (1 - a2) * S2  # 2×2 doubly stochastic

        # Kronecker product: U1 ⊗ U2 = 4×4 exactly doubly stochastic
        # torch.kron computes Kronecker product natively.
        return torch.kron(U1, U2)  # [4, 4]


class HyperConnection(nn.Module):
    """
    Single mHC connection wrapping one branch (attention or FFN).

    Implements: x_{l+1} = H_res · x_l + H_post^T · F(H_pre · x_l)

    H_res is computed via KromHCResidual (arXiv:2601.21579): exact doubly stochastic
    by Kronecker-product factorization. Requires n=4 (factored as 2×2).

    H_pre and H_post use softmax (not sigmoid as in paper Eq.8), matching reference
    code tokenbender/mHC-manifold-constrained-hyper-connections.

    Args:
        n: number of streams — must be 4 (KromHC factorization is 2×2)
        d: model dimension (unused here; kept for API symmetry)
    """

    def __init__(self, n: int, d: int):
        super().__init__()
        assert n == 4, f"KromHC requires n=4 (got n={n}). Factorization is fixed as 2×2."
        self.n = n
        self.d = d

        # H_res: exact doubly stochastic via KromHC (replaces approximate Sinkhorn).
        self.krom_res = KromHCResidual()

        # H_pre: combines n streams → 1 branch input (non-negative via softmax).
        # Init: one randomly chosen stream = 0, rest = -8 → near one-hot selector.
        # Matches tokenbender/mHC H_pre_logits init (init_residual_index pattern).
        init_h_pre = torch.full((n,), -8.0)
        init_h_pre[torch.randint(n, (1,)).item()] = 0.0
        self.pre_logits = nn.Parameter(init_h_pre)

        # H_post: distributes branch output → n streams (non-negative via softmax).
        # Init: zeros → uniform softmax (1/n per stream) at start.
        # Matches tokenbender/mHC H_post_logits init.
        self.post_logits = nn.Parameter(torch.zeros(n))

    def forward(self, streams, branch_fn):
        """
        streams: [B, L, n, d]
        branch_fn: callable that takes [B, L, d] and returns [B, L, d]
        returns: [B, L, n, d]
        """
        B, L, n, d = streams.shape

        # ---- H_res: mix existing streams (exact doubly stochastic, KromHC) ----
        H_res = self.krom_res()  # [4, 4], exactly doubly stochastic
        mixed = torch.einsum("ij, bljd -> blid", H_res, streams)

        # ---- H_pre: combine streams into branch input ----
        pre_weights = F.softmax(self.pre_logits, dim=0)  # [n], non-negative, sums to 1
        branch_input = torch.einsum("n, blnd -> bld", pre_weights, streams)  # [B, L, d]

        # ---- Branch function (attention or FFN) ----
        branch_output = branch_fn(branch_input)  # [B, L, d]

        # ---- H_post: distribute branch output per-stream ----
        # softmax (not softplus): normalized to sum to 1, matching tokenbender/mHC.
        post_weights = F.softmax(self.post_logits, dim=0)  # [n], sums to 1
        # H_post^T · branch_output: each stream gets branch_output * post_weights[i]
        distributed = branch_output.unsqueeze(2) * post_weights.view(1, 1, n, 1)  # [B, L, n, d]

        # ---- Combine: H_res · x + H_post^T · F(H_pre · x) ----
        return mixed + distributed


# ============================================================
# MoL: Mixture of LoRAs
# ============================================================

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

    Q2 ablation control for MoLFFN: same base FFN + approximately equal total
    LoRA parameter count, but no routing or specialization mechanism.

    MoLFFN has 9 adapter sets (1 shared + 8 experts) × rank 8 = rank-72 equivalent.
    We use rank=64 (round number) as the matched-capacity baseline.
    """

    def __init__(self, d: int, rank: int = 64, d_ff: int = None):
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
        # Stack A/B matrices: [n_experts, d_in, rank] and [n_experts, rank, d_out].
        # Parameter names and storage are unchanged — only the forward computation changes.
        # Reduces 3 × 2 × n_experts matmul launches down to 3 × 2 batched GEMMs.
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


# ============================================================
# Transformer Block
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(self, d: int, n_heads: int, n_streams: int = 1,
                 use_mhc: bool = False, use_mol: bool = False,
                 use_single_lora: bool = False,
                 n_experts: int = 8,
                 mol_rank: int = 8, mol_top_k: int = 2,
                 d_ff: int = None,
                 max_len: int = 4096):
        super().__init__()
        self.use_mhc = use_mhc

        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        self.attn = CausalSelfAttention(d, n_heads, max_len)

        if use_mol:
            self.ffn = MoLFFN(d, n_experts=n_experts, top_k=mol_top_k, rank=mol_rank, d_ff=d_ff)
        elif use_single_lora:
            self.ffn = SingleLoRAFFN(d, rank=mol_rank, d_ff=d_ff)
        else:
            self.ffn = SwiGLU(d, d_ff=d_ff)

        if use_mhc:
            self.hc_attn = HyperConnection(n_streams, d)
            self.hc_ffn  = HyperConnection(n_streams, d)

    def forward(self, x):
        if self.use_mhc:
            return self._forward_mhc(x)
        return self._forward_standard(x)

    def _forward_standard(self, x):
        """[B, L, d] -> [B, L, d]"""
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

    def _forward_mhc(self, x):
        """
        [B, L, n, d] -> [B, L, n, d]

        Uses the full mHC update per sub-layer:
          x = H_res · x + H_post^T · F(H_pre · x)

        The HyperConnection handles H_pre (combine streams → branch input),
        H_res (mix streams), and H_post (distribute output per-stream).
        The branch function includes the pre-norm.
        """
        x = self.hc_attn(x, lambda inp: self.attn(self.norm1(inp)))
        x = self.hc_ffn(x, lambda inp: self.ffn(self.norm2(inp)))
        return x


# ============================================================
# Full Model
# ============================================================

class ToyTransformer(nn.Module):

    CONFIGS = {
        "baseline":       dict(use_mhc=False, use_mol=False, use_single_lora=False, n_streams=1),
        "baseline_wide":  dict(use_mhc=False, use_mol=False, use_single_lora=False, n_streams=1),
        "mhc":            dict(use_mhc=True,  use_mol=False, use_single_lora=False, n_streams=4),
        "mol":            dict(use_mhc=False, use_mol=True,  use_single_lora=False, n_streams=1),
        "mol_single":     dict(use_mhc=False, use_mol=False, use_single_lora=True,  n_streams=1),
        "compose":        dict(use_mhc=True,  use_mol=True,  use_single_lora=False, n_streams=4),
    }

    def __init__(self, config: str = "baseline", d: int = 256, n_layers: int = 8,
                 n_heads: int = 8, vocab_size: int = 256, max_len: int = 2048,
                 n_experts: int = 8,
                 mol_rank: int = 8, mol_top_k: int = 2, d_ff: int = None):
        super().__init__()

        cfg = self.CONFIGS[config]
        self.config_name = config
        self.use_mhc = cfg["use_mhc"]
        self.n_streams = cfg["n_streams"]
        self.d = d

        self.embed = nn.Embedding(vocab_size, d)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d=d, n_heads=n_heads, n_streams=self.n_streams,
                use_mhc=cfg["use_mhc"], use_mol=cfg["use_mol"],
                use_single_lora=cfg["use_single_lora"],
                n_experts=n_experts,
                mol_rank=mol_rank, mol_top_k=mol_top_k,
                d_ff=d_ff,
                max_len=max_len,
            )
            for _ in range(n_layers)
        ])

        self.norm_out = RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying

        # Stream collapse: softmax-normalized learned weights (Section 5.4)
        if self.use_mhc and self.n_streams > 1:
            self.stream_collapse_logits = nn.Parameter(torch.zeros(self.n_streams))

        self.apply(self._init_weights)

        # Scaled init for residual branch output projections (GPT-2 / DS-Init).
        # Matrices that write back to the residual stream are initialized with a
        # smaller std = 0.02 / sqrt(2 * n_layers) to prevent residual accumulation
        # from growing with depth. At 8 layers: std ≈ 0.005.
        residual_std = 0.02 / math.sqrt(2 * n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out.weight, mean=0.0, std=residual_std)
            ffn = block.ffn
            if hasattr(ffn, "base_down"):       # MoLFFN or SingleLoRAFFN
                nn.init.normal_(ffn.base_down.weight, mean=0.0, std=residual_std)
            elif hasattr(ffn, "down"):           # SwiGLU
                nn.init.normal_(ffn.down.weight, mean=0.0, std=residual_std)

    def _init_weights(self, m):
        """Only touches nn.Linear and nn.Embedding. LoRAAdapter and
        HyperConnection use nn.Parameter directly and are unaffected."""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x):
        """x: [B, L] -> [B, L, vocab_size]"""
        h = self.embed(x)

        if self.use_mhc and self.n_streams > 1:
            # Expand to n streams — identical at start, will diverge via H_post
            h = h.unsqueeze(2).expand(-1, -1, self.n_streams, -1).clone()

        for block in self.blocks:
            h = block(h)

        if self.use_mhc and self.n_streams > 1:
            w = F.softmax(self.stream_collapse_logits, dim=0)
            h = torch.einsum("blnd, n -> bld", h, w)

        h = self.norm_out(h)
        return self.lm_head(h)

    def get_mol_stats(self):
        stats = []
        for i, block in enumerate(self.blocks):
            if hasattr(block.ffn, "get_load_stats"):
                s = block.ffn.get_load_stats()
                if s:
                    s["layer"] = i
                    stats.append(s)
        return stats

    def reset_mol_counts(self):
        for block in self.blocks:
            if hasattr(block.ffn, "reset_counts"):
                block.ffn.reset_counts()
