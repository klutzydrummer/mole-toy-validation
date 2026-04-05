"""
Differential Attention V2 (January 2026) and DiffMLA composition.

Spec: references/components/diff_attention.md
Sources: references/sources/papers/diff_attn_v1_2410.05258.md,
         references/sources/papers/diff_attn_v2_2026_01.md,
         references/sources/papers/mla_deepseek_v2_2405.04434.md,
         references/sources/code/mla_attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from phase1.components._shared import apply_rope, precompute_rope


class DifferentialCausalAttention(nn.Module):
    """
    Differential Transformer V2 (Microsoft, January 2026).
    https://huggingface.co/blog/microsoft/diff-attn-v2

    Changes from V1 (arXiv:2410.05258):
      1. Q is doubled (2h query heads), KV stays standard (h heads) — GQA-style.
         Adjacent query pairs (0,1), (2,3), ... share the same K and V.
         Eliminates custom kernels; V is no longer double-wide.
      2. No per-head RMSNorm — at large seq_len, V1 norm amplified by ~√n (≈90× at
         seq_len=8192), causing gradient instability. Simply removed in V2.
      3. λ is token-specific via sigmoid projection — sigmoid(W_λ · x) per token
         per head, replacing the static exp reparameterization of V1. Bounds context
         RMS in (0, √2), allowing attention sinks to be eliminated.

    Layout:
      W_Q:      d → 2·n_heads·d_head = 2d   (doubled query heads)
      W_K:      d → n_heads·d_head   = d
      W_V:      d → n_heads·d_head   = d
      W_lambda: d → n_heads               (tiny; token-specific λ projection)
      W_O:      d → d

    Parameter count: ~5d² vs baseline 4d² (extra d² for doubled Q).
    """

    def __init__(self, d: int, n_heads: int, layer_idx: int = 0, max_len: int = 4096):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d // n_heads
        self.d = d

        # Doubled Q projection; standard K and V
        self.W_Q = nn.Linear(d, 2 * d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)

        # Token-specific λ: projected per-head scalar, bounded by sigmoid.
        # bias=True so sigmoid(0) = 0.5 at init — reasonable starting λ.
        self.W_lambda = nn.Linear(d, n_heads, bias=True)

        self.out = nn.Linear(d, d, bias=False)

        cos, sin = precompute_rope(self.d_head, max_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x):
        B, L, D = x.shape
        nh, dh = self.n_heads, self.d_head

        # Q: [B, 2h, L, dh]  K,V: [B, h, L, dh]
        q = self.W_Q(x).reshape(B, L, 2 * nh, dh).transpose(1, 2)
        k = self.W_K(x).reshape(B, L, nh, dh).transpose(1, 2)
        v = self.W_V(x).reshape(B, L, nh, dh).transpose(1, 2)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # Repeat each KV head twice so SDPA sees matched head counts.
        # repeat_interleave(2): [h0,h1,...] → [h0,h0,h1,h1,...] matching Q pairs.
        k_rep = k.repeat_interleave(2, dim=1)   # [B, 2h, L, dh]
        v_rep = v.repeat_interleave(2, dim=1)

        # Single causal attention pass for all 2h query heads
        attn = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)  # [B, 2h, L, dh]

        # Split into even (q1) and odd (q2) heads — pairs share K,V
        attn1 = attn[:, 0::2]   # [B, h, L, dh]
        attn2 = attn[:, 1::2]

        # Token-specific λ: [B, L, h] → [B, h, L, 1] for broadcasting
        lam = torch.sigmoid(self.W_lambda(x)).transpose(1, 2).unsqueeze(-1)

        out = (attn1 - lam * attn2).transpose(1, 2).reshape(B, L, D)
        return self.out(out)


class DiffMLAAttention(nn.Module):
    """
    Differential Attention V2 + MLA KV compression — novel composition.

    MLA component (arXiv:2405.04434, DeepSeek-V2):
      c_KV = x @ W_DKV                [B, L, d_c]
      K    = c_KV @ W_UK              [B, L, n·dh]  — single K (shared by Q pair)
      V    = c_KV @ W_UV              [B, L, n·dh]  — standard width (V2: not doubled)

    Q with low-rank latent, doubled for Diff V2 GQA pairing:
      c_Q  = x @ W_DQ                 [B, L, d_c_q]
      Q    = c_Q @ W_UQ               [B, L, 2·n·dh]  — 2h query heads

    Diff V2 λ: token-specific sigmoid projection (not static exp reparameterization).

    RoPE applied to all Q heads and K at full d_head.
    No per-head RMSNorm (V2 change: removes gradient instability at scale).

    KV compression ratio: d_c / d_model (default 1/4 at toy scale).
    """

    def __init__(self, d: int, n_heads: int, layer_idx: int = 0,
                 d_c: int = None, d_c_q: int = None, max_len: int = 4096):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d // n_heads
        self.d = d

        if d_c is None:
            d_c = d // 4    # KV latent dim — 128 at d=512
        if d_c_q is None:
            d_c_q = d // 2  # Q latent dim  — 256 at d=512
        self.d_c, self.d_c_q = d_c, d_c_q

        nh, dh = n_heads, self.d_head

        # MLA KV: single K and V (V2: V no longer doubled in width)
        self.W_DKV = nn.Linear(d, d_c,      bias=False)
        self.W_UK  = nn.Linear(d_c, nh * dh, bias=False)
        self.W_UV  = nn.Linear(d_c, nh * dh, bias=False)

        # MLA Q: doubled heads for Diff V2 GQA pairing
        self.W_DQ  = nn.Linear(d, d_c_q,          bias=False)
        self.W_UQ  = nn.Linear(d_c_q, 2 * nh * dh, bias=False)

        # Token-specific λ projection (Diff V2)
        self.W_lambda = nn.Linear(d, nh, bias=True)

        self.out = nn.Linear(d, d, bias=False)

        cos, sin = precompute_rope(dh, max_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x):
        B, L, D = x.shape
        nh, dh = self.n_heads, self.d_head

        # KV path: compress → K [B, nh, L, dh], V [B, nh, L, dh]
        c_kv = self.W_DKV(x)
        k = self.W_UK(c_kv).reshape(B, L, nh, dh).transpose(1, 2)
        v = self.W_UV(c_kv).reshape(B, L, nh, dh).transpose(1, 2)

        # Q path: compress → Q [B, 2·nh, L, dh]
        c_q = self.W_DQ(x)
        q   = self.W_UQ(c_q).reshape(B, L, 2 * nh, dh).transpose(1, 2)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # Repeat KV heads to match doubled Q (GQA pairing)
        k_rep = k.repeat_interleave(2, dim=1)  # [B, 2·nh, L, dh]
        v_rep = v.repeat_interleave(2, dim=1)

        attn = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)

        attn1 = attn[:, 0::2]  # [B, nh, L, dh]
        attn2 = attn[:, 1::2]

        lam = torch.sigmoid(self.W_lambda(x)).transpose(1, 2).unsqueeze(-1)  # [B, nh, L, 1]

        out = (attn1 - lam * attn2).transpose(1, 2).reshape(B, L, D)
        return self.out(out)
