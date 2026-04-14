"""
Multi-Head Latent Attention (MLA) — DeepSeek-V2 (arXiv:2405.04434).
Decoupled RoPE follows Equations 14–19 of arXiv:2405.04434.

Spec: references/components/mla_attention.md
Sources: references/sources/papers/mla_deepseek_v2_2405.04434.md,
         references/sources/code/mla_attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from phase1.components._shared import RMSNorm, apply_rope, precompute_rope


class MLACausalAttention(nn.Module):
    """
    Multi-Head Latent Attention (arXiv:2405.04434, DeepSeek-V2).

    KV compression via low-rank latent (Eq. 9–11); Q via separate latent (Eq. 12–13).
    Decoupled RoPE (Eq. 14–19): content and positional paths concatenated per head.
    RMSNorm applied at both latent bottlenecks for training stability.

    KV compression path:
      c_KV  = RMSNorm(x @ W_DKV)                   W_DKV ∈ ℝ^{d × d_c}
      K_C   = c_KV @ W_UK                           W_UK  ∈ ℝ^{d_c × n_h·d_h}  (content, no RoPE)
      V     = c_KV @ W_UV                           W_UV  ∈ ℝ^{d_c × n_h·d_h}

    Q compression path:
      c_Q   = RMSNorm(x @ W_DQ)                     W_DQ  ∈ ℝ^{d × d_c_q}
      Q_C   = c_Q @ W_UQ                            W_UQ  ∈ ℝ^{d_c_q × n_h·d_h} (content, no RoPE)

    Decoupled RoPE (Eq. 14–19):
      Q_R   = RoPE(c_Q @ W_QR)                      W_QR  ∈ ℝ^{d_c_q × n_h·d_h_R} (per-head)
      K_R   = RoPE(x @ W_KR)                        W_KR  ∈ ℝ^{d × d_h_R}  (single shared key)
      Q     = [Q_C ; Q_R]    per head: d_h + d_h_R
      K     = [K_C ; K_R]    per head: d_h + d_h_R

    where d_h_R = d_h // 2 (= 32 at d=512).

    Defaults at d=512, n_heads=8:
      d_c   = d // 2 = 256   (KV latent, 2× compression; empirically safe floor with decoupled RoPE)
      d_c_q = d // 2 = 256   (Q latent)
      d_h_R = d_h // 2 = 32  (decoupled RoPE per-head dim)
    """

    def __init__(self, d: int, n_heads: int,
                 d_c: int = None, d_c_q: int = None, max_len: int = 4096):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d // n_heads
        self.d_h_R   = self.d_head // 2  # decoupled RoPE per-head dim (Eq. 14–19)
        self.d = d

        if d_c is None:
            d_c = d // 2   # KV latent dim (256 at d=512; 2× compression)
        if d_c_q is None:
            d_c_q = d // 2  # Q latent dim (256 at d=512)
        self.d_c, self.d_c_q = d_c, d_c_q

        # KV compression (Eq. 9–11)
        self.W_DKV  = nn.Linear(d, d_c, bias=False)
        self.kv_norm = RMSNorm(d_c)
        self.W_UK   = nn.Linear(d_c, n_heads * self.d_head, bias=False)
        self.W_UV   = nn.Linear(d_c, n_heads * self.d_head, bias=False)

        # Q compression (Eq. 12–13)
        self.W_DQ   = nn.Linear(d, d_c_q, bias=False)
        self.q_norm  = RMSNorm(d_c_q)
        self.W_UQ   = nn.Linear(d_c_q, n_heads * self.d_head, bias=False)

        # Decoupled RoPE projections (Eq. 14–15)
        # W_QR: per-head RoPE query from Q latent
        self.W_QR   = nn.Linear(d_c_q, n_heads * self.d_h_R, bias=False)
        # W_KR: single shared RoPE key directly from input (not from latent)
        self.W_KR   = nn.Linear(d, self.d_h_R, bias=False)

        self.out = nn.Linear(d, d, bias=False)

        # RoPE buffers for the decoupled positional dimension only (d_h_R, not d_head)
        cos, sin = precompute_rope(self.d_h_R, max_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x):
        B, L, D = x.shape
        nh, dh, d_h_R = self.n_heads, self.d_head, self.d_h_R

        # KV path — compress, norm, expand (Eq. 9–11)
        c_kv = self.kv_norm(self.W_DKV(x))                                    # [B, L, d_c]
        k_c  = self.W_UK(c_kv).reshape(B, L, nh, dh).transpose(1, 2)         # [B, nh, L, dh]  content
        v    = self.W_UV(c_kv).reshape(B, L, nh, dh).transpose(1, 2)         # [B, nh, L, dh]

        # Q path — compress, norm, expand (Eq. 12–13)
        c_q  = self.q_norm(self.W_DQ(x))                                      # [B, L, d_c_q]
        q_c  = self.W_UQ(c_q).reshape(B, L, nh, dh).transpose(1, 2)          # [B, nh, L, dh]  content

        # Decoupled RoPE (Eq. 14–15)
        # q_rope: per-head positional query, projected from Q latent
        q_rope = apply_rope(
            self.W_QR(c_q).reshape(B, L, nh, d_h_R).transpose(1, 2),
            self.rope_cos, self.rope_sin
        )  # [B, nh, L, d_h_R]

        # k_rope: single shared positional key projected directly from input (Eq. 15)
        # Shared across heads (single vector broadcast to all nh heads)
        k_rope = apply_rope(
            self.W_KR(x).reshape(B, L, 1, d_h_R).transpose(1, 2),
            self.rope_cos, self.rope_sin
        ).expand(B, nh, L, d_h_R)  # [B, nh, L, d_h_R]

        # Concatenate content and positional components (Eq. 16–17)
        q = torch.cat([q_c, q_rope], dim=-1)  # [B, nh, L, dh + d_h_R]
        k = torch.cat([k_c, k_rope], dim=-1)  # [B, nh, L, dh + d_h_R]

        # Scaled dot-product attention (Eq. 18); denominator = sqrt(dh + d_h_R) implicit in SDPA
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)         # [B, nh, L, dh]
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out(out)
