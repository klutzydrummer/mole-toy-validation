"""
Multi-Head Latent Attention (MLA) — DeepSeek-V2 (arXiv:2405.04434).

Spec: references/components/mla_attention.md
Sources: references/sources/papers/mla_deepseek_v2_2405.04434.md,
         references/sources/code/mla_attention.py
"""

import torch.nn as nn
import torch.nn.functional as F

from phase1.components._shared import apply_rope, precompute_rope


class MLACausalAttention(nn.Module):
    """
    Multi-Head Latent Attention (arXiv:2405.04434, DeepSeek-V2).

    KV compression via low-rank latent — at toy scale (d=512) the latent is
    d_c=128 (1/4 of d_model), giving a 4× KV parameter reduction vs standard MHA.

      c_KV = x @ W_DKV                    W_DKV ∈ ℝ^{d × d_c}
      K    = c_KV @ W_UK                  W_UK  ∈ ℝ^{d_c × n_h·d_h}
      V    = c_KV @ W_UV                  W_UV  ∈ ℝ^{d_c × n_h·d_h}

    Q uses a separate low-rank latent (d_c_q=256 at d=512):
      c_Q  = x @ W_DQ                     W_DQ  ∈ ℝ^{d × d_c_q}
      Q    = c_Q  @ W_UQ                  W_UQ  ∈ ℝ^{d_c_q × n_h·d_h}

    RoPE applied to Q and K at full d_head. Standard causal softmax attention.

    Note: decoupled RoPE (separate rope dim for K) is omitted here — the
    KV-cache absorption trick it enables is not the focus at toy scale.
    """

    def __init__(self, d: int, n_heads: int,
                 d_c: int = None, d_c_q: int = None, max_len: int = 4096):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d // n_heads
        self.d = d

        if d_c is None:
            d_c = d // 4   # KV latent dim (128 at d=512)
        if d_c_q is None:
            d_c_q = d // 2  # Q latent dim (256 at d=512)
        self.d_c, self.d_c_q = d_c, d_c_q

        # KV compression
        self.W_DKV = nn.Linear(d, d_c, bias=False)
        self.W_UK  = nn.Linear(d_c, n_heads * self.d_head, bias=False)
        self.W_UV  = nn.Linear(d_c, n_heads * self.d_head, bias=False)

        # Q compression
        self.W_DQ  = nn.Linear(d, d_c_q, bias=False)
        self.W_UQ  = nn.Linear(d_c_q, n_heads * self.d_head, bias=False)

        self.out = nn.Linear(d, d, bias=False)

        cos, sin = precompute_rope(self.d_head, max_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x):
        B, L, D = x.shape
        nh, dh = self.n_heads, self.d_head

        # KV path
        c_kv = self.W_DKV(x)
        k = self.W_UK(c_kv).reshape(B, L, nh, dh).transpose(1, 2)  # [B, nh, L, dh]
        v = self.W_UV(c_kv).reshape(B, L, nh, dh).transpose(1, 2)

        # Q path
        c_q = self.W_DQ(x)
        q = self.W_UQ(c_q).reshape(B, L, nh, dh).transpose(1, 2)   # [B, nh, L, dh]

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out(out)
