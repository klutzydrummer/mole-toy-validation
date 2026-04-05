"""
Baseline causal self-attention with RoPE positional encoding.

Spec: references/components/attention_rope_norms.md
Sources: references/sources/papers/rope_2104.09864.md,
         references/sources/papers/rmsnorm_1910.07467.md,
         references/sources/papers/swiglu_2002.05202.md
"""

import torch.nn as nn
import torch.nn.functional as F

from phase1.components._shared import apply_rope, precompute_rope


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
