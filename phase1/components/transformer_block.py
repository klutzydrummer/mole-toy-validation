"""
TransformerBlock — wires attention, FFN, and optional mHC connections.

This file is wiring code, not a mathematical implementation.
No component spec — not independently hashed by verify.py.
"""


import torch.nn as nn

from phase1.components._shared import RMSNorm, SwiGLU
from phase1.components.attention_rope_norms import CausalSelfAttention
from phase1.components.diff_attention import DifferentialCausalAttention, DiffMLAAttention
from phase1.components.mhc import HyperConnection
from phase1.components.mla_attention import MLACausalAttention
from phase1.components.mol_ffn import MoLFFN, SingleLoRAFFN


class TransformerBlock(nn.Module):

    def __init__(self, d: int, n_heads: int, n_streams: int = 1,
                 use_mhc: bool = False, use_mol: bool = False,
                 use_single_lora: bool = False,
                 use_mla: bool = False,
                 use_diff_attn: bool = False, use_diff_mla: bool = False,
                 n_experts: int = 8,
                 mol_rank: int = 8, mol_top_k: int = 2,
                 d_ff: int = None,
                 layer_idx: int = 0,
                 max_len: int = 4096):
        super().__init__()
        self.use_mhc = use_mhc

        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)

        if use_diff_mla:
            self.attn = DiffMLAAttention(d, n_heads, layer_idx=layer_idx, max_len=max_len)
        elif use_diff_attn:
            self.attn = DifferentialCausalAttention(d, n_heads, layer_idx=layer_idx, max_len=max_len)
        elif use_mla:
            self.attn = MLACausalAttention(d, n_heads, max_len=max_len)
        else:
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
