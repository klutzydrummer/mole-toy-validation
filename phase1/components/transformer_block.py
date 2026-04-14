"""
TransformerBlock — wires attention, FFN, and optional mHC connections.

This file is wiring code, not a mathematical implementation.
No component spec — not independently hashed by verify.py.
"""

import math

import torch.nn as nn

from phase1.components._shared import RMSNorm, SwiGLU
from phase1.components.attention_rope_norms import CausalSelfAttention
from phase1.components.diff_attention import DifferentialCausalAttention, DiffMLAAttention
from phase1.components.mhc import HyperConnection
from phase1.components.mla_attention import MLACausalAttention
from phase1.components.mol_ffn import MoLFFN, SingleLoRAFFN
from phase1.components.ngpt import (  # noqa: F401
    NGPTCausalAttention,
    NGPTDiffCausalAttention,
    NGPTMLACausalAttention,
    l2_norm,
)


class TransformerBlock(nn.Module):

    def __init__(self, d: int, n_heads: int, n_streams: int = 1,
                 use_mhc: bool = False, use_mol: bool = False,
                 use_single_lora: bool = False,
                 use_mla: bool = False,
                 use_diff_attn: bool = False, use_diff_mla: bool = False,
                 use_ngpt: bool = False,
                 n_experts: int = 8,
                 mol_rank: int = 8, mol_top_k: int = 2,
                 d_ff: int = None,
                 layer_idx: int = 0,
                 max_len: int = 4096):
        super().__init__()
        self.use_mhc = use_mhc
        self.use_ngpt = use_ngpt
        self.d = d

        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)

        # Attention class selection: nGPT variants take priority for ngpt configs.
        # use_diff_mla has no nGPT variant (not in initial roadmap).
        if use_diff_mla:
            self.attn = DiffMLAAttention(d, n_heads, layer_idx=layer_idx, max_len=max_len)
        elif use_ngpt and use_mla:
            self.attn = NGPTMLACausalAttention(d, n_heads, max_len=max_len)
        elif use_ngpt and use_diff_attn:
            self.attn = NGPTDiffCausalAttention(d, n_heads, layer_idx=layer_idx, max_len=max_len)
        elif use_diff_attn:
            self.attn = DifferentialCausalAttention(d, n_heads, layer_idx=layer_idx, max_len=max_len)
        elif use_mla:
            self.attn = MLACausalAttention(d, n_heads, max_len=max_len)
        elif use_ngpt:
            self.attn = NGPTCausalAttention(d, n_heads, max_len=max_len)
        else:
            self.attn = CausalSelfAttention(d, n_heads, max_len)

        # Alpha parameters for nGPT spherical interpolation (Eq. 10–11, arXiv:2410.01131).
        # Per-dim learnable vectors, init 0.05. Effective step ≈ 0.05/√d at start.
        if use_ngpt:
            import torch
            self.alpha_a = nn.Parameter(torch.full((d,), 0.05))
            self.alpha_m = nn.Parameter(torch.full((d,), 0.05))

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
        if self.use_ngpt:
            return self._forward_ngpt(x)
        return self._forward_standard(x)

    def _forward_standard(self, x):
        """[B, L, d] -> [B, L, d]"""
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

    def _forward_ngpt(self, x):
        """
        [B, L, d] -> [B, L, d]

        nGPT spherical update per sub-layer (arXiv:2410.01131, Eq. 10–11):
          h_sub = L2Norm(sublayer(x))
          x     = L2Norm(x + α ⊙ (h_sub − x))

        α is per-dim, init 0.05, applied with scale factor 1/√d.
        Effective init step ≈ 0.05/√512 ≈ 0.0022 (0.2% interpolation).

        norm1/norm2 are NOT applied — x is already unit-norm on the hypersphere.
        Q/K normalization and inverted attention scale are handled inside the
        NGPT attention class (NGPTCausalAttention et al.).
        """
        alpha_scale = 1.0 / math.sqrt(self.d)
        alpha_a = self.alpha_a * alpha_scale   # [d], effective ≈ 0.0022 at init
        alpha_m = self.alpha_m * alpha_scale

        h_a = l2_norm(self.attn(x))
        x   = l2_norm(x + alpha_a * (h_a - x))
        h_m = l2_norm(self.ffn(x))
        x   = l2_norm(x + alpha_m * (h_m - x))
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
