"""
TransformerBlock — wires attention, FFN, and optional mHC connections.

This file is wiring code, not a mathematical implementation.
No component spec — not independently hashed by verify.py.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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
                 ngpt_mhc_variant: str = None,
                 n_experts: int = 8,
                 mol_rank: int = 8, mol_top_k: int = 2,
                 d_ff: int = None,
                 layer_idx: int = 0,
                 max_len: int = 4096):
        super().__init__()
        self.use_mhc = use_mhc
        self.use_ngpt = use_ngpt
        self.ngpt_mhc_variant = ngpt_mhc_variant  # "a" or "c"; None for non-composition configs
        self.d = d
        self.n_streams = n_streams

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
        # Standard nGPT: per-dim [d]. Multi-sphere mHC: per-stream-per-dim [n_streams, d].
        # Effective step ≈ 0.05/√d at start in both cases.
        if use_ngpt:
            if use_mhc:
                # Each stream has its own eigenlearning rate vector
                self.alpha_a = nn.Parameter(torch.full((n_streams, d), 0.05))
                self.alpha_m = nn.Parameter(torch.full((n_streams, d), 0.05))
            else:
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
        if self.use_mhc and self.use_ngpt:
            if self.ngpt_mhc_variant == "a":
                return self._forward_ngpt_mhc_a(x)
            return self._forward_ngpt_mhc_c(x)
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

    def _ngpt_mhc_a_step(self, x, hc, branch_fn, alpha, alpha_scale):
        """
        Single sub-layer update for Option A (multi-sphere mHC).
        x: [B, L, n, d] — each stream on S^{d-1}

        Update rule:
          1. H_res mixing + per-stream L2Norm (Fréchet mean approx on sphere)
          2. Branch output L2Normed to unit sphere
          3. Per-stream α interpolation toward branch output, weighted by H_post
        """
        B, L, n, d = x.shape

        # H_res: mix streams → renorm each stream back to sphere
        H_res_mat = hc.go_res(x)                                            # [B, L, n, n]
        mixed = torch.einsum("blij,bljd->blid", H_res_mat, x)              # [B, L, n, d]
        mixed = l2_norm(mixed)                                              # per-stream renorm

        # H_pre: combine streams → branch input (weighted sum; not on sphere → norm in branch_fn)
        pre_w = F.softmax(hc.pre_logits, dim=0)                            # [n]
        branch_in = torch.einsum("n,blnd->bld", pre_w, x)                 # [B, L, d]
        h = l2_norm(branch_fn(branch_in))                                  # [B, L, d] on sphere

        # H_post weights: per-stream participation in the update
        post_w = F.softmax(hc.post_logits, dim=0)                          # [n]

        # Per-stream α interp: x_i = L2Norm(mixed_i + α_i·post_i·(h − mixed_i))
        # alpha: [n, d]; post_w: [n] → alpha_eff: [n, d]
        alpha_eff = (alpha * alpha_scale) * post_w.unsqueeze(-1)            # [n, d]
        h_streams = h.unsqueeze(2)                                         # [B, L, 1, d]
        return l2_norm(mixed + alpha_eff * (h_streams - mixed))            # [B, L, n, d]

    def _forward_ngpt_mhc_a(self, x):
        """
        Option A: multi-sphere mHC.
        [B, L, n, d] -> [B, L, n, d], each stream stays on S^{d-1}.

        Per-stream α vectors [n, d] give each stream its own eigenlearning rate.
        H_res mixing leaves the sphere; L2Norm approximates the Fréchet mean.
        Branch output is L2Normed before the per-stream α interpolation step.
        norm1/norm2 are applied to branch_input (H_pre·x is not on the sphere).
        """
        alpha_scale = 1.0 / math.sqrt(self.d)
        x = self._ngpt_mhc_a_step(
            x, self.hc_attn, lambda inp: self.attn(self.norm1(inp)),
            self.alpha_a, alpha_scale,
        )
        x = self._ngpt_mhc_a_step(
            x, self.hc_ffn, lambda inp: self.ffn(self.norm2(inp)),
            self.alpha_m, alpha_scale,
        )
        return x

    def _forward_ngpt_mhc_c(self, x):
        """
        Option C: nGPT-around-sublayer with renorm after H_res.
        [B, L, n, d] -> [B, L, n, d]

        The full mHC block (H_res·x + H_post^T·L2Norm(F(H_pre·x))) is treated as
        the "sublayer output". The result is L2Normed then used in a per-stream
        α interpolation, just as in standard nGPT but applied per-stream.

        H_res mixing leaves the sphere; L2Norm after the full block corrects this.
        norm1/norm2 applied inside the branch (H_pre·x is not on sphere).
        """
        alpha_scale = 1.0 / math.sqrt(self.d)
        alpha_a = self.alpha_a * alpha_scale   # [n_streams, d]
        alpha_m = self.alpha_m * alpha_scale

        x_new = self.hc_attn(x, lambda inp: l2_norm(self.attn(self.norm1(inp))))
        h_a = l2_norm(x_new)                   # full mHC output → unit sphere
        x = l2_norm(x + alpha_a * (h_a - x))

        x_new = self.hc_ffn(x, lambda inp: l2_norm(self.ffn(self.norm2(inp))))
        h_m = l2_norm(x_new)
        x = l2_norm(x + alpha_m * (h_m - x))
        return x
