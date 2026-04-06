"""
Zone E: Pluggable encoder architectures for the outer encoder study.

Tracked by:
  - references/components/zone_ed_pipeline.md  (encoder role in the pipeline)

Sources: references/sources/papers/hnet_2507.07955.md,
         references/sources/code/hnet_boundary.py
"""

import torch
import torch.nn as nn

from phase1.components._shared import RMSNorm
from phase1.components.transformer_block import TransformerBlock
from phase2.components.causal_recurrence import CausalRecurrenceLayer


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
        # No norm_out: each CausalRecurrenceLayer already ends with self.norm(out_proj(out)).
        # CRLEncoder also has no extra outer norm — both variants are consistent.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d] → [B, L, d]"""
        for rec in self.recurrence:
            x = rec(x)
        return x


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
