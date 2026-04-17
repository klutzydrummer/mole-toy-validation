"""
Phase 1 Model: facade — imports all components, defines ToyTransformer.

Mathematical implementations live in phase1/components/:
  _shared.py              RMSNorm, RoPE, SwiGLU
  attention_rope_norms.py CausalSelfAttention
  mla_attention.py        MLACausalAttention
  diff_attention.py       DifferentialCausalAttention, DiffMLAAttention
  mhc.py                  GoMHCResidual, HyperConnection
  mol_ffn.py              LoRAAdapter, SingleLoRAFFN, MoLFFN
  transformer_block.py    TransformerBlock

Only ToyTransformer (wiring + CONFIGS) lives here.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Re-export all components so existing callsites continue to work unchanged:
#   from phase1.model import RMSNorm, TransformerBlock   ← phase2/model.py
#   from phase1.model import ToyTransformer              ← train.py, shape_check.py
from phase1.components._shared import RMSNorm, SwiGLU, apply_rope, precompute_rope  # noqa: F401
from phase1.components.attention_rope_norms import CausalSelfAttention  # noqa: F401
from phase1.components.diff_attention import (  # noqa: F401
    DifferentialCausalAttention,
    DiffMLAAttention,
)
from phase1.components.mhc import GoMHCResidual, HyperConnection  # noqa: F401
from phase1.components.mla_attention import MLACausalAttention  # noqa: F401
from phase1.components.mol_ffn import LoRAAdapter, MoLFFN, SingleLoRAFFN  # noqa: F401
from phase1.components.ngpt import (  # noqa: F401
    NGPTCausalAttention,
    NGPTDiffCausalAttention,
    NGPTDiffMLAAttention,
    NGPTMLACausalAttention,
    l2_norm,
    normalize_ngpt_weights,
)
from phase1.components.transformer_block import TransformerBlock  # noqa: F401

# ============================================================
# Full Model
# ============================================================

class ToyTransformer(nn.Module):

    CONFIGS = {
        "baseline":       dict(use_mhc=False, use_mol=False, use_single_lora=False, use_mla=False, use_diff_attn=False, use_diff_mla=False, n_streams=1),
        "baseline_wide":  dict(use_mhc=False, use_mol=False, use_single_lora=False, use_mla=False, use_diff_attn=False, use_diff_mla=False, n_streams=1),
        "mhc":            dict(use_mhc=True,  use_mol=False, use_single_lora=False, use_mla=False, use_diff_attn=False, use_diff_mla=False, n_streams=4),
        "mol":            dict(use_mhc=False, use_mol=True,  use_single_lora=False, use_mla=False, use_diff_attn=False, use_diff_mla=False, n_streams=1),
        "mol_single":     dict(use_mhc=False, use_mol=False, use_single_lora=True,  use_mla=False, use_diff_attn=False, use_diff_mla=False, n_streams=1),
        "compose":        dict(use_mhc=True,  use_mol=True,  use_single_lora=False, use_mla=False, use_diff_attn=False, use_diff_mla=False, n_streams=4),
        # MLA KV compression (arXiv:2405.04434, DeepSeek-V2)
        "mla":            dict(use_mhc=False, use_mol=False, use_single_lora=False, use_mla=True,  use_diff_attn=False, use_diff_mla=False, n_streams=1),
        # Differential attention (arXiv:2410.05258, ICLR 2025)
        "diff_attn":         dict(use_mhc=False, use_mol=False, use_single_lora=False, use_mla=False, use_diff_attn=True,  use_diff_mla=False, n_streams=1),
        # Differential attention, parameter-matched to baseline: d_ff=1240 (vs default 1408)
        # reduces FFN by ~2.06M to compensate for the extra d² Q projection per layer.
        # Total unique params: ~27.83M (baseline: ~27.80M, Δ=+33K, 0.12%). Controlled Q4.
        "diff_attn_matched": dict(use_mhc=False, use_mol=False, use_single_lora=False, use_mla=False, use_diff_attn=True,  use_diff_mla=False, n_streams=1),
        # Diff Attention + MLA KV compression (novel composition, no published precedent)
        "diff_mla":       dict(use_mhc=False, use_mol=False, use_single_lora=False, use_mla=False, use_diff_attn=False, use_diff_mla=True,  n_streams=1),
        # go-mHC compositions — Phase 1 composition roadmap (April 2026)
        "diff_mhc":       dict(use_mhc=True,  use_mol=False, use_single_lora=False, use_mla=False, use_diff_attn=True,  use_diff_mla=False, n_streams=4),
        "mla_mhc":        dict(use_mhc=True,  use_mol=False, use_single_lora=False, use_mla=True,  use_diff_attn=False, use_diff_mla=False, n_streams=4),
        "diff_mla_mhc":   dict(use_mhc=True,  use_mol=False, use_single_lora=False, use_mla=False, use_diff_attn=False, use_diff_mla=True,  n_streams=4),
        # nGPT hypersphere experiments (April 2026, arXiv:2410.01131)
        "ngpt":           dict(use_mhc=False, use_mol=False, use_single_lora=False, use_mla=False, use_diff_attn=False, use_diff_mla=False, use_ngpt=True,  n_streams=1),
        "ngpt_mla":       dict(use_mhc=False, use_mol=False, use_single_lora=False, use_mla=True,  use_diff_attn=False, use_diff_mla=False, use_ngpt=True,  n_streams=1),
        "ngpt_diff_attn": dict(use_mhc=False, use_mol=False, use_single_lora=False, use_mla=False, use_diff_attn=True,  use_diff_mla=False, use_ngpt=True,  n_streams=1),
        # nGPT + mHC compositions (April 2026)
        # A: multi-sphere — each stream pinned to S^{d-1}, per-stream α, Fréchet mean mixing
        # C: nGPT-around-sublayer — full mHC block treated as sublayer, α wraps the block
        "ngpt_mhc_a":    dict(use_mhc=True,  use_mol=False, use_single_lora=False, use_mla=False, use_diff_attn=False, use_diff_mla=False, use_ngpt=True,  n_streams=4, ngpt_mhc_variant="a"),
        "ngpt_mhc_c":    dict(use_mhc=True,  use_mol=False, use_single_lora=False, use_mla=False, use_diff_attn=False, use_diff_mla=False, use_ngpt=True,  n_streams=4, ngpt_mhc_variant="c"),
        # Study F — composition gap fills (April 2026)
        # Completes the nGPT × attention grid (baseline/mla/diff_attn already covered)
        "ngpt_diff_mla":         dict(use_mhc=False, use_mol=False, use_single_lora=False, use_mla=False, use_diff_attn=False, use_diff_mla=True,  use_ngpt=True,  n_streams=1),
        # Controlled diff_attn_matched + mHC (parameter-matched via d_ff=1240 in run_experiments.sh)
        "diff_attn_matched_mhc": dict(use_mhc=True,  use_mol=False, use_single_lora=False, use_mla=False, use_diff_attn=True,  use_diff_mla=False, n_streams=4),
        # MoL × attention cross-studies
        "mol_mla":               dict(use_mhc=False, use_mol=True,  use_single_lora=False, use_mla=True,  use_diff_attn=False, use_diff_mla=False, n_streams=1),
        "mol_diff_attn_matched": dict(use_mhc=False, use_mol=True,  use_single_lora=False, use_mla=False, use_diff_attn=True,  use_diff_mla=False, n_streams=1),
        # nGPT + MoL: unit-norm constraint makes routing = cosine similarity (direction-based)
        "ngpt_mol":              dict(use_mhc=False, use_mol=True,  use_single_lora=False, use_mla=False, use_diff_attn=False, use_diff_mla=False, use_ngpt=True,  n_streams=1),
        # Completes MoL × attention grid
        "mol_diff_mla":          dict(use_mhc=False, use_mol=True,  use_single_lora=False, use_mla=False, use_diff_attn=False, use_diff_mla=True,  n_streams=1),
        # nGPT + diff_attn_matched: controlled Q4 under unit-norm constraint (d_ff=1240 in run_experiments.sh)
        "ngpt_diff_attn_matched": dict(use_mhc=False, use_mol=False, use_single_lora=False, use_mla=False, use_diff_attn=True,  use_diff_mla=False, use_ngpt=True,  n_streams=1),
        # Three-way: nGPT + DiffMLA + go-mHC (variant "a": multi-sphere, per-stream α)
        "ngpt_diff_mla_mhc":     dict(use_mhc=True,  use_mol=False, use_single_lora=False, use_mla=False, use_diff_attn=False, use_diff_mla=True,  use_ngpt=True,  n_streams=4, ngpt_mhc_variant="a"),
        # Three-way: MoL + DiffMLA + go-mHC
        "mol_diff_mla_mhc":      dict(use_mhc=True,  use_mol=True,  use_single_lora=False, use_mla=False, use_diff_attn=False, use_diff_mla=True,  n_streams=4),
    }

    def __init__(self, config: str = "baseline", d: int = 256, n_layers: int = 8,
                 n_heads: int = 8, vocab_size: int = 256, max_len: int = 2048,
                 n_experts: int = 8,
                 mol_rank: int = 8, mol_top_k: int = 2, d_ff: int = None,
                 grad_checkpoint: bool = False):
        super().__init__()

        cfg = self.CONFIGS[config]
        self.config_name = config
        self.use_mhc = cfg["use_mhc"]
        self.use_ngpt = cfg.get("use_ngpt", False)
        self.n_streams = cfg["n_streams"]
        self.d = d
        self.grad_checkpoint = grad_checkpoint

        self.embed = nn.Embedding(vocab_size, d)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d=d, n_heads=n_heads, n_streams=self.n_streams,
                use_mhc=cfg["use_mhc"], use_mol=cfg["use_mol"],
                use_single_lora=cfg["use_single_lora"],
                use_mla=cfg["use_mla"],
                use_diff_attn=cfg["use_diff_attn"],
                use_diff_mla=cfg["use_diff_mla"],
                use_ngpt=self.use_ngpt,
                ngpt_mhc_variant=cfg.get("ngpt_mhc_variant", None),
                n_experts=n_experts,
                mol_rank=mol_rank, mol_top_k=mol_top_k,
                d_ff=d_ff,
                layer_idx=i,
                max_len=max_len,
            )
            for i in range(n_layers)
        ])

        self.norm_out = RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying

        # Stream collapse: softmax-normalized learned weights (Section 5.4)
        if self.use_mhc and self.n_streams > 1:
            self.stream_collapse_logits = nn.Parameter(torch.zeros(self.n_streams))

        # nGPT logit scale s_z: hidden states are unit-norm so logits are raw cosine
        # similarities. s_z (init √d ≈ 22.6) rescales to a useful softmax temperature.
        # Reference: arXiv:2410.01131, Section 2.5.
        if self.use_ngpt:
            self.ngpt_s_z = nn.Parameter(torch.tensor(math.sqrt(d)))

        self.apply(self._init_weights)

        # Scaled init for residual branch output projections (GPT-2 / DS-Init).
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

        # nGPT: embed outputs land on the hypersphere from the first block onward.
        # normalize_ngpt_weights keeps embed.weight unit-normed after each optimizer step,
        # but we normalize the lookup result here to handle the first forward pass and
        # any residual floating-point drift.
        if self.use_ngpt:
            h = l2_norm(h)

        if self.use_mhc and self.n_streams > 1:
            # Expand to n streams — identical at start, will diverge via H_post
            h = h.unsqueeze(2).expand(-1, -1, self.n_streams, -1).clone()

        for block in self.blocks:
            if self.grad_checkpoint and self.training:
                h = checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)

        if self.use_mhc and self.n_streams > 1:
            w = F.softmax(self.stream_collapse_logits, dim=0)
            h = torch.einsum("blnd, n -> bld", h, w)

        h = self.norm_out(h)
        logits = self.lm_head(h)
        # nGPT: logits are cosine similarities (in [−1, 1]); scale by s_z to restore
        # a useful softmax temperature. s_z is learnable, init √d ≈ 22.6.
        if self.use_ngpt:
            logits = logits * self.ngpt_s_z
        return logits

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
