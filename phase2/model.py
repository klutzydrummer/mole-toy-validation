"""
Phase 2 Model: facade — imports all components, defines OuterModel.

Mathematical implementations live in phase2/components/:
  causal_recurrence.py  _parallel_scan, CausalRecurrenceLayer
  zone_e.py             CRLEncoder, CRLEncoderFull, TransformerEncoder,
                        DiffAttnEncoder, MLAEncoder, IdentityEncoder
  boundary_router.py    BoundaryRouter
  zone_d.py             SimpleDecoder

Only OuterModel (wiring + CONFIGS) lives here.

Architecture:
  embed(x) [B, L, d]
    → ZoneE (pluggable encoder) → encoder_out [B, L, d]
    → BoundaryRouter (threshold p > 0.5) → concept_tokens [B, M_max, d] (padded)
                                          → boundary_idx  [B, M_max]    (padded with 0)
                                          → concept_mask  [B, M_max]    (True = valid)
    → SimpleDecoder (H-Net Eq. 5–9):
        → EMA smooth over M concept tokens
        → plug-back via cumsum(boundary_mask) - 1 → [B, L, d]   (H-Net Eq. 8)
        → confidence scoring + STE scaling                        (H-Net Eq. 6–7–9)
        → residual_proj (plain linear, near-zero init)            (H-Net Eq. 3)
        → lm_head → logits [B, L, vocab]

Key correctness invariants:
  - CausalRecurrenceLayer: sqrt(1 - a_t²) normalization (Griffin arXiv:2402.19427)
  - BoundaryRouter: threshold p > 0.5, NOT topk. Variable M per sequence, padded to M_max.
  - boundary_probs[:, 0] == 1.0 for cosine_rule and learned_e2e
  - SimpleDecoder EMA: p_at_bounds.clamp(min=0.1) prevents EMA collapse
  - STE: ste(c_t) = c_t + stopgradient(1 - c_t) — Eq. 7; = 1.0 in forward pass
  - residual_proj initialized near-zero (weight=0, no bias) — H-Net warm-start (Eq. 3)
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Re-export so existing callsites work: from phase2.model import OuterModel
# Also re-export phase1 primitives used by downstream code
from phase1.model import RMSNorm, TransformerBlock  # noqa: F401
from phase2.components.boundary_router import BoundaryRouter  # noqa: F401
from phase2.components.causal_recurrence import CausalRecurrenceLayer, _parallel_scan  # noqa: F401
from phase2.components.zone_d import SimpleDecoder  # noqa: F401
from phase2.components.zone_e import (  # noqa: F401
    CRLEncoder,
    CRLEncoderFull,
    DiffAttnEncoder,
    IdentityEncoder,
    MLAEncoder,
    TransformerEncoder,
)

# ============================================================
# OuterModel
# ============================================================

class OuterModel(nn.Module):
    """
    Outer encoder study model: pluggable ZoneE + threshold BoundaryRouter + SimpleDecoder.

    Forward returns (logits [B, L, vocab], boundary_probs [B, L], compression_ratio scalar).
    """

    CONFIGS = {
        # ── Core study configs ────────────────────────────────────────────────
        "outer_crl":              dict(encoder="crl",          router="cosine_rule"),
        "outer_crl_learned":      dict(encoder="crl",          router="learned_e2e"),
        "outer_crl_full":         dict(encoder="crl_full",     router="cosine_rule"),
        "outer_crl_full_learned": dict(encoder="crl_full",     router="learned_e2e"),
        "outer_transformer":      dict(encoder="transformer",  router="learned_e2e"),
        "outer_diff_attn":        dict(encoder="diff_attn",    router="learned_e2e"),
        "outer_mla":              dict(encoder="mla",          router="learned_e2e"),
        "outer_strided":          dict(encoder="identity",     router="fixed_stride"),
        # ── Ablations ─────────────────────────────────────────────────────────
        "outer_crl_learned_noste": dict(encoder="crl",         router="learned_e2e",
                                        use_ste=False),
        "outer_crl_r2":            dict(encoder="crl",         router="cosine_rule",
                                        target_rate=0.5),
        # Encoder param counts at d=512 (for cross-config interpretation):
        #   outer_crl / outer_crl_learned          CRLEncoder (d//4=128 bottleneck):  ~282K
        #   outer_crl_full / outer_crl_full_learned CRLEncoderFull (d=512, no proj):  ~1.57M
        #   outer_transformer                       TransformerEncoder (4×d=512):     ~12.8M
        #   outer_diff_attn                         DiffAttnEncoder (4×d=512):        ~13.9M
        #   outer_mla                               MLAEncoder (4×d=512):             ~11.5M
        # These are architecture-vs-architecture comparisons, NOT param-matched.
    }

    def __init__(
        self,
        config:         str   = "outer_crl",
        d:              int   = 512,
        n_layers:       int   = 8,       # kept for API compatibility; unused in Phase 2 outer
        n_heads:        int   = 8,
        vocab_size:     int   = 4096,
        seq_len:        int   = 256,
        n_layers_outer: int   = 4,       # transformer encoder depth
        target_rate:    float = 0.25,    # soft sparsity target for loss_comp
    ):
        super().__init__()
        if config not in self.CONFIGS:
            raise ValueError(f"Unknown config '{config}'. Valid: {list(self.CONFIGS)}")

        cfg = self.CONFIGS[config]
        self.config_name = config
        # Config can override target_rate (e.g. outer_crl_r2 uses 0.5)
        self.target_rate = cfg.get("target_rate", target_rate)
        self.use_ste     = cfg.get("use_ste", True)
        self.d           = d

        # Shared embedding (weight-tied to lm_head)
        self.embed = nn.Embedding(vocab_size, d)

        # Pluggable encoder
        encoder_type = cfg["encoder"]
        if encoder_type == "crl":
            self.encoder = CRLEncoder(d=d)
        elif encoder_type == "crl_full":
            self.encoder = CRLEncoderFull(d=d)
        elif encoder_type == "transformer":
            self.encoder = TransformerEncoder(d=d, n_heads=n_heads,
                                              n_layers_outer=n_layers_outer,
                                              max_len=seq_len)
        elif encoder_type == "diff_attn":
            self.encoder = DiffAttnEncoder(d=d, n_heads=n_heads,
                                           n_layers_outer=n_layers_outer,
                                           max_len=seq_len)
        elif encoder_type == "mla":
            self.encoder = MLAEncoder(d=d, n_heads=n_heads,
                                      n_layers_outer=n_layers_outer,
                                      max_len=seq_len)
        elif encoder_type == "identity":
            self.encoder = IdentityEncoder()
        else:
            raise ValueError(f"Unknown encoder type '{encoder_type}'")

        # Threshold-based boundary router
        self.router = BoundaryRouter(d=d, routing=cfg["router"],
                                     target_rate=self.target_rate)

        # Simple decoder (no inner transformer, no CRL decoder)
        self.decoder = SimpleDecoder(d=d, use_ste=self.use_ste)

        # LM head (weight-tied)
        self.lm_head        = nn.Linear(d, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        # Standard init for all nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Re-apply eye init for BoundaryRouter W_q/W_k after _init_weights
        if hasattr(self.router, "W_q"):
            nn.init.eye_(self.router.W_q.weight)
            nn.init.eye_(self.router.W_k.weight)

        # residual_proj near-zero init: H-Net warm-start requirement (Eq. 3).
        nn.init.zeros_(self.decoder.residual_proj.weight)

    def _init_weights(self, m):
        """Standard init for nn.Linear and nn.Embedding.
        CausalRecurrenceLayer.log_a (nn.Parameter) keeps its role-specific init.
        BoundaryRouter W_q/W_k are re-initialized to eye_ after this pass."""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor):
        """
        x: [B, L] token indices

        Returns: (logits [B, L, vocab_size], boundary_probs [B, L], compression_ratio scalar)
        """
        h = self.embed(x)                                               # [B, L, d]
        encoder_out = self.encoder(h)                                   # [B, L, d]

        concept_tokens, _, boundary_probs, boundary_idx, concept_mask = \
            self.router(encoder_out)

        token_repr = self.decoder(
            concept_tokens, encoder_out, boundary_probs,
            boundary_idx, concept_mask,
        )                                                               # [B, L, d]

        logits = self.lm_head(token_repr)                               # [B, L, vocab_size]

        B, L = x.shape
        compression_ratio = concept_mask.float().sum(dim=1).mean() / L

        return logits, boundary_probs, compression_ratio
