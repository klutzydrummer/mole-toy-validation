"""
Zone D: SimpleDecoder — EMA smooth → plug-back → STE → residual projection.

Spec: references/components/zone_ed_pipeline.md
Sources: references/sources/papers/hnet_2507.07955.md,
         references/sources/code/hnet_boundary.py
"""

import torch
import torch.nn as nn

from phase2.components.causal_recurrence import _parallel_scan


class SimpleDecoder(nn.Module):
    """
    Reconstructs L-length representations from M_max concept tokens (padded).

    No inner transformer. No CRL decoder layers.

    Implements H-Net Equations 5, 6, 7, 8, 9, 3 in order:

    1. EMA smoothing (H-Net Eq. 5) over M_max concept tokens:
         h̄_i = p_i * concept_i + (1 - p_i) * h̄_{i-1}
         p_i  = boundary_probs at boundary position i, clamped to [0.1, 1.0]
         h0   = concept_tokens[:, 0]  (always present since p_0=1.0)

    2. Plug-back (H-Net Eq. 8) via cumsum(boundary_mask) - 1:
         plug_back_idx = cumsum(boundary_mask.long(), dim=1) - 1  [B, L]
         maps each of L positions to its nearest preceding concept token in smoothed [B, M_max, d]

    3. Confidence scoring + STE (H-Net Eq. 6–7–9):
         c_t     = p_t  if b_t=1,  (1 - p_t) if b_t=0          (Eq. 6)
         ste_c_t = c_t + stopgradient(1 - c_t)  = 1.0 in fwd   (Eq. 7)
         upsampled_t = ste_c_t * z̃_t                             (Eq. 9)
         Incentivizes the router to make confident, decisive decisions.

    4. Residual (H-Net Eq. 3):
         out = upsampled + residual_proj(encoder_out)
         residual_proj is nn.Linear(d, d, bias=False) initialized near-zero (weight=0).
         No sigmoid, no p-modulation — plain linear skip, suppressed at init.

    lm_head is weight-tied to embed — applied in OuterModel.forward.
    """

    def __init__(self, d: int, use_ste: bool = True):
        super().__init__()
        self.use_ste = use_ste
        # Plain linear skip connection (H-Net Eq. 3); initialized near-zero in OuterModel.__init__
        self.residual_proj = nn.Linear(d, d, bias=False)

    def forward(
        self,
        concept_tokens: torch.Tensor,   # [B, M_max, d]  concept token representations
        encoder_out:    torch.Tensor,   # [B, L, d]      Zone E output (skip connection)
        boundary_probs: torch.Tensor,   # [B, L]         soft boundary probabilities
        boundary_idx:   torch.Tensor,   # [B, M_max]     sorted boundary positions (padded)
        concept_mask:   torch.Tensor,   # [B, M_max]     True = valid, False = padding
    ) -> torch.Tensor:
        """Returns [B, L, d] reconstructed token representations."""
        B, L, d = encoder_out.shape
        M_max   = concept_tokens.shape[1]

        # ── Step 1: EMA smoothing over M_max concept tokens (H-Net Eq. 5) ─────
        # EMA recurrence: h_i = p_i * concept_i + (1 - p_i) * h_{i-1}
        # h0 = concept_tokens[:, 0]  (pos 0 always valid since boundary_probs[:,0]=1.0)
        #
        # p_at_bounds: boundary_probs at each concept position.
        # Padding slots get p=1.0 so EMA treats them as fresh boundaries (no bleed).
        p_at_bounds_raw = boundary_probs.gather(1, boundary_idx)        # [B, M_max]
        p_at_bounds = torch.where(concept_mask, p_at_bounds_raw,
                                  torch.ones_like(p_at_bounds_raw))     # [B, M_max]
        p_at_bounds = p_at_bounds.clamp(min=0.1)                        # EMA collapse guard

        h0 = concept_tokens[:, 0]                                       # [B, d]
        if M_max > 1:
            p_rest   = p_at_bounds[:, 1:]                               # [B, M_max-1]
            decay    = (1.0 - p_rest).clamp(min=1e-7)                  # [B, M_max-1]
            b_rest   = p_rest.unsqueeze(-1) * concept_tokens[:, 1:]    # [B, M_max-1, d]
            a_rest   = decay.unsqueeze(-1).expand_as(b_rest)           # [B, M_max-1, d]
            h_rest   = _parallel_scan(a_rest, b_rest, h0)              # [B, M_max-1, d]
            smoothed = torch.cat([h0.unsqueeze(1), h_rest], dim=1)     # [B, M_max, d]
        else:
            smoothed = h0.unsqueeze(1)                                  # [B, 1, d]

        # ── Step 2: Plug-back via cumsum (H-Net Eq. 8) ───────────────────────
        # boundary_mask[b, j] = True iff position j is a boundary in sequence b.
        # cumsum(boundary_mask) - 1 gives the 0-indexed concept bucket for each L position:
        #   positions before the first boundary clamp to 0 (safe since p_0=1.0 always)
        #   each boundary position starts a new bucket
        boundary_mask  = (boundary_probs >= 0.5)                        # [B, L] bool
        plug_back_idx  = torch.cumsum(boundary_mask.long(), dim=1) - 1 # [B, L]
        plug_back_idx  = plug_back_idx.clamp(min=0)                    # guard against -1 at pos 0
        plugback = smoothed.gather(
            1, plug_back_idx.unsqueeze(-1).expand(-1, -1, d),
        )                                                                # [B, L, d]

        # ── Step 3: Confidence scoring + STE (H-Net Eq. 6–7–9) ──────────────
        # c_t = p_t if b_t=1, (1-p_t) if b_t=0 — quantifies router confidence
        # ste(c_t) = c_t + stopgrad(1-c_t) = 1.0 in fwd, gradient flows through c_t
        # Incentivizes confident decisions: uncertain router (p≈0.5) gets output scaled by 0.5
        # use_ste=False: ablation — use plain c_t without STE (no rounding to 1.0 in fwd)
        b_float = boundary_mask.float()                                 # [B, L]
        c_t     = b_float * boundary_probs + (1.0 - b_float) * (1.0 - boundary_probs)
        if self.use_ste:
            ste_c = c_t + (1.0 - c_t).detach()                        # [B, L] = 1.0 in fwd
        else:
            ste_c = c_t                                                 # plain confidence scaling
        upsampled = ste_c.unsqueeze(-1) * plugback                     # [B, L, d]

        # ── Step 4: Residual from Zone E skip (H-Net Eq. 3) ─────────────────
        # out = upsampled + residual_proj(encoder_out)
        # residual_proj weight=0 at init → residual path suppressed at start of training
        out = upsampled + self.residual_proj(encoder_out)              # [B, L, d]

        return out
