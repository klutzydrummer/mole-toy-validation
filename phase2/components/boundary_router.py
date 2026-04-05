"""
BoundaryRouter — threshold-based concept token selection.

Spec: references/components/boundary_router.md
Sources: references/sources/papers/hnet_2507.07955.md,
         references/sources/code/hnet_boundary.py,
         references/sources/papers/dlcm.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryRouter(nn.Module):
    """
    Selects concept-token positions from L encoder outputs using threshold p > 0.5.

    Unlike the old topk(M) approach, this produces variable M per sequence, padded
    to M_max = max boundary count in the batch. This is the correct H-Net formulation.

    Three routing modes:

    cosine_rule:
      p_t = (1 - cos_sim(enc_t, enc_{t-1})) / 2, no learned params.
      Most stable. Validates pipeline before adding learned routing.

    learned_e2e:
      p_t = (1 - dot(normalize(W_q·enc_t), normalize(W_k·enc_{t-1}))) / 2
      W_q, W_k initialized to identity (pure cosine similarity at init).
      LM gradients flow through boundary_probs via SimpleDecoder's EMA and gated residual.
      This is the H-Net approach (arXiv:2507.07955).

    fixed_stride (outer_strided):
      Select evenly spaced positions (round(L * target_rate) positions across L).
      No boundary_probs computation; boundary_probs = zeros except at selected positions.

    IMPORTANT: always uses k_{t-1} (adjacent key), NOT k_ema.
    """

    def __init__(self, d: int, routing: str = "cosine_rule", target_rate: float = 0.25):
        super().__init__()
        self.d           = d
        self.routing     = routing
        self.target_rate = target_rate

        if routing == "learned_e2e":
            # Identity init: router starts as pure cosine similarity
            self.W_q = nn.Linear(d, d, bias=False)
            self.W_k = nn.Linear(d, d, bias=False)
            nn.init.eye_(self.W_q.weight)
            nn.init.eye_(self.W_k.weight)

    def forward(self, enc: torch.Tensor):
        """
        enc: [B, L, d] encoder output

        Returns:
          concept_tokens: [B, M_max, d]  gathered encoder outputs at boundaries (padded)
          encoder_out:    [B, L, d]      pass-through (same as enc)
          boundary_probs: [B, L]         soft boundary probability in [0, 1]
          boundary_idx:   [B, M_max]     positions of selected concept tokens (padded with 0)
          concept_mask:   [B, M_max]     True for valid concept tokens, False for padding
        """
        B, L, d = enc.shape

        # ── Fixed stride ──────────────────────────────────────────────────────
        if self.routing == "fixed_stride":
            M = max(1, round(L * self.target_rate))
            # Evenly spaced positions across L
            positions = torch.linspace(0, L - 1, steps=M, device=enc.device).long()
            # All sequences get the same positions; M_max = M (no padding needed)
            boundary_idx   = positions.unsqueeze(0).expand(B, -1).contiguous()  # [B, M]
            boundary_probs = torch.zeros(B, L, device=enc.device, dtype=enc.dtype)
            boundary_probs.scatter_(1, boundary_idx, 1.0)
            concept_mask   = torch.ones(B, M, device=enc.device, dtype=torch.bool)
            concept_tokens = enc.gather(
                1, boundary_idx.unsqueeze(-1).expand(-1, -1, d)
            )  # [B, M, d]
            return concept_tokens, enc, boundary_probs, boundary_idx, concept_mask

        # ── Compute boundary probabilities ────────────────────────────────────
        if self.routing == "cosine_rule":
            enc_n = F.normalize(enc, dim=-1)                   # [B, L, d]
            sim   = (enc_n[:, 1:] * enc_n[:, :-1]).sum(dim=-1) # [B, L-1]
            p     = (1.0 - sim) / 2.0                          # [B, L-1] in [0, 1]
            # Position 0: no predecessor → pad with 1.0 (certain boundary).
            # Must be 1.0, not 0.0: SimpleDecoder's EMA uses p_at_bounds[:,0] directly.
            boundary_probs = F.pad(p, (1, 0), value=1.0)       # [B, L]

        else:  # learned_e2e
            q = F.normalize(self.W_q(enc), dim=-1)             # [B, L, d]
            k = F.normalize(self.W_k(enc), dim=-1)             # [B, L, d]
            # Compare each q_t to k_{t-1} (adjacent, NOT ema) — H-Net Eq. 4
            sim = (q[:, 1:] * k[:, :-1]).sum(dim=-1)           # [B, L-1]
            p   = (1.0 - sim) / 2.0                            # [B, L-1]
            boundary_probs = F.pad(p, (1, 0), value=1.0)       # [B, L], pos 0 = 1.0

        # ── Threshold selection: p > 0.5, variable M per sequence ─────────────
        # Position 0 is always 1.0 → always selected. Subsequent positions selected
        # where cosine dissimilarity is high (tokens differ from predecessor).
        boundary_mask = (boundary_probs > 0.5)                 # [B, L] bool
        counts        = boundary_mask.sum(dim=1)               # [B]
        M_max         = int(counts.max().item())
        M_max         = max(M_max, 1)                          # at least 1 slot

        # Build padded boundary_idx and concept_mask — fully vectorised, no Python loop.
        # Compile-friendly: no data-dependent control flow, no per-element Python ops.
        #
        # Strategy: assign each boundary position a 0-indexed rank within its row.
        #   rank[b, l] = cumsum(b_long)[b, l] * b_long[b, l]
        #              = 1-indexed rank if boundary, 0 if non-boundary.
        # Non-boundary positions (rank=0) must not clobber slot 0.
        # Fix: route non-boundary positions to a dummy slot M_max and discard it.
        #   slot_safe[b, l] = rank[b,l]-1  if boundary   (0-indexed valid slot)
        #                   = M_max        if non-boundary (discarded dummy slot)
        # Allocate M_max+1 slots, scatter, then trim the last slot.
        b_long   = boundary_mask.long()                         # [B, L]
        rank     = b_long.cumsum(dim=1) * b_long                # [B, L] 1-indexed rank, 0 if not boundary
        slot_safe = torch.where(
            boundary_mask,
            rank - 1,                                           # 0-indexed slot for boundaries
            torch.full_like(rank, M_max),                       # dummy slot for non-boundaries
        )                                                        # [B, L]

        pos = torch.arange(L, device=enc.device).unsqueeze(0).expand(B, -1)  # [B, L]
        # Allocate M_max+1 slots; the last (M_max) is the dummy absorbing non-boundary writes.
        _bidx = torch.zeros(B, M_max + 1, device=enc.device, dtype=torch.long)
        _cmsk = torch.zeros(B, M_max + 1, device=enc.device, dtype=torch.bool)
        _bidx.scatter_(1, slot_safe, pos)
        _cmsk.scatter_(1, slot_safe, boundary_mask)
        boundary_idx = _bidx[:, :M_max]                         # [B, M_max] — drop dummy slot
        concept_mask = _cmsk[:, :M_max]                         # [B, M_max]

        # Gather concept tokens at boundary positions
        concept_tokens = enc.gather(
            1, boundary_idx.unsqueeze(-1).expand(-1, -1, d)   # [B, M_max, d]
        )

        return concept_tokens, enc, boundary_probs, boundary_idx, concept_mask
