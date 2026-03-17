"""
Phase 2 Model: HDC (Hierarchical Dynamic Chunking) wrapped around the mol inner network.

Architecture:
  Zone E: embed → Linear d→d/4 → 3× CausalRecurrenceLayer(d/4) → Linear d/4→d
          → BoundaryRouter → scatter to M = L//R concept tokens
  Inner:  MoL transformer (8 layers, 8 experts, rank-4) on M concept tokens
  Zone D: EMA smooth → plug-back → gated residual from Zone E skip
          → Linear d→d/4 → 3× CausalRecurrenceLayer(d/4) → Linear d/4→d → RMSNorm
  Head:   lm_head (weight-tied to embed)

Configs — run in this order:
  hdc_rulebased     Cosine threshold, no learned router. Run FIRST to validate pipeline.
  hdc_gate          Learned router, LM gradients flow (H-Net e2e style). Main A1 test.
  hdc_stride        Fixed-stride selection. Lower bound on what routing adds.
  hdc_r2 / hdc_r8   Compression ratio sweep (A2). Run after A1 passes.
  hdc_e2e_isolated  Learned router, gradient isolation (A5 comparison only).

Key correctness invariants — verified against literature (see CLAUDE.md):
  - CausalRecurrenceLayer: sqrt(1 - a_t²) normalization required (Griffin arXiv:2402.19427)
  - BoundaryRouter: compare q_t to k_{t-1} (adjacent), NOT k_ema (H-Net/DLCM consensus)
  - Zone E position indexing: dense re-indexing for inner network (H-Net default)
  - Zone D EMA + plug-back + gated residual matches H-Net Equations 3, 5, 8
  - Gradient isolation (hdc_e2e_isolated only): detach boundary_probs before Zone D;
    expected to produce near-random boundaries — use only as A5 comparison
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1.model import RMSNorm, TransformerBlock


# ============================================================
# Outer Network: CausalRecurrenceLayer
# ============================================================

class CausalRecurrenceLayer(nn.Module):
    """
    Griffin/Hawk-style Real-Gated Linear Recurrent Unit (RG-LRU).

    Per Griffin (arXiv:2402.19427):
      a_t = sigmoid(log_a)^(8 * sigmoid(W_r · x_t))   — input-dependent decay
      h_t = a_t * h_{t-1} + sqrt(1 - a_t²) * (sigmoid(W_i · x_t) * x_t)

    The sqrt(1 - a_t²) term is NOT optional — it maintains E[||h||²] bounded.
    Without it, hidden state norm grows unboundedly at small d.

    log_a initialized to 7.5 so that sigmoid(7.5)^8 ≈ 0.9953 (long initial memory).
    This matches Griffin's intent: decay starts near 1, learns to compress over training.
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d

        # Causal depthwise conv for local mixing (kernel=4, left-pad manually)
        # groups=d makes it depthwise: each channel convolved independently
        self.conv_weight = nn.Parameter(torch.randn(d, 1, 4) * 0.02)
        self.conv_bias   = nn.Parameter(torch.zeros(d))

        # Recurrence gates
        self.W_r = nn.Linear(d, d, bias=True)   # recurrence gate: controls decay strength
        self.W_i = nn.Linear(d, d, bias=True)   # input gate: controls write strength

        # Per-channel log decay. sigmoid(7.5) ≈ 0.9994, ^8 ≈ 0.9953 (long initial memory)
        self.log_a = nn.Parameter(torch.full((d,), 7.5))

        # Output
        self.out_proj = nn.Linear(d, d, bias=False)
        self.norm = RMSNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d] → [B, L, d]"""
        B, L, d = x.shape

        # 1. Causal depthwise conv (kernel=4, pad 3 on left only → causal)
        x_t      = x.transpose(1, 2)                          # [B, d, L]
        x_padded = F.pad(x_t, (3, 0))                         # [B, d, L+3]
        x_conv   = F.conv1d(x_padded, self.conv_weight,
                             self.conv_bias, groups=d)         # [B, d, L]
        x_conv   = x_conv.transpose(1, 2)                     # [B, L, d]

        # 2. Per-timestep gates
        r_t = torch.sigmoid(self.W_r(x_conv))                 # [B, L, d] in (0, 1)
        i_t = torch.sigmoid(self.W_i(x_conv))                 # [B, L, d] in (0, 1)

        # Per-channel base decay raised to 8*r_t (input-dependent)
        a_base = torch.sigmoid(self.log_a)                    # [d] in (0, 1), near 1
        a_t    = a_base.pow(8.0 * r_t)                        # [B, L, d] in (~0.9, 1)

        # 3. Recurrence scan with norm conservation
        h = x.new_zeros(B, d)
        outputs = []
        for t in range(L):
            a   = a_t[:, t]                                    # [B, d]
            inp = i_t[:, t] * x_conv[:, t]                    # [B, d]
            # sqrt(1 - a²) ensures unit-norm input contributes predictably
            h = a * h + torch.sqrt((1.0 - a.pow(2)).clamp(min=0.0)) * inp
            outputs.append(h)
        out = torch.stack(outputs, dim=1)                      # [B, L, d]

        # 4. Output projection + norm
        return self.norm(self.out_proj(out))


# ============================================================
# Boundary Router
# ============================================================

class BoundaryRouter(nn.Module):
    """
    Selects exactly M = L // R concept-token positions from L encoder outputs.

    Three routing modes:

    cosine_rule (hdc_rulebased):
      p_t = (1 - cos_sim(enc_t, enc_{t-1})) / 2, no learned params.
      Most stable. Validates pipeline before adding learned routing.

    learned_e2e (hdc_gate, hdc_r2, hdc_r8):
      p_t = (1 - dot(normalize(W_q·enc_t), normalize(W_k·enc_{t-1}))) / 2
      W_q, W_k initialized to identity (pure cosine similarity at init).
      LM gradients flow through boundary_probs via Zone D's EMA and gated residual.
      This is the H-Net approach (arXiv:2507.07955).

    fixed_stride (hdc_stride):
      Select every R-th token. No signal whatsoever. Lower bound baseline.

    learned_isolated (hdc_e2e_isolated, A5 only):
      Same as learned_e2e but boundary_probs is detached before Zone D.
      Router only receives gradient from loss_comp. Expected to produce
      near-random boundaries. For comparison only — not recommended as default.

    IMPORTANT: always uses k_{t-1} (adjacent key), NOT k_ema.
    EMA of keys dilutes the local-contrast signal. H-Net and DLCM both use
    adjacent-token comparison.
    """

    def __init__(self, d: int, R: int = 4, routing: str = "cosine_rule"):
        super().__init__()
        self.d = d
        self.R = R
        self.routing = routing

        if routing in ("learned_e2e", "learned_isolated"):
            # Identity init: router starts as pure cosine similarity
            self.W_q = nn.Linear(d, d, bias=False)
            self.W_k = nn.Linear(d, d, bias=False)
            nn.init.eye_(self.W_q.weight)
            nn.init.eye_(self.W_k.weight)

    def forward(self, enc: torch.Tensor):
        """
        enc: [B, L, d] encoder output (after Zone E recurrence)

        Returns:
          boundary_probs:        [B, L]  soft boundary probability in [0, 1]
          boundary_idx:          [B, M]  positions of M selected concept tokens, sorted
          boundary_probs_for_zd: [B, L]  same as boundary_probs, or detached if isolated
        """
        B, L, d = enc.shape
        M = L // self.R

        # ── Fixed stride ──────────────────────────────────────────────────────
        if self.routing == "fixed_stride":
            # Select positions R-1, 2R-1, ... (last token of each stride window)
            idx = torch.arange(self.R - 1, L, self.R, device=enc.device)[:M]
            boundary_idx   = idx.unsqueeze(0).expand(B, -1).contiguous()
            boundary_probs = torch.zeros(B, L, device=enc.device, dtype=enc.dtype)
            boundary_probs.scatter_(1, boundary_idx, 1.0)
            return boundary_probs, boundary_idx, boundary_probs

        # ── Compute boundary probabilities ────────────────────────────────────
        if self.routing == "cosine_rule":
            enc_n = F.normalize(enc, dim=-1)                   # [B, L, d]
            # Cosine sim between adjacent tokens
            sim = (enc_n[:, 1:] * enc_n[:, :-1]).sum(dim=-1)  # [B, L-1]
            p   = (1.0 - sim) / 2.0                            # [B, L-1] in [0, 1]
            # Position 0: no predecessor → pad with 1.0 (certain boundary).
            # Must be 1.0, not 0.0: Zone D's EMA uses p_at_bounds[:,0] directly.
            # value=0.0 would zero out the first concept token in EMA smoothing,
            # causing all tokens in the first chunk to receive zeros from the
            # inner network (plugback = smoothed[0] = zeros).
            boundary_probs = F.pad(p, (1, 0), value=1.0)       # [B, L]

        else:  # learned_e2e or learned_isolated
            q = F.normalize(self.W_q(enc), dim=-1)             # [B, L, d]
            k = F.normalize(self.W_k(enc), dim=-1)             # [B, L, d]
            # Compare each q_t to k_{t-1} (adjacent, NOT ema)
            sim = (q[:, 1:] * k[:, :-1]).sum(dim=-1)           # [B, L-1]
            p   = (1.0 - sim) / 2.0                            # [B, L-1]
            boundary_probs = F.pad(p, (1, 0), value=1.0)       # [B, L], pos 0 = 1.0

        # ── Top-M selection (always exact M = L // R) ─────────────────────────
        # Position 0 is already 1.0 — always wins topk. No boost needed.
        _, topk_idx  = boundary_probs.topk(M, dim=1)           # [B, M]
        boundary_idx = topk_idx.sort(dim=1).values             # [B, M] ascending by position

        # For isolated routing: block LM gradients from reaching the router
        if self.routing == "learned_isolated":
            boundary_probs_for_zd = boundary_probs.detach()
        else:
            boundary_probs_for_zd = boundary_probs

        return boundary_probs, boundary_idx, boundary_probs_for_zd


# ============================================================
# Zone E: Encoder + Router
# ============================================================

class ZoneE(nn.Module):
    """
    Lightweight encoder that produces M = L // R concept tokens from L input tokens.

    Pipeline:
      [B, L, d] (pre-embedded) → Linear d→d/4
      → 3× CausalRecurrenceLayer(d/4)
      → Linear d/4→d  → encoder_out [B, L, d]  ← saved as Zone D skip connection
      → BoundaryRouter → select M positions
      → gather concept_tokens [B, M, d]

    Note: embedding is done in HDCModel.forward before calling Zone E.
    Zone E receives embedded tokens [B, L, d], not raw indices.

    Dense re-indexing: concept tokens pass through the inner network with positions
    0, 1, ..., M-1. H-Net's validated approach for causal LM. Sparse original-position
    indexing is left as ablation A-new.
    """

    def __init__(self, d: int, d_outer: int, R: int, routing: str):
        super().__init__()
        self.d = d
        self.down_proj  = nn.Linear(d, d_outer, bias=False)
        self.recurrence = nn.ModuleList([CausalRecurrenceLayer(d_outer) for _ in range(3)])
        self.up_proj    = nn.Linear(d_outer, d, bias=False)
        self.router     = BoundaryRouter(d=d, R=R, routing=routing)

    def forward(self, h: torch.Tensor):
        """
        h: [B, L, d] embedded token representations

        Returns:
          concept_tokens:       [B, M, d]
          encoder_out:          [B, L, d] — skip connection for Zone D
          boundary_probs:       [B, L]    — full probabilities for loss_comp and metrics
          boundary_probs_for_zd:[B, L]    — detached for isolated routing, else same
          boundary_idx:         [B, M]    — original positions of concept tokens
        """
        B, L, d = h.shape

        # Down-project, recurrence, up-project
        h = self.down_proj(h)                  # [B, L, d/4]
        for rec in self.recurrence:
            h = rec(h)                         # [B, L, d/4]
        encoder_out = self.up_proj(h)          # [B, L, d]  — skip connection

        # Route: select M concept token positions
        boundary_probs, boundary_idx, boundary_probs_for_zd = self.router(encoder_out)

        # Gather selected encoder outputs as concept tokens
        concept_tokens = encoder_out.gather(
            1,
            boundary_idx.unsqueeze(-1).expand(-1, -1, d),   # [B, M, d]
        )

        return concept_tokens, encoder_out, boundary_probs, boundary_probs_for_zd, boundary_idx


# ============================================================
# Inner Transformer (mol config, no embed / lm_head)
# ============================================================

class InnerTransformer(nn.Module):
    """
    Transformer blocks only — no embedding, no LM head.
    Receives concept tokens [B, M, d] from Zone E, returns processed [B, M, d].

    Uses mol config: MoL FFN (8 experts, top-2, rank-4 LoRA).
    Concept tokens use dense positions 0..M-1 for RoPE (H-Net default).
    """

    def __init__(self, d: int = 256, n_layers: int = 8, n_heads: int = 8,
                 n_experts: int = 8, mol_rank: int = 4, mol_top_k: int = 2,
                 max_len: int = 2048):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d=d, n_heads=n_heads, n_streams=1,
                use_mhc=False, use_mol=True,
                n_experts=n_experts, mol_rank=mol_rank, mol_top_k=mol_top_k,
                max_len=max_len,
            )
            for _ in range(n_layers)
        ])
        self.norm_out = RMSNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, M, d] → [B, M, d]"""
        for block in self.blocks:
            x = block(x)
        return self.norm_out(x)

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


# ============================================================
# Zone D: De-Chunker + Decoder
# ============================================================

class ZoneD(nn.Module):
    """
    Reconstructs L-length token representations from M concept tokens.

    Four operations, matching H-Net Equations 3, 5, 8 with one refinement:

    1. EMA smooth (H-Net Eq. 5):
         h̄_i = p_i * concept_i + (1 - p_i) * h̄_{i-1}
         p_i = boundary_probs at boundary position i

    2. Plug-back (H-Net Eq. 8):
         token j gets h̄ of the nearest preceding boundary position
         Implemented via torch.bucketize on sorted boundary_idx

    3. Gated residual (H-Net Eq. 3, refined):
         h_j = (1 - p_j) * sigmoid(W_gate · encoder_out_j) * encoder_out_j + plugback_j
         Non-boundary tokens (low p_j) lean more on Zone E's encoder_out,
         preserving fine-grained local information not seen by inner network.
         This mitigates the U-shaped loss (DLCM Section 4.2).

    4. Decoder recurrence:
         Linear d→d/4 → 3× CausalRecurrenceLayer(d/4) → Linear d/4→d → RMSNorm
    """

    def __init__(self, d: int, d_outer: int):
        super().__init__()
        self.gate_proj  = nn.Linear(d, d, bias=True)
        self.down_proj  = nn.Linear(d, d_outer, bias=False)
        self.recurrence = nn.ModuleList([CausalRecurrenceLayer(d_outer) for _ in range(3)])
        self.up_proj    = nn.Linear(d_outer, d, bias=False)
        self.norm_out   = RMSNorm(d)

    def forward(
        self,
        concept_out:    torch.Tensor,   # [B, M, d]  inner network output
        encoder_out:    torch.Tensor,   # [B, L, d]  Zone E skip connection
        boundary_probs: torch.Tensor,   # [B, L]     soft boundary probabilities
        boundary_idx:   torch.Tensor,   # [B, M]     sorted positions of concept tokens
    ) -> torch.Tensor:
        B, L, d = encoder_out.shape
        M = concept_out.shape[1]

        # ── Step 1: EMA smoothing over M concept tokens ───────────────────────
        # p at each selected boundary position
        p_at_bounds = boundary_probs.gather(1, boundary_idx)  # [B, M]

        h_prev     = concept_out.new_zeros(B, d)
        smoothed   = []
        for i in range(M):
            p_i    = p_at_bounds[:, i:i+1]                    # [B, 1]
            h_prev = p_i * concept_out[:, i] + (1.0 - p_i) * h_prev
            smoothed.append(h_prev)
        smoothed = torch.stack(smoothed, dim=1)                # [B, M, d]

        # ── Step 2: Plug-back — map each of L positions to nearest boundary ───
        # For position j: find largest boundary_idx[b] <= j
        # torch.bucketize(j, sorted_boundaries, right=True) returns count of
        # boundaries <= j, so index = count - 1.
        # Since position 0 is always selected, count >= 1 for all j >= 0.
        plugback = torch.empty(B, L, d,
                               device=concept_out.device, dtype=concept_out.dtype)
        arange_L = torch.arange(L, device=boundary_idx.device)
        for b in range(B):
            count       = torch.bucketize(arange_L, boundary_idx[b], right=True)  # [L]
            bucket      = (count - 1).clamp(min=0)                                # [L]
            plugback[b] = smoothed[b][bucket]                                     # [L, d]

        # ── Step 3: Gated residual from Zone E skip connection ─────────────────
        gate       = torch.sigmoid(self.gate_proj(encoder_out))  # [B, L, d]
        p_expanded = boundary_probs.unsqueeze(-1)                 # [B, L, 1]
        h          = (1.0 - p_expanded) * gate * encoder_out + plugback  # [B, L, d]

        # ── Step 4: Decoder recurrence ─────────────────────────────────────────
        h = self.down_proj(h)               # [B, L, d/4]
        for rec in self.recurrence:
            h = rec(h)                      # [B, L, d/4]
        h = self.up_proj(h)                 # [B, L, d]

        return self.norm_out(h)             # [B, L, d]


# ============================================================
# Full HDC Model
# ============================================================

class HDCModel(nn.Module):
    """
    HDC-wrapped mol inner network.

    Forward pass:
      embed(x) → ZoneE → InnerTransformer(M concept tokens) → ZoneD → lm_head

    Returns (logits [B, L, vocab], boundary_probs [B, L]).
    The training loop adds: loss = loss_ntp + lambda_comp * loss_comp
    where loss_comp = (boundary_probs.mean() - 1/R)^2

    Weight tying: lm_head.weight = embed.weight (same as Phase 1).
    """

    CONFIGS = {
        "hdc_rulebased":    dict(routing="cosine_rule",      R=4),
        "hdc_gate":         dict(routing="learned_e2e",      R=4),
        "hdc_stride":       dict(routing="fixed_stride",     R=4),
        "hdc_r2":           dict(routing="learned_e2e",      R=2),
        "hdc_r8":           dict(routing="learned_e2e",      R=8),
        "hdc_e2e_isolated": dict(routing="learned_isolated", R=4),
    }

    def __init__(
        self,
        config:     str = "hdc_rulebased",
        d:          int = 256,
        n_layers:   int = 8,
        n_heads:    int = 8,
        vocab_size: int = 256,
        seq_len:    int = 512,
        n_experts:  int = 8,
        mol_rank:   int = 4,
        mol_top_k:  int = 2,
    ):
        super().__init__()
        if config not in self.CONFIGS:
            raise ValueError(f"Unknown config '{config}'. Valid: {list(self.CONFIGS)}")

        cfg = self.CONFIGS[config]
        self.config_name = config
        self.R           = cfg["R"]
        self.d           = d
        d_outer          = d // 4   # Zone E and D operate at d/4 = 64

        # Shared embedding (weight-tied to lm_head)
        self.embed = nn.Embedding(vocab_size, d)

        # Zone E
        self.zone_e = ZoneE(d=d, d_outer=d_outer, R=self.R, routing=cfg["routing"])

        # Inner transformer (mol, operates on M = seq_len // R concept tokens)
        # max_len covers up to seq_len concept tokens (M <= seq_len)
        self.inner = InnerTransformer(
            d=d, n_layers=n_layers, n_heads=n_heads,
            n_experts=n_experts, mol_rank=mol_rank, mol_top_k=mol_top_k,
            max_len=seq_len,
        )

        # Zone D
        self.zone_d = ZoneD(d=d, d_outer=d_outer)

        # LM head (weight-tied)
        self.lm_head        = nn.Linear(d, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        # Standard init for all nn.Linear and nn.Embedding
        self.apply(self._init_weights)
        # Re-apply eye init for BoundaryRouter (apply above would overwrite it)
        router = self.zone_e.router
        if hasattr(router, "W_q"):
            nn.init.eye_(router.W_q.weight)
            nn.init.eye_(router.W_k.weight)

    def _init_weights(self, m):
        """Touches only nn.Linear and nn.Embedding.
        CausalRecurrenceLayer.log_a (nn.Parameter, not Linear) keeps its 7.5 init.
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
        Returns: (logits [B, L, vocab_size], boundary_probs [B, L])

        Training loop:
          logits, bp = model(x)
          loss_ntp   = F.cross_entropy(logits.view(-1, V), y.view(-1))
          loss_comp  = (bp.mean() - 1.0 / self.R) ** 2
          loss       = loss_ntp + lambda_comp * loss_comp
        """
        # Embed once — shared between Zone E and the weight-tied lm_head
        h = self.embed(x)                                   # [B, L, d]

        # Zone E: encode + route → M concept tokens
        concept_tokens, encoder_out, boundary_probs, boundary_probs_for_zd, boundary_idx = \
            self.zone_e(h)

        # Inner transformer: process M concept tokens
        concept_out = self.inner(concept_tokens)            # [B, M, d]

        # Zone D: reconstruct L token representations
        token_repr = self.zone_d(
            concept_out, encoder_out, boundary_probs_for_zd, boundary_idx,
        )                                                   # [B, L, d]

        logits = self.lm_head(token_repr)                   # [B, L, vocab_size]
        return logits, boundary_probs

    def get_mol_stats(self):
        return self.inner.get_mol_stats()

    def reset_mol_counts(self):
        self.inner.reset_mol_counts()
