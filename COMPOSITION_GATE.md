# Composition Gate

**Purpose:** Single authoritative prerequisite list for the MLA + Diff Attn V2 + mHC + MoL
composition experiment (Phase 3). No composition experiment may begin until all items below
are marked `complete`. This file is the single source of truth — partial documentation in
CLAUDE.md and STUDY_DESIGN.md defers to this file.

The composition target: a single `ToyTransformer` block using all four Phase 1 components
simultaneously, composed with the best Phase 2 outer encoder as the concept-token producer.

---

## Prerequisites

### 1. mHC grad norm diagnostic
**Status:** PENDING

**What:** Run two diagnostic configs and observe grad norm trajectory over 25k steps:
```bash
bash run_experiments.sh mhc --max_lr 1.5e-4 --total_steps 25000  # current best LR
bash run_experiments.sh mhc --max_lr 7.5e-5  --total_steps 25000  # half LR
```

**Complete when:** Grad norm trajectory is understood — either it stabilizes at a lower LR
(LR sensitivity), or the root cause is identified in KromHCResidual / H_post softmax / the
4-stream interaction. A written diagnosis must accompany the result.

**Why blocking:** mHC shows rising grad norm 0.80→1.37 over 100k steps with no stabilization.
Composing an unstable component amplifies its pathology. The `compose` config result (3.5416)
may be a local minimum before divergence. Any composition claim is unreliable until this is
understood.

---

### 2. Q1 multi-seed confirmation (baseline_wide vs. mol)
**Status:** PENDING

**What:** Run `baseline`, `baseline_wide`, and `mol` each at 3 seeds (42, 1, 2) for 100k steps.
Apply Melis et al. 2018 reporting standard: claim is valid if margin > 3× cross-seed std,
or worst seed of treatment beats best seed of control.

**Current result (single-seed):** baseline_wide=3.5355, mol=3.5441, margin=0.0086 BPC.
Typical cross-seed std at this scale: 0.003–0.010 BPC. Margin may be within noise.

**Complete when:** 3-seed results are logged, std is computed, and the Melis et al. criterion
is applied. The conclusion ("capacity dominates routing at this scale" or its negation) must
be written in RESULTS.md.

**Why blocking:** Q1 is the primary justification for whether MoL is worth scaling. If the
conclusion reverses, MoL's role in the composition changes.

---

### 3. Q2 multi-seed confirmation (mol vs. mol_single)
**Status:** PENDING

**What:** Run `mol` and `mol_single` each at 3 seeds (42, 1, 2) for 100k steps. Apply Melis
et al. criterion.

**Current result (single-seed):** mol=3.5441, mol_single=3.5516, margin=0.0075 BPC.

**Complete when:** Same standard as Q1. The conclusion ("routing over adapters adds value" or
its negation) must be written in RESULTS.md.

**Why blocking:** If Q2 reverses (mol_single ≥ mol), the MoL routing mechanism provides no
benefit and the composition should use SingleLoRAFFN instead, significantly changing the
architecture.

---

### 4. Architectural decision: mHC wrapping of MLA / DiffAttn
**Status:** PENDING

**What:** Make and document an explicit architectural decision for how mHC composes with
non-standard attention at the block level. Two options:

  **Option A (collapse-then-attend):** H_pre collapses 4 streams → 1 input → MLA/DiffAttn
  processes → H_post distributes output to 4 streams. Currently implemented for standard
  attention in `_forward_mhc`. Extending to MLA/DiffAttn is straightforward — remove the
  guard in `transformer_block.py`.

  **Option B (per-stream attention):** Each of the 4 streams attends independently with its
  own KV, then H_post combines. Requires 4× the attention compute. Not implemented.

  **Hybrid Option C:** Collapse for Q computation, keep separate KV per stream. Untested.

**Complete when:** One option is chosen and justified with reference to the mHC paper's
intent (full-residual-stream design). A brief written rationale (2–3 sentences) is added
to `references/components/mhc.md` under a new "Composition with MLA/DiffAttn" section.
The guard in `transformer_block.py` is updated to either enforce the chosen option or
remove the restriction with a documented rationale.

**Why blocking:** Without this decision, any composed mHC+MLA or mHC+DiffAttn experiment
is running an undocumented architecture that may differ from the paper's intent in unknown
ways.

---

### 5. Parameter budget for the composed model
**Status:** PENDING

**What:** Compute the exact parameter count of the composed model (MLA + DiffAttn + mHC + MoL
in a single `TransformerBlock`) and identify a capacity-matched baseline (`baseline_composed_wide`
with scaled d_ff) so that any BPC improvement from composition can be attributed to architecture
rather than capacity.

**Complete when:** The composed config is added to `ToyTransformer.CONFIGS` with an explicit
`n_params` note, and a `baseline_composed_wide` config exists with d_ff chosen to match total
params within 5% (same tolerance as the Q1 comparison). Both are validated by `shape_check.py`.

**Why blocking:** Without a matched control, a composed model BPC improvement is uninterpretable —
it could be architecture or capacity. The Q1 experience (baseline_wide needed because mol had
more params) must not be repeated at the composition level.

---

### 6. Phase 2 outer encoder study complete
**Status:** PENDING

**What:** All 10 Phase 2 configs (see `OuterModel.CONFIGS`) must be run to completion (50k steps
each). The best outer encoder architecture must be selected based on `boundary_bpc`,
`midchunk_bpc`, and `encoder_diversity` metrics. The Q3 comparison
(`outer_crl_learned` vs. `outer_crl_fixed_stride`) must yield a clear conclusion.

**Complete when:** All 10 configs are in `checkpoints/report.md` with final BPC values. The
Q3 conclusion is written in RESULTS.md. The selected outer encoder for Phase 3 is documented
here (update the line below):

> **Selected outer encoder for Phase 3:** TBD

**Why blocking:** Phase 3 composition uses the outer encoder as the concept-token producer.
Choosing the wrong encoder architecture invalidates the composition experiment before it starts.

---

### 7. Phase 3 inner transformer interface spec
**Status:** PENDING

**What:** Write a spec (as a markdown doc at `references/components/phase3_interface.md`) defining
the contract between `OuterModel` and `ToyTransformer` for Phase 3:

  - What tensor the inner transformer receives: `concept_tokens [B, M_max, d]`, padded with
    zeros at positions where `concept_mask = False`
  - What it returns: processed concept tokens `[B, M_max, d]`
  - How `concept_mask` maps to an attention `key_padding_mask` — and what code changes are
    needed in `CausalSelfAttention`, `MLACausalAttention`, and `DifferentialCausalAttention`
    to accept an optional padding mask
  - Whether the inner transformer uses causal attention over concept tokens (recommended: yes,
    following H-Net), or bidirectional attention (not recommended without ablation)
  - How plug-back index (`boundary_idx`) is threaded through the inner transformer to the
    Zone D decoder

**Complete when:** The spec document exists and has been reviewed. The required code changes
to attention modules are listed as concrete function signature changes. No code needs to be
written yet — the spec is the gate.

**Why blocking:** Without this spec, Phase 3 code will be written against an undocumented
interface. The padding mask issue alone (attention over zero-padded slots corrupts inner
transformer output near sequence boundaries) could silently degrade results.

---

## Summary table

| # | Prerequisite | Requires compute? | Status |
|---|-------------|-------------------|--------|
| 1 | mHC grad norm diagnostic | Yes (2 × 25k steps) | PENDING |
| 2 | Q1 multi-seed (baseline_wide vs. mol) | Yes (6 × 100k steps) | PENDING |
| 3 | Q2 multi-seed (mol vs. mol_single) | Yes (6 × 100k steps) | PENDING |
| 4 | mHC wrapping architecture decision | No | PENDING |
| 5 | Composed model parameter budget + matched control | No | PENDING |
| 6 | Phase 2 outer encoder study complete | Yes (10 × 50k steps) | PENDING |
| 7 | Phase 3 inner transformer interface spec | No | PENDING |

Items 4, 5, 7 can be completed immediately without running any experiments.
Items 1, 2, 3, 6 require cloud compute on Lightning.ai.

The minimum compute path to unblocking composition: run the mHC diagnostic (1) first,
then run Phase 2 (6) in parallel with the multi-seed Phase 1 reruns (2, 3).
