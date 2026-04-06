# Results

All results are on WikiText-103-raw, BPE tokenizer (vocab=4096, sentencepiece), seq_len=256.
Base config: `d=512, n_layers=8, n_heads=8, SwiGLU, RoPE, RMSNorm, weight-tying`.

---

## Phase 1 — Component ablation at ~28M parameters

### Status: Preliminary (single-seed, runs in progress)

| Config | Best val BPC | Params | Steps | Seed | Status |
|--------|-------------|--------|-------|------|--------|
| compose | 3.5316 | 31.1M | 100k | N/A (pre-seed-control) | complete |
| mol | 3.5702 | 31.1M | 100k | N/A (pre-seed-control) | complete |
| mhc (original, n=2, SK) | 3.5736 | 27.8M | 100k | N/A (pre-seed-control) | complete |
| mhc (rerun, n=4, KromHC) | TBD | ~29M | 100k | — | pending |
| baseline | 3.5875 | 27.8M | 100k | N/A (pre-seed-control) | complete |
| baseline_wide | TBD | ~31.1M | 100k | — | pending |
| mol_single (rank=72) | TBD | ~31.1M | 100k | — | pending (seed42 run at rank=64 invalidated — rank corrected to 72) |

**Important caveats:**
- All completed runs predate seed control (added 2026-03-23). Seeds not recorded.
- Multi-seed reruns (3 seeds each) are required for primary claims Q1 and Q2.
- `mol vs. baseline` is NOT a controlled comparison — mol has 3.35M more parameters.
  The controlled comparison for Q1 is `mol vs. baseline_wide`.
  The controlled comparison for Q2 is `mol vs. mol_single`.
- `mhc` failure (lost to baseline by 0.0139 BPC) is explained in
  `references/components/mhc.md §Scale limitations` — scale below validated minimum,
  n=2 streams, SK approximation gap.

### Multi-seed rerun plan (pending)

Requires seed control to be deployed on cloud (merged 2026-03-23). Run order:
1. `baseline` × 3 seeds (42, 1, 2) — establishes variance
2. `baseline_wide` × 3 seeds — Q1 parameter-matched control
3. `mol` × 3 seeds — Q1 treatment
4. `mol_single` × 3 seeds — Q2 comparison

Reporting standard (Melis et al. 2018): a claim is valid if margin > 3× cross-seed std,
or worst seed of new config beats best seed of baseline.

---

## Phase 2 — Outer encoder study

### Status: Not started (pending Phase 1 completion)

Run order: `outer_crl → outer_crl_learned → outer_crl_full → outer_crl_full_learned → outer_transformer → outer_diff_attn → outer_mla → outer_strided → outer_crl_learned_noste`

A1 success criterion: `outer_crl_learned` val BPC < `outer_strided` val BPC at same target compression ratio (0.25), boundary_entropy decreasing.

---

## Phase 3 — 1B scale validation

### Status: Hypothetical (hardware upgrade required)

See `PHASE3_STUDY_DESIGN.md`. Requires A100 minimum. Estimated ~1,500 A100 hours for
unconditional runs (Q4 + Q5). Pending Phase 1 and Phase 2 completion.
