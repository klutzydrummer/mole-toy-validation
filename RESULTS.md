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
| mhc | 3.5736 | 27.8M | 100k | N/A (pre-seed-control) | complete |
| baseline | 3.5875 | 27.8M | 100k | N/A (pre-seed-control) | complete |
| baseline_wide | TBD | ~31.1M | 100k | — | pending |
| mol_single | TBD | ~31.1M | 100k | — | pending |

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

## Phase 2 — HDC compression pipeline

### Status: Not started (pending Phase 1 completion)

Run order: `hdc_rulebased → hdc_gate → hdc_stride → hdc_r2/r8`

A1 success criterion: `hdc_gate` val BPC < 3.5702 (mol Phase 1 baseline),
compression ratio stable near 0.25, boundary_entropy decreasing.

---

## Phase 3 — 1B scale validation

### Status: Hypothetical (hardware upgrade required)

See `PHASE3_STUDY_DESIGN.md`. Requires A100 minimum. Estimated ~1,500 A100 hours for
unconditional runs (Q4 + Q5). Pending Phase 1 and Phase 2 completion.
