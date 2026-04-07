# Results

All results are on WikiText-103-raw, BPE tokenizer (vocab=4096, sentencepiece), seq_len=256.
Base config: `d=512, n_layers=8, n_heads=8, SwiGLU, RoPE, RMSNorm, weight-tying`.

---

## Phase 1 — Component ablation at ~28M parameters

### Status: Seed42 runs complete. Multi-seed reruns pending for primary claims Q1/Q2.

#### Seed42 results (primary dataset — all at d=512, 100k steps, seed=42)

| Config | Best val BPC | Params | Notes |
|--------|-------------|--------|-------|
| diff_attn_seed42 | 3.5131 | ~33.5M | Best overall; ~25% more attn params than baseline — not param-matched |
| baseline_wide_seed42 | 3.5355 | ~31.1M | Q1 control (param-matched to mol via d_ff=1600) |
| compose_seed42 | 3.5416 | 31.1M | grad_norm rising 0.80→1.37; diagnostic pending |
| mhc_seed42 | 3.5428 | ~29M | n=4, KromHC; grad_norm rising — do not compose until diagnosed |
| mol_seed42 | 3.5441 | 31.1M | Phase 2 inner network |
| diff_mla_seed42 | 3.5474 | ~31M | 1 loss spike; KV bottleneck still limiting |
| mol_single_seed42 | 3.5516 | ~31.1M | rank=72 (capacity-matched to mol); prior rank=64 run invalidated |
| baseline_seed42 | 3.5605 | 27.8M | Reference |
| mla_seed42 | 3.5859 | ~28.5M | KV compression (d_c=128) too lossy at this scale |

#### Pre-seed-control runs (seeds not recorded; for historical reference only)

| Config | Best val BPC | Params |
|--------|-------------|--------|
| compose | 3.5316 | 31.1M |
| mol | 3.5702 | 31.1M |
| mhc (n=2, Sinkhorn) | 3.5736 | 27.8M |
| baseline | 3.5875 | 27.8M |

**Important caveats:**
- **All seed42 results are single-seed.** Multi-seed reruns (3 seeds each) are required
  for primary claims Q1 and Q2 per Melis et al. 2018 reporting standard.
- **Q1:** baseline_wide (3.5355) beats mol (3.5441) by 0.0086 BPC — capacity dominates
  routing at this scale. Margin is within typical cross-seed variance; result is preliminary.
- **Q2:** mol (3.5441) beats mol_single (3.5516) by 0.0075 BPC — routing adds modest value.
  Also preliminary pending multi-seed confirmation.
- **diff_attn capacity caveat:** diff_attn's result (3.5131) reflects architecture +
  capacity advantage combined (~25% more attention params). Not a controlled comparison.
- **mhc grad_norm:** rising 0.80→1.37 over 100k steps with no stabilization. Do not
  compose mHC until diagnosed. Diagnostic: `--config mhc --max_lr 1.5e-4 --total_steps 25000`.

### Multi-seed rerun plan (pending)

Reporting standard (Melis et al. 2018): a claim is valid if margin > 3× cross-seed std,
or worst seed of new config beats best seed of baseline.

Run order:
1. `baseline` × 3 seeds (42, 1, 2) — establishes variance
2. `baseline_wide` × 3 seeds — Q1 parameter-matched control
3. `mol` × 3 seeds — Q1 treatment
4. `mol_single` × 3 seeds — Q2 comparison

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
