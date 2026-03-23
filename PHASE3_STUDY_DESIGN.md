# MoLE Phase 3 Study Design — 1B Scale

**Status:** Hypothetical. Hardware upgrade required before execution.
**Depends on:** Phase 1 and Phase 2 results (read those first).
**Target scale:** ~1B parameters
**Target hardware:** Single A100 (40 GB) minimum; T4 is technically feasible but
                     practically infeasible (~48 days/run for minimum token budget).

---

## 1. Why Phase 3

Phase 1 (28M params) and Phase 2 (HDC pipeline) were designed to validate architecture
*concepts* cheaply before committing to scale. Phase 3 tests whether those concepts
hold at the scale they were designed for.

Three Phase 1/2 results motivate scaling:

| Phase 1/2 finding | What Phase 3 tests |
|-------------------|--------------------|
| mHC *lost* to baseline at 28M (3.5643 vs 3.5582) | Does mHC work at 1B with n=4 and exact doubly stochastic constraint? |
| MoL *beat* baseline at 28M (margin TBD) | Does the routing advantage widen or narrow at scale? |
| HDC pipeline validated; compute budget concern (14–27 tokens/param) | Does HDC improve BPC at proper Chinchilla-adjacent scale? |

**If Phase 3 shows mHC and MoL both work at 1B, the natural next step is composition**
(mHC + MoL + HDC) at 1B — which is the full MoLE architecture at meaningful scale.

---

## 2. The Four Questions

| Question | Axis | Primary comparison |
|----------|------|--------------------|
| **Q4 — Does mHC work at scale with a correct doubly stochastic implementation?** | Architecture | `mhc_lite_1b` (n=4, exact DS) vs. `baseline_1b` |
| **Q5 — Does MoL routing scale?** | Capacity-matched | `mol_1b` vs. `baseline_wide_1b` (matched total params) |
| **Q6 — Does mHC + MoL compose at 1B?** | Both simultaneously | `compose_1b` vs. `mol_1b` and `mhc_lite_1b` |
| **Q7 — Does HDC improve BPC at proper compute scale?** | Tokens/param | `hdc_1b` vs. `mol_1b` at ≥5 tokens/param |

Q6 and Q7 are **conditional**: run only if Q4 and Q5 respectively show positive results.
Do not train `compose_1b` if either mHC or MoL fails to beat baseline at 1B.

---

## 3. What "Fair" Means at 1B Scale

### Q4: mHC at scale

Phase 1 mHC failure had three compounding causes:
1. Scale below validated minimum (28M vs 1B minimum in literature)
2. n_streams=2 below paper's validated n=4
3. SK approximation gap allowing H_res constraint drift

Phase 3 fixes all three:
- Scale: 1B params (matches OLMo-1B-DHC from original HC paper)
- n_streams=4 (paper's validated default)
- Implementation: **mHC-lite** (arXiv:2601.05732) or **KromHC** (arXiv:2601.21579)
  — both guarantee exact doubly stochasticity, eliminating the SK approximation gap

**Why mHC-lite for n=4 specifically**: 4! = 24 permutation matrices — tractable.
For n=4, KromHC (Kronecker 2×2) and mHC-lite (full Birkhoff-von Neumann) are both
practical. KromHC uses 48% fewer parameters and achieves lower gradient norms;
it is the preferred implementation.

**Control**: `baseline_1b` must match `mhc_lite_1b` in total non-mHC parameter count.
The mHC matrices add a small overhead (~1–3% of total params at 1B); account for this
by verifying param counts before training.

### Q5: MoL routing at scale

Same matched-params principle as Phase 1 Q1:
- `mol_1b`: MoL with n_experts=8, top_k=2, mol_rank=16 (scaled from Phase 1's rank=8)
- `baseline_wide_1b`: dense SwiGLU with d_ff expanded to match `mol_1b` total params

Verify total param counts are within 1% before training begins.

### Q6: Composition (conditional)

Only run if Q4 (mHC > baseline) AND Q5 (mol > baseline_wide) both hold.
`compose_1b` = mHC-lite (n=4) + MoL (n_experts=8, top_k=2, mol_rank=16).
Compare against `mol_1b` (not `baseline_1b`) to isolate mHC's additive contribution.

### Q7: HDC at proper compute scale (conditional)

Phase 2's compute budget was ~14–27 tokens/param — well below the inference-optimal
target of ~192 tokens/param (2024 scaling laws). Phase 3 HDC runs at a minimum of
**5 tokens/param** (5B tokens for 1B model).

Only run if Phase 2 `hdc_gate` validated the pipeline (val BPC < mol Phase 2 baseline).

---

## 4. Model Architecture

### Baseline 1B configuration

| Hyperparameter | Value | Notes |
|---------------|-------|-------|
| d | 2048 | Standard 1B width |
| n_layers | 24 | Standard 1B depth |
| n_heads | 16 | 128 dim/head |
| d_ff | 5504 | `int(2048 × 8/3)` rounded to 64 |
| vocab_size | TBD | Retrain BPE tokenizer at larger vocab (32k recommended) or reuse 4096 |
| seq_len | 1024 | Doubled from Phase 1 |
| weight tying | Yes | embed ↔ lm_head |
| norm | RMSNorm | Unchanged |
| pos encoding | RoPE | Unchanged |
| activation | SwiGLU | Unchanged |
| n_params (approx) | ~1.1B | Verify before training |

**Tokenizer note**: Phase 1 used BPE vocab=4096. At 1B scale on a larger dataset
(FineWeb/OpenWebText), a 32k vocabulary is standard and improves token efficiency.
This requires a new tokenizer trained on the Phase 3 dataset. If Phase 3 uses the same
4096 vocab, results are more comparable to Phase 1 but suboptimal for 1B pretraining.
Decision should be made and locked before any Phase 3 run begins.

### MoL 1B configuration

| Change from baseline | Value |
|---------------------|-------|
| FFN type | MoLFFN (same as Phase 1) |
| n_experts | 8 |
| top_k | 2 |
| mol_rank | 16 (scaled from Phase 1's 8 to keep rank/d ratio constant) |
| Total params | ~1.15B (verify — must match baseline_wide_1b within 1%) |

### mHC-lite 1B configuration

| Change from baseline | Value |
|---------------------|-------|
| n_streams | 4 (paper's validated default) |
| H_res implementation | KromHC exact (2×2 Kronecker, n=4=2×2) |
| H_pre / H_post | Unchanged from Phase 1 (softmax, matching reference code) |
| mHC overhead | ~2–3% additional params vs baseline_1b |

**Implementation requirement**: KromHC/mHC-lite is not yet implemented in this codebase.
Phase 3 requires a new `HyperConnectionExact` class in `phase1/model.py` (or a `phase3/`
directory) replacing the existing `HyperConnection` which uses approximate SK.
This is a pre-condition for Phase 3 execution.

---

## 5. Hardware Requirements

| Hardware | Tokens/s (est.) | Time per 10B-token run | Verdict |
|----------|----------------|----------------------|---------|
| Single T4 (16GB) | ~1,200 | ~97 days | **Infeasible** |
| Single A100 40GB | ~50,000 | ~56 hours | ✓ Minimum viable |
| Single A100 80GB | ~60,000 | ~46 hours | ✓ Preferred |
| Single H100 80GB | ~150,000 | ~19 hours | ✓ Ideal |

A single A100 is the minimum viable hardware. All Phase 3 estimates below assume A100 40GB.

### Memory configuration (A100, 1B model)

With 8-bit optimizer + gradient checkpointing + Flash Attention 2 (A100 supports FA2):

| Component | VRAM |
|-----------|------|
| FP32 params (BF16 on A100) | 2 GB |
| 8-bit optimizer states | ~2 GB |
| FP32 master weights | 4 GB |
| BF16 gradients | 2 GB |
| Activations (B=8, seq=1024, grad ckpt) | ~6–8 GB |
| **Total** | **~16–18 GB** |

Fits comfortably in 40GB A100 with room for larger batch sizes.
**On A100: use BF16** (Ampere supports native BF16 tensor cores — no GradScaler needed,
existing auto-detection code in train.py handles this correctly).

---

## 6. Dataset

Phase 1/2 used WikiText-103 (~100M tokens). **WikiText-103 is not suitable for 1B
pretraining** — the entire dataset is only 0.1 tokens/param.

### Recommended: FineWeb 10BT sample

- **Source**: HuggingFace `HuggingFaceFW/fineweb` (10B token curated subset)
- **Size**: 10 billion tokens
- **Quality**: Specifically designed and validated for 1B-scale architecture ablations
  (used in the mHC-lite paper for all experiments)
- **Tokens/param at 1.1B model**: ~9 tokens/param — approaching but below Chinchilla

### Alternative: OpenWebText

- **Source**: HuggingFace `Skylion007/openwebtext`
- **Size**: ~38 billion tokens
- **Advantage**: Covers full Chinchilla budget (38B >> 20B optimal for 1B)
- **Disadvantage**: Noisier than FineWeb; slower data pipeline setup

### Recommendation

Use **FineWeb 10BT** for Q4/Q5 ablations (speed + reproducibility against mHC-lite paper).
Use **OpenWebText** if budget allows Chinchilla-optimal runs for final configs.

### Data pipeline requirements

- New `TokenDataset` implementation for streaming large datasets
  (WikiText-103 fit in memory; FineWeb 10BT does not)
- New BPE tokenizer training (if upgrading to vocab=32k)
- `set_dataset("fineweb_10bt")` and `set_dataset("openwebtext")` in `utils/data.py`

---

## 7. Required New Code

Phase 3 cannot run on the current codebase without these additions:

| Component | File | Status |
|-----------|------|--------|
| KromHC / mHC-lite exact H_res implementation | `phase3/model.py` or `phase1/model.py` | **Not implemented** |
| 8-bit optimizer (bitsandbytes integration) | `phase3/train.py` | **Not implemented** |
| Flash Attention 2 (FA2) | `phase3/model.py` | **Not implemented** |
| Streaming data loader (FineWeb / OpenWebText) | `utils/data.py` | **Not implemented** |
| New BPE tokenizer (32k vocab, optional) | `utils/tokenizer.py` | **Not implemented** |
| `phase3/train.py` training script | `phase3/train.py` | **Not implemented** |
| `run_phase3.sh` | root | **Not implemented** |

**Verification requirement**: Before any Phase 3 training, the KromHC implementation
must be verified against arXiv:2601.21579 using the same Phase B' protocol used for
Phase 1/2 components. Add `references/components/kromhc.md` spec file before implementing.

---

## 8. Training Configuration

| Hyperparameter | Value | Notes |
|---------------|-------|-------|
| total_steps | 500k (FineWeb 10BT) or 1.2M (OpenWebText 20B) | Compute from batch × seq |
| batch_size | 32 (effective) | micro-batch=4, grad accumulation=8 |
| seq_len | 1024 | Doubled from Phase 1 |
| max_lr | 3e-4 | Same as Phase 1; monitor — 1B may need 1e-4 |
| lr schedule | cosine + warmup | Warmup = 1% of total steps |
| optimizer | AdamW 8-bit (bitsandbytes) | bnb.optim.AdamW8bit |
| weight_decay | 0.1 (2D+ params), 0.0 (1D) | Same ndim split as Phase 1 |
| grad_clip | 1.0 | Same as Phase 1 |
| dtype | BF16 (A100) | GradScaler not needed |
| gradient_checkpointing | True | Required for memory |
| attention | Flash Attention 2 | FA2 on A100; FA1 on T4 if needed |
| eval_interval | 5000 steps | Every ~160M tokens |
| log_interval | 100 steps | Same as Phase 1 |

**LR note**: The optimal LR for 1B models is empirically ~1e-4 to 3e-4. Phase 1 used
3e-4 for 28M. Chinchilla scaling laws suggest LR ∝ 1/√N, so 1B → ~3.4× smaller than
28M → optimal ~9e-5. **Run a brief LR sweep (1e-4, 2e-4, 3e-4) on `baseline_1b`
before committing all configs to a fixed LR.** This is a Phase 3 prerequisite.

---

## 9. Statistical Requirements

Same standard as Phase 1 (Melis et al. 2018):

**Minimum 3 seeds** for any config making a primary claim (Q4, Q5).
**1 seed** for conditional configs (Q6, Q7) until positive results confirmed.

**Reporting**: Mean ± std across seeds. A claim is only valid if:
- Margin > 3× cross-seed standard deviation, OR
- Worst seed of new config beats best seed of baseline

At 1B scale, cross-seed variance is typically smaller than at 28M (larger models are more
stable), so 3 seeds should be sufficient to resolve meaningful differences.

---

## 10. Compute Budget Estimate (A100 40GB)

| Config | Seeds | Steps | Est. time/run | Total A100 hours |
|--------|-------|-------|--------------|-----------------|
| LR sweep (baseline_1b × 3 LRs) | 1 | 50k | ~3 hrs | 9 hrs |
| `baseline_1b` | 3 | 500k | ~28 hrs | 84 hrs |
| `baseline_wide_1b` | 3 | 500k | ~28 hrs | 84 hrs |
| `mol_1b` | 3 | 500k | ~30 hrs | 90 hrs |
| `mhc_lite_1b` | 3 | 500k | ~30 hrs | 90 hrs |
| `compose_1b` (conditional) | 3 | 500k | ~32 hrs | 96 hrs |
| `hdc_1b` (conditional) | 1 | 500k | ~35 hrs | 35 hrs |
| **Total (without conditionals)** | | | | **~357 hrs** |
| **Total (with all conditionals)** | | | | **~488 hrs** |

At A100 cloud pricing (~$2–4/hr), unconditional runs: **~$700–1,400**.
With all conditionals: **~$1,000–2,000**.

Note: estimates assume FineWeb 10BT (500k steps). OpenWebText at 20B tokens ≈ 2.4× more
steps and cost.

---

## 11. Execution Order

```
Phase 3 prerequisites (before any training):
  1. Implement KromHC in phase3/model.py
  2. Write references/components/kromhc.md spec
  3. Run Phase B' verification on KromHC implementation
  4. Implement streaming data loader for FineWeb 10BT
  5. Train BPE tokenizer (if upgrading to 32k vocab)
  6. Implement 8-bit optimizer + Flash Attention 2 in phase3/train.py
  7. Write run_phase3.sh with smoke test gate

Phase 3 execution:
  1. LR sweep: baseline_1b × 3 LRs (50k steps each, 1 seed) — pick max_lr
  2. baseline_1b × 3 seeds (establishes variance)
  3. baseline_wide_1b × 3 seeds (Q5 control)
  4. mol_1b × 3 seeds (Q5 treatment)
  5. mhc_lite_1b × 3 seeds (Q4 treatment)
  6. [conditional on Q4+Q5 positive] compose_1b × 3 seeds (Q6)
  7. [conditional on Phase 2 HDC validated] hdc_1b × 1 seed (Q7)
```

---

## 12. What Phase 3 Can and Cannot Conclude

### Can conclude

- Whether mHC-lite (exact doubly stochastic, n=4) improves over baseline at 1B
- Whether MoL routing mechanism adds value beyond capacity at 1B scale
- Whether the 28M → 1B scaling trend for each architecture is positive/negative/flat
- A two-point scaling story (28M and 1B) with consistent experimental controls
- Whether KromHC's lower gradient norms translate to better final BPC in this codebase

### Cannot conclude

- Transfer to >1B without a third data point
- That results generalize across datasets (FineWeb ≠ WikiText-103)
- Absolute rankings vs. published models (tokenizer, data, and hyperparameter differences)
- Anything about mHC at 1B if Phase 3 uses approximate SK — **the implementation must
  be KromHC or mHC-lite, not the existing HyperConnection with 10-iteration Sinkhorn**

---

## 13. Relationship to Prior Phases

```
Phase 1 (28M, WikiText-103, BPE-4096, 100k steps)
  → establishes: baseline BPC, MoL behavior, mHC failure mode at small scale
  → feeds Phase 3: mol_rank scaling factor, LR baseline, grad norm reference

Phase 2 (1–15M HDC wrapper, WikiText-103, 50k steps)
  → establishes: HDC pipeline correctness, compression ratio behavior
  → feeds Phase 3: whether Q7 (HDC at 1B) is worth running

Phase 3 (1B, FineWeb/OWT, BPE-4096 or 32k, 500k steps)
  → tests: architectures at design scale with verified implementation
  → first point in a proper scaling curve
  → prerequisite for any claim that MoLE components compose beneficially
```

Cross-phase BPC comparisons are not valid (different scale, tokenizer, dataset).
Phase 3 results stand alone; Phase 1/2 provide motivation and failure-mode context.

---

## Sources

- Hoffmann et al., "Training Compute-Optimal LLMs" (Chinchilla, 2022) — token budget
- Dettmers et al., "8-bit Optimizers via Block-wise Quantization" (arXiv:2110.02861) —
  8-bit AdamW accuracy parity at 1B scale
- Yang & Gao, "mHC-lite" (arXiv:2601.05732) — FineWeb 10BT as 1B ablation benchmark;
  mHC-lite exact construction
- Zhou et al., "KromHC" (arXiv:2601.21579) — preferred exact H_res implementation at n=4
- Penedo et al., "FineWeb" (arXiv:2406.17557) — dataset choice and 1B ablation methodology
- Dao et al., "Flash Attention 2" (arXiv:2307.08691) — FA2 on A100
- Wortsman et al., "mHC original / HC" (arXiv:2409.19606) — OLMo-1B-DHC as comparison target
