# MoLE Toy Validation — Study Design

This document defines the experimental protocol for the MoLE toy validation. It governs
what we train, how we compare results, and what conclusions we can draw. Every config
addition or change to training procedure should be evaluated against this document first.

---

## 1. Research Questions

The project is organized into eight research questions across five study groups. Q1–Q3 were
the original study design. Q4–Q7 were added as the project scope expanded in April 2026.
Q8 was added with Study E.

### Core MoLE study (Q1–Q2): Study A

| Question | Axis | Primary comparison |
|----------|------|--------------------|
| **Q1 — Does MoL's routing mechanism help, beyond just having more parameters?** | Total trainable params | `mol` vs. `baseline_wide` (matched total params) |
| **Q2 — Is routing over adapters better than a single high-rank adapter?** | Total LoRA params | `mol` vs. `mol_single` (same total LoRA params, no routing) |

### Outer encoder study (Q3): Study F (Phase 2)

| Question | Axis | Primary comparison |
|----------|------|--------------------|
| **Q3 — Does content-aware compression improve over fixed-stride compression?** | Active FLOPs + training steps | `outer_crl_learned` vs. `outer_crl_fixed_stride` (same encoder, same steps, same target compression ratio) |

### Attention variant study (Q4–Q5): Study B

| Question | Axis | Primary comparison |
|----------|------|--------------------|
| **Q4 — Exploratory: combined effect of differential attention mechanism + increased attention capacity** | Attention mechanism + capacity | `diff_attn` vs. `baseline` (not controlled: +25% attn params) |
| **Q5 — Does MLA KV compression hurt at this scale?** | KV bottleneck dimension | `mla` vs. `baseline`; `diff_mla` vs. `diff_attn` |

**Q4 framing:** `diff_attn` is NOT a controlled comparison vs. `baseline`. Doubled Q heads add ~25% more attention parameters (+2.1M = 8 layers × d²). Any BPC difference reflects architecture + capacity combined and cannot attribute improvement to the differential attention mechanism alone. Q4 is an **exploratory observation**, not a hypothesis-driven research question. It is listed here for completeness and to document the confound explicitly. Do not interpret Q4 results as evidence for or against differential attention as a mechanism. **Q5 prior observation:** KV compression at d_c=128 appeared lossy in prior runs; to be confirmed with current codebase.

### go-mHC composition study (Q6): Study C

| Question | Axis | Primary comparison |
|----------|------|--------------------|
| **Q6 — Does mHC stream mixing improve when combined with specialized attention?** | Attention × stream topology | `diff_mhc` vs. `diff_attn`; `mla_mhc` vs. `mla`; `diff_mla_mhc` vs. `diff_mla` |

**Context:** mHC alone shows rising grad norm (0.80→1.37 over 100k steps). Study C tests whether the grad norm issue is specific to the baseline attention or is architectural. The three attention variants give a gradient topology landscape for mHC.

### nGPT hyperspherical study (Q7–Q8): Study D + Study E

| Question | Axis | Primary comparison |
|----------|------|--------------------|
| **Q7 — Does hyperspherical constraint improve convergence/BPC?** | Normalization topology | `ngpt` vs. `baseline`; `ngpt_mla` vs. `mla`; `ngpt_diff_attn` vs. `diff_attn` |
| **Q8 — Does multi-sphere stream mixing outperform wrap-sublayer composition?** | Geometric inductive bias | `ngpt_mhc_a` vs. `ngpt_mhc_c` (same params, different manifold structure) |

**Q7 context:** nGPT constrains hidden states to S^{d-1}. Prior work reports 4–20× convergence
speedup on language modeling (nGPT paper, arXiv:2410.01131) — but these are step counts, not
wall-clock time, and the speedup may not hold at all scales or tokenizations.

**Q8 context:** Two novel compositions, same 28M params, different geometric structure:
- `ngpt_mhc_a` (Option A): n=4 streams each on S^{d-1}; go-mHC mixing preserves multi-sphere
- `ngpt_mhc_c` (Option C): full mHC block treated as single sublayer; sphere enforced once

Q1 and Q2 are Phase 1 questions. Q3 is the Phase 2 A1 question.
The original Phase 1 comparison (mol vs. baseline at different total param counts) does not
cleanly answer any of these — it conflates architecture and capacity. This document fixes that.

---

## 2. What "Fair" Means for Each Question

### Q1: Routing mechanism vs. capacity

MoL at d=512, n_experts=8, top_k=2, rank=8 has **31.1M total params**.
The current baseline has **27.8M total params** — 3.35M fewer.

A fair test of whether the MoL *architecture* (routing + specialization) adds value beyond
raw capacity requires a dense baseline with the same total parameter count.

**`baseline_wide`**: SwiGLU with d_ff=1600 (vs. default 1408).

Verified param counts (from `utils/shape_check.py`):
- `baseline_wide`: **30,155,264** (30.2M)
- `mol`:           **31,146,496** (31.1M)
- Gap: 991,232 params, **3.2%** — within the 5% Q1 tolerance.

Note: baseline_wide has slightly *fewer* params than mol. This makes Q1 conservative:
if baseline_wide beats mol, capacity explains the gain; if mol wins, the routing benefit
is understated (mol had the capacity disadvantage).

### Q2: Routing vs. high-rank single adapter

MoL has 9 adapter sets (1 shared + 8 experts) each at rank=8, for 3 projections (gate/up/down).
Total LoRA params per layer = 9 × 3 × (d×rank + rank×d_ff) = 9 × 46,080 = 414,720.
An equivalent single LoRA has rank = 9 × 8 = **rank-72** (exact capacity match).

**`mol_single`**: MoLFFN with a single LoRA at rank=72 applied on top of base SwiGLU, no
routing. Same base FFN, same total adapter params (exact match), no expert mechanism.

This tests: is the routing + specialization mechanism responsible for MoL's gains, or is it
simply that more LoRA capacity (regardless of how it is structured) improves the model?

### Q3: Content-aware routing vs. fixed routing (Phase 2)

The primary comparison is `outer_crl_learned` vs. `outer_crl_fixed_stride` (same CRL encoder,
different routing policy). This is the clean controlled comparison: routing quality isolated
from encoder quality. Both run for 50k steps at target_rate=0.25.

`outer_strided` (identity encoder + fixed_stride) is a secondary lower bound — useful for
bounding the benefit of the encoder itself, but **not** the Q3 controlled comparison because
it conflates "no encoder" with "fixed routing."

**Important caveats for Q3:**
- **Iso-step, not iso-FLOP:** `outer_crl_learned` runs the CRL encoder (282K params, 3 CRL
  layers) on every step. `outer_crl_fixed_stride` runs the same encoder but skips the
  learned router forward pass. FLOPs per step differ by the router computation cost.
  Any positive result for learned routing cannot isolate routing benefit from slightly
  higher per-step compute. Flag this explicitly in any write-up.
- **Iso-step, not iso-FLOP (outer_strided):** `outer_strided` has zero encoder computation.
  If it outperforms `outer_crl_learned`, the interpretation is ambiguous — the CRL encoder
  may be adding noise, not signal. The FLOPs difference is large (~282K params worth).

**Do not** compare `outer_crl_learned` BPC directly to `mol` BPC to claim Phase 2 improves
on Phase 1 — they train for different step counts (50k vs. 100k). Use
`compression_ratio × FLOPs` framing or note the step count difference explicitly.

---

## 3. Active Parameters and FLOPs (per token)

For transparency, the following should be computed and reported for every config.
Param counts are unique parameters (tied lm_head/embedding counted once); verified via
`utils/shape_check.py` at d=512, vocab=4096.

### Study A — MoLE core

| Config | Total params | Active params/token | Approx. FFN FLOPs/token |
|--------|-------------|--------------------|-----------------------|
| `baseline` | 27.8M | 27.8M | 2 × d × d_ff = 1.44M |
| `baseline_wide` | 30.2M | 30.2M | 2 × 512 × 1600 = 1.64M |
| `mol` | 31.1M | ~28.9M† | ~1.50M† |
| `mol_single` | 31.1M | 31.1M | ~1.48M† |
| `mhc` | 28.0M | 28.0M | 1.44M + go-mHC overhead‡ |
| `compose` | 31.4M | ~29.2M† | ~1.50M + go-mHC overhead‡ |

†MoL active params = base FFN + shared LoRA + 2 selected expert LoRAs + router.
6 unselected expert LoRAs are trained but inactive per token (2,211,840 inactive params).
The shared base FFN dominates; active LoRA overhead is ~6% of total FFN FLOPs.

‡go-mHC overhead per token: mean-pool → Linear(512, 28) → skew → Cayley solve (8×8) →
block Frobenius. n=4, s=2, ns=8. Plus H_pre/H_post softmax (4 values each). Not negligible
but sub-dominant vs. attention and FFN at d=512.

**Key insight**: MoL's active-param/FLOPs cost is close to baseline (~6% overhead), not 12%
more. The extra total params are almost entirely in non-selected expert LoRAs that are trained
but inactive at inference. Q1 (capacity control) matters more than FLOPs control for
interpreting MoL's result.

### Study B — Attention variants

| Config | Total params | Active params/token | Approx. FFN FLOPs/token | Attention note |
|--------|-------------|--------------------|-----------------------|---------------|
| `mla` | 27.4M | 27.4M | 1.44M | KV latent d_c=128 (25% of d) |
| `diff_attn` | 29.9M | 29.9M | 1.44M | +2.1M vs baseline: doubled Q heads |
| `diff_mla` | 26.3M | 26.3M | 1.44M | MLA KV compression + diff Q (no per-head norm) |

**diff_attn capacity confound:** The +2.1M over baseline (8 layers × d² extra Q params = 2,097,152)
is an uncontrolled capacity advantage. Q4 comparisons must acknowledge this — any BPC
improvement over `baseline` reflects both architecture and capacity.

### Study C — go-mHC compositions

| Config | Total params | Active params/token | Approx. FFN FLOPs/token |
|--------|-------------|--------------------|-----------------------|
| `diff_mhc` | 30.2M | 30.2M | 1.44M + go-mHC overhead |
| `mla_mhc` | 27.6M | 27.6M | 1.44M + go-mHC overhead |
| `diff_mla_mhc` | 26.5M | 26.5M | 1.44M + go-mHC overhead |

Each adds ~230K go-mHC params over its non-mHC counterpart (8 layers × 2 HyperConnections
× ~14,400 params each, plus stream_collapse_logits).

### Study D — nGPT hyperspherical

| Config | Total params | Active params/token | Approx. FFN FLOPs/token |
|--------|-------------|--------------------|-----------------------|
| `ngpt` | 27.8M | 27.8M | 1.44M |
| `ngpt_mla` | 27.4M | 27.4M | 1.44M |
| `ngpt_diff_attn` | 29.9M | 29.9M | 1.44M |

nGPT adds ~8,200 params over the corresponding non-nGPT config: 8 layers × 2 sublayers ×
d=512 α parameters + 1 scalar s_z. Negligible overhead.

### Study E — Multi-sphere compositions

| Config | Total params | Active params/token | Approx. FFN FLOPs/token |
|--------|-------------|--------------------|-----------------------|
| `ngpt_mhc_a` | 28.1M | 28.1M | 1.44M + go-mHC overhead |
| `ngpt_mhc_c` | 28.1M | 28.1M | 1.44M + go-mHC overhead |

Identical param counts — the two variants differ in where the L2Norm renorm is applied
within the mHC block, not in the number of parameters.

---

## 4. Required Configs

### Study A — MoLE core (Q1, Q2)

| Config | Purpose | Notes |
|--------|---------|-------|
| `baseline` | Dense SwiGLU reference (27.8M) | pending re-run |
| `baseline_wide` | Capacity-matched dense baseline (30.2M) | pending re-run; 3-seed required for Q1 claim |
| `mol` | MoL routing (31.1M, top-2 of 8) | pending re-run; 3-seed required for Q1/Q2 |
| `mol_single` | Single rank-72 LoRA, no routing (31.1M) | pending re-run; 3-seed required for Q2 |
| `mhc` | Hyper-connections on baseline (28.0M) | pending re-run with go-mHC |
| `compose` | mHC + MoL (31.4M) | pending re-run with go-mHC |

### Study B — Attention variants (Q4, Q5)

| Config | Purpose | Notes |
|--------|---------|-------|
| `mla` | MLA KV compression (27.4M, d_c=128) | pending re-run |
| `diff_attn` | Differential attention V2 (29.9M; capacity confound: +25% attn params) | pending re-run |
| `diff_mla` | DiffMLA V2 + MLA composition (26.3M, novel) | pending re-run |

### Study C — go-mHC compositions (Q6)

| Config | Purpose | Notes |
|--------|---------|-------|
| `diff_mhc` | go-mHC + Diff Attn V2 | pending |
| `mla_mhc` | go-mHC + MLA | pending |
| `diff_mla_mhc` | go-mHC + DiffMLA | pending |

### Study D — nGPT hyperspherical (Q7)

| Config | Purpose | Notes |
|--------|---------|-------|
| `ngpt` | nGPT on baseline (hyperspherical constraint) | pending |
| `ngpt_mla` | nGPT + MLA | pending |
| `ngpt_diff_attn` | nGPT + Diff Attn V2 | pending |

### Study E — Multi-sphere compositions (Q8)

| Config | Purpose | Notes |
|--------|---------|-------|
| `ngpt_mhc_a` | Multi-sphere (Option A): n=4 streams, each on S^{d-1} | pending |
| `ngpt_mhc_c` | Wrap-sublayer (Option C): sphere enforced once after full mHC | pending |

### Scaling study (cross-cutting)

5 configs × {d=256, d=768}: `baseline`, `mla`, `diff_attn`, `diff_mla`, `mol`.
Checkpoint prefix: `{cfg}_d{d}_seed42`.

**Research question (restated):** How does each architecture's BPC, relative to the
`baseline` at the same scale, change as model size scales from d=256 (~6M) to d=512
(~28M) to d=768 (~58M)? Specifically: does MLA's KV compression deficit (Q5) grow,
shrink, or remain constant in absolute BPC terms as model size increases?

**What this design cannot answer:** d_c/d=25% is held constant at all scales
(d=256→d_c=64, d=512→d_c=128, d=768→d_c=192). Both the ratio and the absolute
bottleneck dimension increase proportionally with d. The study cannot isolate whether
MLA's performance is ratio-driven or absolute-dimension-driven — both effects are
confounded. To isolate them would require a fixed-d_c run (e.g., d=768 with d_c=128
keeps absolute dimension constant while ratio shrinks). That run is not scheduled.

**Correct framing for any write-up:** Report "MLA BPC deficit vs. baseline at each
scale" and note whether the deficit is stable, growing, or shrinking. Do not claim
this isolates ratio vs. absolute dimension effects.

### Phase 2 — outer encoder study (Q3)

The Q3 controlled comparison is `outer_crl_learned` vs. `outer_crl_fixed_stride` (same
encoder, different routing). `outer_strided` is a secondary no-encoder lower bound.
10 configs total (see phase2/model.py CONFIGS).

---

## 5. Statistical Requirements

Based on literature consensus (Melis et al. 2018, NLP ablation practice):

**Minimum**: 3 seeds for any config used in a primary claim.
**Primary claims** (Q1, Q2): `baseline`, `baseline_wide`, `mol`, `mol_single` — all need 3 seeds.
**Supporting configs** (`mhc`, `compose`): 1 seed until optimization issues are resolved.
**Phase 2**: 3 seeds for both `outer_crl_learned` (treatment) AND `outer_crl_fixed_stride`
(control). Running the treatment at 3 seeds and the control at 1 seed makes the
"margin > 3× std" criterion asymmetric — variance is known for the treatment but unknown
for the control. An n=1 control is not a defensible control arm. All other Phase 2 configs:
1 seed.

**Checkpoint policy for multi-seed comparisons**: Use **final-step BPC** (not best-checkpoint BPC) when computing cross-seed standard deviation and the "margin > 3× std" criterion. Best-checkpoint selection introduces an implicit hyperparameter search over checkpoint index; variance computed over best-BPC values across seeds conflates architecture quality with checkpoint selection luck. Best-checkpoint BPC may be reported separately as a secondary metric but must not be used for the primary margin calculation.

**Reporting**: Always report mean ± std across seeds. A result is only claimable if:
- The margin is larger than 3× the cross-seed standard deviation, OR
- The worst seed of the new config beats the best seed of the baseline.

**Do not** report only best-seed results.

---

## 6. Metrics to Report for Every Config

### Per run (final values)
- `final_step_bpc` — **primary metric for statistical comparisons** (used in §5 margin calculations; see checkpoint policy note in §5)
- `best_val_bpc` — secondary metric; reported for reference and to track convergence trajectory
- `n_params_total` — total parameter count
- `n_params_active` — active params per forward pass (compute from architecture)
- `final_grad_norm` — last 1k steps mean
- `grad_norm_trend` — first 10k steps mean vs. last 10k steps mean (rising = instability signal)

### Training dynamics (curves, not scalars)
- Val BPC vs. training step curve (every eval_interval=2500 steps)
- Grad norm trajectory (every log_interval=100 steps)
- For MoL configs: expert utilization entropy per layer at convergence

### For MoL configs only
- Expert load balance (routing entropy): should be close to log(8)=2.08 bits at convergence
- Expert counts distribution: report as a table in checkpoints/report.md

### For Study D and E configs (nGPT and nGPT+mHC)
nGPT's primary published claim is convergence speedup (4–20× fewer steps, arXiv:2410.01131),
not final BPC at matched step count. A step-100k BPC comparison may miss the speedup
entirely if both models saturate before 100k steps.

**Required additional reporting for `ngpt`, `ngpt_mla`, `ngpt_diff_attn`, `ngpt_mhc_a`,
`ngpt_mhc_c` and their non-nGPT matched baselines:**
- BPC at steps 5k, 10k, 25k, 50k, 100k (already logged every 2500 steps; extract from JSONL)
- **Steps to reach baseline final BPC** — the step at which the nGPT config first matches
  its matched baseline's step-100k BPC. If never reached, report "not achieved within 100k."

This metric distinguishes "nGPT converges faster but saturates at the same BPC" from
"nGPT reaches a better final BPC" from "nGPT does not help at this scale." These are
different conclusions with different implications.

### For Phase 2 HDC configs
- `compression_ratio` mean and std across training steps
- `boundary_entropy` trajectory
- `boundary_bpc` vs. `mid_chunk_bpc` gap at convergence

---

## 7. Hyperparameter Controls

All configs must use identical hyperparameters unless the hyperparameter itself is the
thing being ablated:

| Hyperparameter | Value | Locked? |
|---------------|-------|---------|
| d | 512 | Yes |
| n_heads | 8 | Yes |
| n_layers | 8 | Yes |
| total_steps | 100k (Phase 1), 50k (Phase 2) | Yes |
| batch_size | 32 | Yes |
| seq_len | 256 | Yes |
| max_lr | 3e-4 | Yes |
| optimizer | AdamW (β1=0.9, β2=0.95, wd=0.1) | Yes |
| lr schedule | cosine warmup | Yes |
| grad clip | 1.0 | Yes |
| dtype | AMP float16 | Yes |
| mol_rank | 8 | Yes |

**The LR is not tuned per-architecture.** If mHC needs a different LR to be stable, that is
a finding about mHC's optimization landscape — it should be reported as such, not used to
give mHC a silent advantage.

---

## 8. What We Can and Cannot Conclude at This Scale

### Can conclude
- Relative ranking of architectures on WikiText-103 BPE at d=512 (if margin > 3σ)
- Which architectures are unstable (rising grad norm, expert collapse, NaN)
- Sample efficiency: which configs reach a given BPC threshold in fewer steps
- Whether MoL's routing mechanism adds value over capacity alone (Q1) and over high-rank adapters (Q2)
- Whether content-aware HDC compression adds value over fixed-stride (Q3)

### Cannot conclude
- That results transfer to larger scale without a second data point at a different model size
- Absolute BPC rankings vs. published numbers (different tokenizer, sequence length, vocab)
- That mHC is definitively broken from a single rising grad norm observation — possible causes include LR sensitivity, n=4 stream initialization variance, or a fundamental instability in the Cayley parameterization at this scale; investigate before concluding
- Cross-phase BPC comparisons (Phase 1 at 100k steps ≠ Phase 2 at 50k steps)

---

## 9. Execution Order

Run studies in dependency order. Later studies may be conditioned on earlier results.

```
Study A — MoLE core (Q1, Q2):
  bash run_experiments.sh study_mole
  Re-running from scratch with current codebase. Multi-seed required before Q1/Q2 claims.

Study B — Attention variants (Q4, Q5):
  bash run_experiments.sh study_attention
  Re-running from scratch. Single-seed sufficient for exploratory Q4/Q5 observations.

Study C — go-mHC compositions (Q6):
  bash run_experiments.sh study_mhc_compose
  Pending. Interpret after Study B results confirmed.
  Diagnose mHC grad norm before drawing conclusions from Study C.

Study D — nGPT hyperspherical (Q7):
  bash run_experiments.sh study_ngpt
  Pending. Independent of Study C.

Study E — Multi-sphere compositions (Q8):
  bash run_experiments.sh study_sphere
  Pending. Conditioned on Study D results (need ngpt baseline first).

Scaling study (cross-cutting):
  bash run_experiments.sh phase1_scaling
  Pending. Independent; run in parallel with Study C–E.

Phase 2 — outer encoder (Q3):
  bash run_experiments.sh phase2
  Requires: smoke test + verify check (automatic gates in run_experiments.sh)
  Run after Phase 1 mol result confirmed. Run order: outer_crl first.

mHC grad norm investigation (conditional — only if go-mHC runs show rising grad norm):
  If trend > 0.1 units over 100k steps, pause Studies C and E and investigate.
  Possible causes: LR sensitivity (compare 1.5e-4 and 7.5e-5), n=4 stream initialization
  variance, or a fundamental instability in the Cayley parameterization at this scale.
  Determine the cause before prescribing a fix or blocking multi-seed runs.
```

---

## 10. Interpreting the Results

### Q1 answer (mol vs. baseline_wide — same total params)
- mol BPC < baseline_wide BPC → routing + specialization mechanism adds value beyond capacity
- mol BPC ≈ baseline_wide BPC → MoL's gain is explained by extra parameters alone
- mol BPC > baseline_wide BPC → MoL underperforms a simple capacity increase (routing hurts)

### Q2 answer (mol vs. mol_single — same total LoRA params, no routing)
- mol BPC < mol_single BPC → routing and expert specialization add value beyond high-rank adapters
- mol BPC ≈ mol_single BPC → a single high-rank LoRA is sufficient; routing adds nothing
- mol BPC > mol_single BPC → routing over adapters is worse than a unified high-rank adapter

### Q3 answer (outer_crl_learned vs. outer_strided — same steps, same compression ratio)
- outer_crl_learned BPC < outer_strided BPC → content-aware routing improves over fixed stride
- outer_crl_learned BPC ≈ outer_strided BPC → routing adds no value; compression mechanism is what matters
- outer_crl_learned BPC > outer_strided BPC → learned routing is actively harmful at this scale

### Q8 answer (ngpt_mhc_a vs. ngpt_mhc_c — same params, different manifold structure)

**Theoretical motivation:** In `ngpt_mhc_a` (multi-sphere), each of the n=4 streams is
independently renormalized to S^{d-1} after every LERP step. The go-mHC H_res mixing
between streams is a doubly-stochastic convex combination of points on their respective
hyperspheres, and each stream's α learns its own update rate. In `ngpt_mhc_c`
(wrap-sublayer), the full mHC block is treated as a single sublayer: the sphere constraint
is enforced once at the block output, not stream-by-stream.

**Directional prediction (falsifiable):** `ngpt_mhc_a` should perform better if
per-stream sphere geometry provides a meaningful inductive bias — specifically, if
maintaining each stream on S^{d-1} throughout the mixing step reduces the effective
search space and improves optimization. This prediction is supported by the spectral-sphere
framing in sHC (arXiv:2603.20896), which argues that negative curvature interactions are
beneficial when hidden states are sphere-constrained. It is also consistent with the
product manifold hypothesis (arXiv:2412.07033) that factored geometry improves attention.

`ngpt_mhc_c` should perform better (or match) if the per-stream renorm is unnecessary
overhead — i.e., if the single block-level renorm provides sufficient spherical
regularization and the intra-block stream geometry is irrelevant.

**Interpretation table:**

| Outcome | Interpretation | Implication |
|---------|---------------|-------------|
| ngpt_mhc_a BPC < ngpt_mhc_c by >3σ | Per-stream sphere geometry adds value | Multi-sphere product manifold inductive bias is beneficial at this scale |
| ngpt_mhc_a ≈ ngpt_mhc_c (within 3σ) | Manifold structure within mHC block is irrelevant | Sphere constraint only needs to be applied at block boundaries; interior geometry is noise |
| ngpt_mhc_c BPC < ngpt_mhc_a by >3σ | Per-stream renorm hurts | Forcing each stream to stay on S^{d-1} after every LERP is over-constraining at this scale |

**Exploratory note:** Both variants are novel with no published direct precedent.
The closest precedents (sHC, PM-Transformer, JPmHC) are cited in `references/components/ngpt.md`
but do not compose go-mHC with nGPT directly. Q8 is exploratory — a positive finding for
either variant motivates the composition but does not establish a general principle without
replication at larger scale.

### Cross-question interpretation
If Q1 shows mol ≈ baseline_wide (capacity explains the gain) AND Q2 shows mol < mol_single
(routing helps over high-rank adapters), then the conclusion is: MoL's architecture benefits
from both routing AND the specific LoRA factorization, but a wider dense baseline achieves
the same BPC without the complexity — and the architecture advantage is in efficiency (same
BPC at lower active FLOPs), not in raw quality.

---

## 11. Resource Allocation and Prioritization

### Estimated compute (Lightning.ai T4, single GPU)

| Study | Configs | Est. T4 hours | Status |
|-------|---------|--------------|--------|
| Study A — MoLE core (Q1, Q2) | 6 × 100k steps | ~12–15h | Pending re-run (prior results superseded) |
| Study B — Attention variants (Q4, Q5) | 3 × 100k steps | ~6–8h | Pending re-run (prior results superseded) |
| Study C — go-mHC compositions (Q6) | 3 × 100k steps | ~8–10h | Pending |
| Study D — nGPT hyperspherical (Q7) | 3 × 100k steps | ~7–9h | Pending |
| Study E — Multi-sphere compositions (Q8) | 2 × 100k steps | ~5–7h | Pending |
| Phase 2 — Outer encoder (Q3) | 10 × 50k steps | ~10–15h | Pending |
| Scaling study | 10 × 100k steps | ~20–25h | Pending |
| Multi-seed reruns (A primary claims) | 6 configs × 2 more seeds | ~12–15h | Pending |

**Total estimated:** ~80–105 T4-hours for full execution.

### Priority order if compute is constrained

The following order maximizes scientific value per compute-hour:

**Priority 1 — Required for any publishable result:**
- Study A multi-seed reruns (Q1, Q2): currently single-seed; results are preliminary
- Study B seed42 is complete; no multi-seed required for primary claims
- Phase 2 Study F (Q3): requires Study A mol result confirmed — run `outer_crl` first

**Priority 2 — High value, independent:**
- Study D (Q7, nGPT): independent of C; tests a well-motivated published claim
- Study C (Q6, go-mHC compositions): requires mHC diagnostic to be resolved first

**Priority 3 — Exploratory, conditioned on prior studies:**
- Study E (Q8, multi-sphere): requires Study D nGPT baseline; novel, no prior precedent
- Scaling study: independent but long; only relevant after primary studies confirm trends

**Minimum viable result set** (if only Priority 1 completes):
Studies A (multi-seed) + B (seed42) answer Q1, Q2, Q4, Q5. Phase 2 answers Q3. This
constitutes a complete, publishable MoLE validation. Studies C–E and the scaling study
are extensions, not prerequisites.

### Compute risk flags

- **Study A multi-seed mHC/compose**: if go-mHC mhc runs show rising grad norm (trend > 0.1 units over 100k steps), pause and investigate before running 3-seed mHC or compose (see §9).
- **Study C**: same condition — go-mHC compositions depend on understanding the mHC baseline if instability is observed.
- **Scaling study**: at ~20–25h it is the single most expensive study and answers the
  narrowest question. Schedule last.

---

## Sources

This study design is grounded in:

- Melis et al., "On the State of the Art of Evaluation in Neural Language Models" (ICLR 2018)
  — hyperparameter control as the #1 confound in architecture papers
- Fedus et al., "Switch Transformers" (JMLR 2022) — iso-FLOPs as the primary fair axis
- Hoffmann et al., "Training Compute-Optimal LLMs" / Chinchilla (2022) — step count and
  cosine schedule sensitivity
- MoELoRA (Liu et al., 2024) — total-param-matched single-LoRA baseline as standard practice
- Mixture of Cluster-Conditional LoRA (2023) — dense ensemble can outperform sparse routing;
  routing must be validated, not assumed
- MoLA (NAACL 2025) — layer-wise routing value varies; lower layers are redundant
- "Parameters vs FLOPs: Scaling Laws for Optimal Sparsity" (arXiv:2501.12370, 2025)
  — 3D IsoFLOP surface for MoE architecture search
