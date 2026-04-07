# MoLE Toy Validation — Study Design

This document defines the experimental protocol for the MoLE toy validation. It governs
what we train, how we compare results, and what conclusions we can draw. Every config
addition or change to training procedure should be evaluated against this document first.

---

## 1. The Three Questions

This study asks three separable questions. Each requires different controls.

| Question | Axis | Primary comparison |
|----------|------|--------------------|
| **Q1 — Does MoL's routing mechanism help, beyond just having more parameters?** | Total trainable params | `mol` vs. `baseline_wide` (matched total params) |
| **Q2 — Is routing over adapters better than a single high-rank adapter?** | Total LoRA params | `mol` vs. `mol_single` (same total LoRA params, no routing) |
| **Q3 — Does content-aware compression improve over fixed-stride compression?** | Active FLOPs + training steps | `outer_crl_learned` vs. `outer_strided` (same steps, same target compression ratio) |

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

**`baseline_wide`**: SwiGLU with d_ff scaled up to match MoL's total params.

To compute the target d_ff: MoL adds 3 LoRA sets × 9 experts × (d_in×r + r×d_out) × 8 layers
over the base SwiGLU. Solving for a SwiGLU that adds the same number of params by increasing
d_ff gives approximately d_ff ≈ 1600 (vs. current 1408). Verify empirically that
`baseline_wide` hits ~31.1M before training.

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

For transparency, the following should be computed and reported for every config:

| Config | Total params | Active params/token | Approx. FFN FLOPs/token |
|--------|-------------|--------------------|-----------------------|
| `baseline` | 27.8M | 27.8M | 2 × d × d_ff = 1.44M |
| `baseline_wide` | ~31.1M | ~31.1M | ~1.60M |
| `mol` | 31.1M | ~28.9M* | ~1.50M* |
| `mol_single` | ~31.1M | ~31.1M | ~1.48M* |
| `mhc` | 27.8M | 27.8M | same as baseline |
| `compose` | 31.1M | ~28.9M | ~1.50M |

*MoL active params = base FFN + shared LoRA + 2 selected expert LoRAs + router.
The shared base FFN dominates; LoRA overhead is ~6% of total FFN FLOPs.

**Key insight**: MoL's active-param/FLOPs cost is close to baseline (~6% overhead), not 12%
more. The 3.35M extra total params are almost entirely in the 6 non-selected expert LoRAs
that are trained but inactive at inference. This means Q1 (capacity control) matters more
than Q3 (FLOPs control) for interpreting MoL's result.

---

## 4. Required Configs

### Phase 1 — full set

| Config | Purpose | Status |
|--------|---------|--------|
| `baseline` | Dense SwiGLU reference (27.8M) | Run complete — **rerun at 3 seeds** |
| `baseline_wide` | Capacity-matched dense baseline (31.1M) | **New — must add** |
| `mol` | MoL routing (31.1M, top-2 of 8) | Run complete — **rerun at 3 seeds** |
| `mol_single` | Single rank-64 LoRA, no routing (31.1M) | **New — must add** |
| `mhc` | Hyper-connections on baseline (27.8M) | Run complete — diagnose grad norm first |
| `compose` | mHC + MoL (31.1M) | Run complete — interpret after mHC diagnosed |

### Phase 1 — configs that can be dropped for now

`mhc` and `compose` have known issues (rising grad norm, optimization instability). Until
the mHC diagnostic runs (`--max_lr 1.5e-4` and `7.5e-5`) are complete, do not re-run these
at 3 seeds. They are not required for the Q1/Q2 questions.

### Phase 2 — outer encoder study

The Phase 2 config set is structured around Q3. 10 configs (OuterModel.CONFIGS).
The Q3 controlled comparison is `outer_crl_learned` vs. `outer_crl_fixed_stride` (same
encoder, different routing). `outer_strided` is a secondary no-encoder lower bound.

**Note:** The MLA scaling study (§7, if added) cannot independently isolate ratio vs.
absolute dimension effects because d_c/d=25% is held constant at all scales. At d=256,
d_c=64; at d=512, d_c=128; at d=768, d_c=192. Both the ratio and the absolute bottleneck
size vary proportionally with d. To isolate these effects requires running `mla` at d=512
with a non-default d_c (e.g., d_c=64 to hold absolute dimension constant). This run is
not currently scheduled but is needed before any scaling claim about MLA can be made.

---

## 5. Statistical Requirements

Based on literature consensus (Melis et al. 2018, NLP ablation practice):

**Minimum**: 3 seeds for any config used in a primary claim.
**Primary claims** (Q1, Q2): `baseline`, `baseline_wide`, `mol`, `mol_single` — all need 3 seeds.
**Supporting configs** (`mhc`, `compose`): 1 seed until optimization issues are resolved.
**Phase 2**: 3 seeds for `outer_crl_learned` (primary A1 config); 1 seed for all others.

**Reporting**: Always report mean ± std across seeds. A result is only claimable if:
- The margin is larger than 3× the cross-seed standard deviation, OR
- The worst seed of the new config beats the best seed of the baseline.

**Do not** report only best-seed results.

---

## 6. Metrics to Report for Every Config

### Per run (final values)
- `best_val_bpc` — primary metric
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
- That mHC is definitively broken — the diagnostic runs at 1.5e-4 and 7.5e-5 LR are required first
- Cross-phase BPC comparisons (Phase 1 at 100k steps ≠ Phase 2 at 50k steps)

---

## 9. Execution Order

```
Phase 1 (revised):
  1. baseline        × 3 seeds   (rerun — establishes variance baseline)
  2. baseline_wide   × 3 seeds   (new — Q1 capacity control)
  3. mol             × 3 seeds   (rerun — Q1 and Q2 treatment)
  4. mol_single      × 3 seeds   (new — Q2 routing-vs-rank control)
  5. mhc diagnostic  × 1 seed each (LR sweep: 1.5e-4, 7.5e-5)
  6. mhc             × 3 seeds   (after diagnostic, if stable)
  7. compose         × 1 seed    (only after mhc is resolved)

Phase 2 (run after Phase 1 mol result is confirmed):
  smoke_test → outer_crl → outer_crl_learned × 3 seeds → outer_crl_full → outer_crl_full_learned
  → outer_transformer → outer_diff_attn → outer_mla → outer_strided → outer_crl_learned_noste
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

### Cross-question interpretation
If Q1 shows mol ≈ baseline_wide (capacity explains the gain) AND Q2 shows mol < mol_single
(routing helps over high-rank adapters), then the conclusion is: MoL's architecture benefits
from both routing AND the specific LoRA factorization, but a wider dense baseline achieves
the same BPC without the complexity — and the architecture advantage is in efficiency (same
BPC at lower active FLOPs), not in raw quality.

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
