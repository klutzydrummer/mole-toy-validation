# Training Methodology Reference

This document captures current best practices (2023–2025) for training small-to-medium
transformer language models, with direct annotations about this codebase's configuration.

Sources are cited inline. All empirical claims are grounded in peer-reviewed papers or
widely replicated practitioner results. **"→ Codebase"** annotations call out decisions
specific to this project.

---

## 1. Optimizer Choice

### Recommendation: AdamW with β₁=0.9, β₂=0.95

AdamW remains the reliable universal baseline for LM pretraining. Key hyperparameters:
- β₁ = 0.9, **β₂ = 0.95** (not 0.999 — 0.95 is validated for LMs; 0.999 causes slow
  adaptation to loss landscape changes)
- Decoupled weight decay λ = 0.1
- Gradient clip max_norm = 1.0

→ **Codebase**: Current AdamW config (β₁=0.9, β₂=0.95, wd=0.1) is correct.

### Free improvement: C-AdamW (Cautious Optimizer)

A one-line modification — mask gradient updates where the update direction disagrees with
the current gradient:

```python
# In AdamW step, after computing update u and current gradient g:
mask = (u * g > 0).float()  # 1 where update agrees with gradient
param.data -= lr * mask * u  # apply only agreeing updates
```

Provides ~1.47× sample efficiency on LLaMA-scale pretraining with **no hyperparameter
retuning**. Wins 5/7 downstream tasks vs. vanilla AdamW at 130M–1.2B scale.

Source: "Cautious Optimizers: Improving Training with One Line of Code" — arXiv:2411.16085

### Future consideration: Muon

Muon applies orthogonalized gradient momentum (Newton-Schulz iteration) to 2D weight
matrices only; embeddings and lm_head still use AdamW. ~2× compute efficiency vs. AdamW
in compute-optimal training (Moonlight, arXiv:2502.16982). Hybrid setup required: Muon
for hidden layers, AdamW for embeddings and output head. Does not improve SFT when
pre-training was done with AdamW.

---

## 2. Learning Rate Schedule

### Recommendation: Cosine decay (current) or WSD

**Cosine decay** decays LR to 10% of max value. Works well for fixed compute budgets.
Current implementation (cosine with linear warmup) is correct.

**WSD (Warmup-Stable-Decay)** is the emerging flexible alternative:
- Phase 1: linear warmup (~1% of steps)
- Phase 2: constant plateau (~89% of steps)
- Phase 3: short cosine/sqrt cooldown (~10% of steps) — sharpest convergence gains happen here

WSD advantage: checkpoint anytime in the stable phase, add a cooldown on demand. Enables
continual learning without schedule restarts.

**Cooldown shape**: sqrt or "lowered linear" decay outperforms pure linear or cosine for
the cooldown phase. Raise β₂ to 0.999 during cooldown only (from its 0.95 training value)
for additional gains.

→ **Codebase**: Current cosine schedule is fine. If training needs to be interruptible or
resumed, WSD is a drop-in replacement. For the mHC diagnostic LR sweep (1.5e-4, 7.5e-5),
keep the same cosine schedule to isolate LR as the only variable.

Source: "Understanding Warmup-Stable-Decay LRs" — arXiv:2410.05192;
"Training Dynamics of the Cooldown Stage in WSD" — arXiv:2508.01483

---

## 3. Warmup

### Recommendation: Linear warmup over ~1% of total steps

Warmup allows Adam's β₂ variance estimates to stabilize before the peak LR is reached.

- **Duration**: 1–2% of total steps. 1% is robust across scales.
  - Phase 1 (100k steps): ~1000 steps warmup
  - Phase 2 (50k steps): ~500 steps warmup
- **Shape**: Linear warmup is sufficient at small scale (5M–100M params).
- **Token vs. step basis**: Step-based is fine when batch size is fixed.

Source: "Why Warmup the Learning Rate?" — arXiv:2406.09405

→ **Codebase**: Confirm the `warmup_steps` argument in `train.py` is near 1% of
`total_steps`. This is a locked hyperparameter per STUDY_DESIGN.md.

---

## 4. Gradient Clipping

### Recommendation: Global L2 norm clipping, max_norm=1.0

- **Method**: `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)` — preserves gradient
  direction while limiting magnitude. Never use per-value clipping.
- **Threshold**: 1.0 is the universal LM pretraining standard (GPT-2, LLaMA, Pythia).
- **Key diagnostic**: Log the **raw (pre-clip) gradient norm every step**. If it frequently
  hits the clip threshold, that is an early warning of instability, not a healthy sign.
- **Rising trend**: If the rolling mean of the grad norm trends upward over 1000+ steps
  (not just a spike), investigate per-layer output norms. A 2× increase vs. early training
  in any QKV or projection layer predicts divergence.

→ **Codebase**: Current clip at 1.0 is correct. The mHC rising grad norm (0.54→0.70,
1.29× over training) is exactly the "sustained upward trend" signal — diagnostic LR runs
at 1.5e-4 and 7.5e-5 are the right next step. Consider adding per-layer norm logging
for the first 5k steps of new mHC runs.

Source: "Methods for Improving LLM Training Stability" — arXiv:2410.16682

---

## 5. Weight Decay

### Recommendation: Fixed λ=0.1, decoupled, exclude biases/LN/embeddings

- Keep λ=0.1 constant throughout training (do not schedule it to follow the LR).
- Apply weight decay only to weight matrices (Linear layers). Exclude:
  - Biases
  - LayerNorm / RMSNorm weights and biases
  - Embedding weights
  - Output projection tied to embeddings (weight tying)
- For small datasets (WikiText-103 ~100M tokens), λ=0.05 could be tried if overfitting
  is observed, but 0.1 is the safer default.

The "timescale" framing τ = 1/(LR × λ) provides a useful invariant: at LR=3e-4, λ=0.1,
τ ≈ 33,000 — meaning parameters decay to 37% of their magnitude every ~33k steps at peak LR.

→ **Codebase**: Verify that `train.py` uses a parameter group that excludes biases and
LN parameters from weight decay. This is a common implementation oversight.

Source: "How to set AdamW's weight decay as you scale" — arXiv:2405.13698

---

## 6. Mixed Precision

### Strong recommendation: bfloat16 over float16

**bfloat16** is strongly preferred for transformer pretraining on Ampere+ GPUs:
- 8-bit exponent matches FP32 dynamic range — overflow/underflow is extremely rare
- No loss scaling required (unlike float16 which needs `GradScaler`)
- Training "just works" with `torch.amp.autocast('cuda', dtype=torch.bfloat16)`

**float16 pitfalls** for this codebase specifically:
- `sigmoid(7.5)` rounds to exactly 1.0 in float16 (float16 max representable < 1 is
  0.9990), making `1 - a_t² = 0` and nullifying the `CausalRecurrenceLayer` sqrt
  normalization. Already guarded by computing sigmoid in float32 in the current code,
  but this guard would not be needed with bfloat16.
- Gradient overflow (activations >65504) triggers loss spikes hard to distinguish from
  architectural instability.

→ **Codebase**: Currently using AMP float16. **Consider migrating to bfloat16**. Change:
`torch.cuda.amp.autocast()` → `torch.amp.autocast('cuda', dtype=torch.bfloat16)`.
Remove the `GradScaler`. Verify GPU supports bfloat16 (all Ampere+ GPUs do).

Source: "Defeating the Training-Inference Mismatch via FP16" — arXiv:2510.26788

---

## 7. Initialization

### Recommendation: Scaled residual init + standard elsewhere

**Standard init** (GPT-style, currently in use):
- Linear weights: Normal(0, 0.02)
- Embeddings: Normal(0, 0.02)
- Biases: zero
- LN/RMSNorm: weight=1, bias=0

**Scaled init for residual branch projections**:
Apply `std = 0.02 / sqrt(2 × n_layers)` to the output projections of attention and FFN
blocks (the matrices that write back to the residual stream). This prevents residual
accumulation from growing with depth.

At 8 layers: std = 0.02 / sqrt(16) ≈ 0.005

This is the "GPT-2 residual scaling" and is one of the highest-impact initialization
changes for deep transformers. Formally motivated by DS-Init (arXiv:1908.11365).

→ **Codebase**: Check `_init_weights` in `phase1/model.py` and `phase2/model.py`. The
attention output projection and FFN down projection should use the scaled std. If they
currently use a flat 0.02, this is a potential stability improvement for mHC.

Source: "Improving Deep Transformer with Depth-Scaled Initialization" — arXiv:1908.11365

---

## 8. Dropout

### Recommendation: Zero dropout throughout pretraining

Removing dropout improves downstream performance at all tested scales (160M–1.4B params).
Even "early dropout" (applied only for the first N steps) degrades performance relative
to no dropout at all. Use weight decay and gradient clipping as the primary regularization.

→ **Codebase**: Confirm dropout = 0 in all configs. Do not add dropout as a stability
aid for mHC — the rising grad norm is not a regularization issue.

Source: "Drop Dropout on Single-Epoch Language Model Pretraining" — arXiv:2505.24788
(ACL Findings 2025)

---

## 9. Data Repetition

### WikiText-103 repeat count for this codebase

- WikiText-103-raw: ~100M BPE tokens (vocab=4096)
- Phase 1: 100k steps × 32 batch × 256 seq_len = **819M tokens ≈ 8.2 passes**
- Phase 2: 50k steps × 32 batch × 256 seq_len = **410M tokens ≈ 4.1 passes**

**The threshold for significant degradation is ~16 passes** (Muennighoff et al., NeurIPS
2023). Up to 4 passes causes negligible change to validation loss; 4–16 passes causes
mild degradation. Both phases are within the safe zone.

Signs that data repetition is causing overfitting:
- Training loss decreases while validation BPC plateaus or rises (train–val gap widening)
- Expert specialization collapses (MoL experts become increasingly similar)
- Gradient norm rising trend in later training

Monitor: track train–val gap. Flag overfitting if the gap widens by >10% relative to
its value at the training midpoint.

Source: "Scaling Data-Constrained Language Models" — arXiv:2305.16264

---

## 10. Checkpointing and Loss Spike Recovery

### Checkpoint frequency: every 500 steps

At 5M–31M params, one checkpoint is ~20–100MB. Maintain a rotating buffer of the last
3 checkpoints to enable rollback.

**Spike classification**:
- Normal spike: loss rises to ≤2× baseline, recovers within 500 steps — no action needed
- Divergence: loss rises and does not recover within 1000 steps — roll back and reduce LR

**Recovery procedure when divergence occurs**:
1. Roll back to the pre-spike checkpoint
2. Reduce LR by 30–50%
3. Resume; the same cosine schedule can continue from the rolled-back step count

**Prevention**:
- bfloat16 eliminates most FP16-originated numerical spikes
- Scaled residual init (Section 7) reduces spike frequency
- Raw grad norm logged every step — a grad norm spike consistently precedes a loss spike
  by 50–200 steps

Source: "Spike No More" — arXiv:2312.16903

---

## 11. Evaluation Frequency and Overfitting Detection

### Recommendation: Eval every 500 steps; track train–val gap

- **Interval**: 500 steps for Phase 1 (200 eval points); 250 steps for Phase 2
- **Primary signal**: val BPC (already primary metric in this codebase)
- **Overfitting signal**: train–val gap widening by >10% relative to its midpoint value
- Do not use ReduceLROnPlateau for pretraining
- Always run the full cosine schedule to completion; early stopping cuts off the
  low-LR "polishing" phase that accounts for a disproportionate share of final BPC gains

Source: "OverfitGuard" — arXiv:2401.10359

---

## 12. Statistical Requirements

### Minimum 3 seeds for any architectural claim

Seed variance at 5M params can produce BPC differences as large as the architectural
differences being tested. A single-seed result for mol (1.3357) vs. baseline is not
statistically reliable — any margin < 3× cross-seed std is not claimable.

Per STUDY_DESIGN.md:
- `baseline`, `baseline_wide`, `mol`, `mol_single`: 3 seeds each
- `mhc`, `compose`: 1 seed until optimization issues resolved
- `hdc_gate` (primary Phase 2 claim): 3 seeds; all other Phase 2 configs: 1 seed

Report mean ± std. A result is claimable only if:
- The margin > 3× the cross-seed standard deviation, OR
- The worst seed of the new config beats the best seed of the baseline

Source: "Assessing Macro and Micro Effects of Random Seeds" — arXiv:2503.07329;
Melis et al. "On the State of the Art of Evaluation in NLMs" (ICLR 2018)

---

## 13. Compute Budget Assessment

### Phase 1: Appropriate

- 5M–31M param models × 820M tokens ≈ 26–164 tokens/param
- Chinchilla compute-optimal: ~20 tokens/param
- 2024 inference-optimal norm (Llama-3 style): ~192 tokens/param
- **Assessment**: Well-calibrated. Near the 2024 inference-optimal target for larger
  configs (~31M params). Over-training at smaller params is appropriate for architectural
  research — it flattens the loss curve and reduces comparison noise.

### Phase 2: Potentially under-trained

- HDC models (ZoneE + InnerTransformer + ZoneD): estimated 15–30M total params
- Phase 2: 410M tokens
- 410M / 15M params ≈ **27 tokens/param** — well below the 2024 inference-optimal target
- **Implication**: If hdc_gate fails to beat mol, it may be a step-count issue rather
  than an architectural one. Report the tokens/param ratio for each Phase 2 config.
- **Mitigation**: Compute HDC model param count explicitly and consider extending Phase 2
  to 100k steps if the model is >15M params.

Source: Chinchilla — arXiv:2203.15556; arXiv:2405.18392

---

## 14. Training Stability Checklist

Before launching any new config, verify:

| Item | Target | Where to check |
|------|--------|----------------|
| Optimizer β₂ | 0.95 | `train.py` AdamW kwargs |
| Weight decay excludes biases/LN/embed | Yes | Parameter group split in `train.py` |
| Gradient clip max_norm | 1.0 | `clip_grad_norm_` call |
| Dropout | 0 | Model config |
| LR warmup | ~1% of total_steps | `warmup_steps` param |
| Raw grad norm logged | Every step (or ≤100 steps) | `train.py` logging |
| Checkpoint interval | ≤500 steps | Save interval in `train.py` |
| Residual projection init std | 0.02 / sqrt(2 × n_layers) | `_init_weights` |
| AMP precision | bfloat16 preferred; float16 requires GradScaler | `autocast` call |
| Sigmoid in float32 under AMP | Yes (CausalRecurrenceLayer) | `phase2/model.py` |

---

## Sources

**Optimizers:**
- "Muon is Scalable for LLM Training" — arXiv:2502.16982
- "Cautious Optimizers: Improving Training with One Line of Code" — arXiv:2411.16085
- "Pre-Training LLMs on a budget: comparison of optimizers" — arXiv:2507.08472

**LR Scheduling:**
- "Understanding Warmup-Stable-Decay LRs" — arXiv:2410.05192
- "Training Dynamics of the Cooldown Stage in WSD" — arXiv:2508.01483
- "Why Warmup the Learning Rate?" — arXiv:2406.09405
- "Scaling Laws and Compute-Optimal Training Beyond Fixed Durations" — arXiv:2405.18392

**Weight Decay and Initialization:**
- "How to set AdamW's weight decay as you scale" — arXiv:2405.13698
- "Improving Deep Transformer with Depth-Scaled Initialization" — arXiv:1908.11365
- "u-μP: The Unit-Scaled Maximal Update Parametrization" — arXiv:2407.17465

**Stability and Regularization:**
- "Spike No More" — arXiv:2312.16903
- "Methods for Improving LLM Training Stability" — arXiv:2410.16682
- "Small-Scale Proxies for Large-Scale Training Instabilities" (ICLR 2024)
- "Drop Dropout on Single-Epoch Language Model Pretraining" — arXiv:2505.24788

**Data and Compute:**
- "Scaling Data-Constrained Language Models" — arXiv:2305.16264
- "Training Compute-Optimal Large Language Models" (Chinchilla) — arXiv:2203.15556
- "Curriculum Learning for LLM Pretraining" — arXiv:2601.21698

**Reproducibility:**
- "Assessing Macro and Micro Effects of Random Seeds" — arXiv:2503.07329
- Melis et al. "On the State of the Art of Evaluation in NLMs" (ICLR 2018)
