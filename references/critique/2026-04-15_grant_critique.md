# Grant Critique Report
**Date:** 2026-04-15
**Reviewer:** Dr. Evelyn Marsh, Program Officer, ML Foundations Grant Committee
**Project:** MoLE Toy Validation — Independent Research Proposal
**Verdict:** CONDITIONAL — MAJOR REVISIONS REQUIRED

---

## Executive Summary

The core MoLE study (Q1/Q2) has sound experimental controls and its failure modes are
honestly documented — that is commendable. However, the project has accumulated eight
research questions and seventeen configs without a corresponding increase in analytical
rigor or resource planning, and a fundamental provenance problem with the mHC training
results makes the experimental record internally inconsistent. Three FATAL and five MAJOR
issues must be resolved before this project presents credible results.

---

## FATAL — Blocks grant recommendation

### F1: mHC seed42 results are from an unidentified implementation

**Issue:** The project reports mhc seed42 results (3.5428 BPC, grad norm rising 0.80→1.37)
in README.md and CLAUDE.md as current findings. go-mHC replaced KromHC/Sinkhorn in April
2026 — the same month the verification report was written. The training results predate this
migration. Nowhere in the documentation is it stated which implementation produced the
reported mhc seed42 numbers. The codebase now contains go-mHC; it is not established whether
the results table reflects go-mHC or the superseded KromHC implementation.

**Why it blocks:** If the mhc result is from KromHC/Sinkhorn, it is not reproducible with
the current codebase. Study A draws on mhc as a data point; Study C (mHC compositions) and
Study E (multi-sphere) explicitly build on the mHC baseline. Any conclusion that depends on
mhc results is evaluating a component that no longer exists in the codebase. Furthermore,
the mhc.md spec attributes the rising grad norm to "the Sinkhorn approximation gap" — a
cause that go-mHC eliminates by construction. Either the diagnosis is wrong (grad norm rises
even with exact doubly stochastic H_res) or the reported seed42 result used the old
implementation and the issue may already be solved. Neither scenario is documented.

**Required fix:** Document unambiguously which implementation produced the reported mhc
seed42 result. If KromHC/Sinkhorn: re-run mhc seed42 with go-mHC before treating any mhc
result as valid. If go-mHC: update the spec's grad norm explanation, which currently
attributes the problem to a cause that doesn't apply to the current implementation.

---

### F2: Study E (Q8) has no falsifiable hypothesis

**Issue:** ngpt_mhc_a (multi-sphere) and ngpt_mhc_c (wrap-sublayer) are described as two
design variants with the same parameter count. The research question is "Does multi-sphere
stream mixing outperform wrap-sublayer composition?" STUDY_DESIGN.md §10 provides
interpretation tables for Q1, Q2, and Q3. It provides nothing for Q8. There is no stated
prediction, no theoretical motivation for which variant should win, and no interpretive
framework for any outcome.

**Why it blocks:** A comparison with no prior hypothesis and no interpretation framework
is not a research question — it is an exploratory engineering run. Grant funding is for
science, not for "run two variants and see." Additionally, both variants are novel
compositions with no published precedent. The project cites closest precedents (sHC,
PM-Transformer, JPmHC) in the ngpt.md spec but does not use them to derive a prediction.
If sHC's spectral-sphere framing suggests that negative stream interactions are beneficial
when hidden states are on S^{d-1}, that is a testable hypothesis. State it.

**Required fix:** Add to STUDY_DESIGN.md §10 an interpretation table for Q8 with
at minimum: what outcome would support or refute each design choice, and what the
theoretical basis is for any directional prediction. If no prediction is possible, reframe
Q8 as "exploratory" rather than "hypothesis-driven" and remove it from primary claims.

---

### F3: The MLA scaling study cannot answer the question it poses

**Issue:** The scaling study goal is stated as "determine whether MLA's Q5 deficit is
ratio-driven or absolute-dimension-driven." The study holds d_c/d=25% constant at all
scales: at d=256 d_c=64, at d=512 d_c=128, at d=768 d_c=192. STUDY_DESIGN.md §4
explicitly acknowledges "cannot independently isolate the two effects." The study is
scheduled anyway.

**Why it blocks:** Running an experiment that you have already documented cannot answer
its stated question, then scheduling it, is not defensible science. Either the study is
justified by a *different* question it can actually answer (e.g., "how does MLA's
absolute deficit in BPC scale with model size?"), or it should not be scheduled until
a confound-isolating variant is added (e.g., mla at d=512 with d_c=64 to isolate
absolute dimension while keeping d constant).

**Required fix:** Either restate the scaling study's research question as one that the
design can actually answer, or add a run that isolates the ratio vs. absolute dimension
effect. A single additional mla config at d=512, d_c=64 would suffice to separate the
two effects.

---

## MAJOR — Substantive response required

### M1: mHC grad norm explanation is incoherent post-migration

**Issue:** mhc.md "Scale limitations / Rising grad norm" section states: "This is the
most likely explanation for the rising grad norm observed in Phase 1 mHC training
(gnorm: 0.77 early → 1.395 at step 100k). The doubly stochastic constraint is not
being fully enforced, allowing H_res to drift and spectral norm to slowly exceed 1.0."
This explanation is specifically about Sinkhorn approximation error. go-mHC is exactly
doubly stochastic by construction — Cayley transform guarantees it. The explanation
cannot apply to the current implementation. Either the grad norm issue has been resolved
(and should be stated as such) or the cause is something else entirely (and should be
investigated, not attributed to a fixed implementation detail).

**Impact:** Any diagnostic plan (the LR sweep at 1.5e-4 and 7.5e-5 recommended in
CLAUDE.md) that is designed to address optimization instability caused by imperfect DS
constraints is solving the wrong problem if go-mHC is in use.

**Suggested fix:** (1) Run go-mHC seed42 mhc config for 25k steps and observe grad norm
trajectory. (2) Update mhc.md "Rising grad norm pattern" section with the current
explanation — if the problem persists with exact DS, the explanation must change. If it
does not persist, remove the warning from CLAUDE.md.

---

### M2: Q3 statistical requirements are asymmetric

**Issue:** §5 states "3 seeds for outer_crl_learned (primary A1 config); 1 seed for all
others." The primary Q3 comparison is outer_crl_learned vs. outer_crl_fixed_stride (same
encoder, different routing). outer_crl_fixed_stride is the control arm of this comparison.
Running the treatment at 3 seeds and the control at 1 seed makes the margin estimate
(treatment mean minus control point estimate) asymmetric — variance is estimated for the
treatment but unknown for the control. The "margin > 3× cross-seed std" requirement
cannot be applied symmetrically.

**Impact:** Any positive result for learned routing will be undefendable: the control
result could be an outlier. An n=1 control is not a control.

**Suggested fix:** Run outer_crl_fixed_stride at 3 seeds alongside outer_crl_learned.
Update §5 to state both require 3 seeds.

---

### M3: FLOPs/active-params table covers only 6 of 17 configs

**Issue:** §3 provides active params and FFN FLOPs for baseline, baseline_wide, mol,
mol_single, mhc, and compose. The remaining 11 configs — mla, diff_attn, diff_mla, all
three mHC composition variants, ngpt, ngpt_mla, ngpt_diff_attn, ngpt_mhc_a, ngpt_mhc_c —
are absent. diff_attn notably adds ~25% more attention parameters (doubled Q heads) and
this confound is acknowledged in the README but not quantified in §3. The nGPT configs
add weight normalization overhead and a logit-scale parameter; the mHC compositions
change active streams.

**Impact:** Without complete FLOPs accounting, any cross-config BPC comparison is
potentially confounded by compute differences. This is especially acute for diff_attn
(acknowledged capacity confound) and the mHC composition variants (mHC adds n=4 stream
compute overhead on top of the attention variant).

**Suggested fix:** Complete §3 for all 17 configs. At minimum, add active param counts
for attention variants and nGPT configs. Use the shape_check.py verified param counts
as the source.

---

### M4: nGPT convergence speedup not measurable with fixed step budget

**Issue:** Q7 asks "Does hyperspherical constraint improve convergence/BPC?" The primary
published claim for nGPT is convergence speedup in training steps (4–20× per
arXiv:2410.01131), not a final BPC improvement at matched step count. All Phase 1 configs
run for a fixed 100k steps. If both nGPT and baseline converge well before step 100k, a
BPC comparison at 100k measures asymptotic performance, not convergence speed, and may
miss the speedup advantage entirely. STUDY_DESIGN.md §8 notes "Cannot conclude that
results transfer to larger scale" but does not note that the study design may miss the
primary nGPT advantage.

**Impact:** The project may run nGPT for 100k steps, observe no BPC improvement over
baseline at step 100k, and report "nGPT does not help at this scale" — when the actual
finding is "nGPT converges faster, but both models saturate before 100k steps at d=512."
These are opposite conclusions.

**Suggested fix:** Log and report BPC at steps 5k, 10k, 25k, 50k, and 100k for nGPT
and its matched baseline configs. Add "steps to reach baseline final BPC" as a reported
metric. Update §6 to require this for Study D and E configs.

---

### M5: Scope has grown from 3 to 8 research questions with no resource planning

**Issue:** Q1–Q3 were the original study design. Q4–Q8 were added in April 2026. The
document acknowledges this but provides no analysis of whether all 37 configs (17 Phase 1
+ 10 Phase 2 + 10 scaling) are executable within a realistic cloud compute budget. Phase 1
alone requires approximately 14–21 hours of T4 time; Phase 2 requires 10–15 hours; scaling
study adds another ~10 hours. The execution order in §9 presents these as sequential but
says nothing about prioritization if resources are constrained.

**Impact:** A study design that cannot be fully executed within its resource envelope is
aspirational, not scientific. The project may end with Studies A and B complete and
Studies C–F never started — at which point the "8 research questions" framing is
misleading.

**Suggested fix:** Add a §11 ("Resource allocation and prioritization") stating the
estimated compute per study, the priority order if resources are constrained, and which
studies must complete for the project to report any publishable result.

---

## MINOR — Should address

### m1: CLAUDE.md component table references superseded implementation

"mHC hyper-connections | ... | H_res/H_pre/H_post init, **KromHC (not Sinkhorn)**, softmax
not softplus" — KromHC was superseded by go-mHC. Should read: "go-mHC (Cayley transform +
block Frobenius, not Sinkhorn), softmax not softplus."
**Fix:** One-line edit to CLAUDE.md.

---

### m2: Q3 primary comparison stated inconsistently

§1 research question table: primary comparison is `outer_crl_learned` vs. `outer_strided`.
§2 text: primary comparison is `outer_crl_learned` vs. `outer_crl_fixed_stride`. §2 is
correct. §1 table is wrong and should match.
**Fix:** Update §1 table to read: `outer_crl_learned` vs. `outer_crl_fixed_stride`.

---

### m3: "Best checkpoint" vs. "final step" for primary metric is undefined

README.md reports "Best checkpoint val BPC." For multi-seed comparison and margin
calculations, using best-checkpoint BPC introduces an implicit hyperparameter search
over checkpoint selection. The margin could reflect checkpoint selection luck rather than
architecture quality. §5's "margin > 3× std" should specify whether std is computed over
best-checkpoint BPC or final-step BPC across seeds.
**Fix:** Specify in §5 which checkpoint policy is used for multi-seed comparisons.

---

## OBSERVATIONS

- The verification pipeline (Phase A/B/B') is the most rigorous aspect of this project.
  Equations traced to primary papers, deviations from reference implementations documented
  explicitly, checklist-gated. This standard should extend to the novel compositions
  (ngpt_mhc_a/c) which currently live in unverified wiring code.

- The transformer_block.py label "wiring only — no component spec" is a gap. The nGPT +
  mHC composition mathematics (per-stream LERP, multi-sphere Fréchet mean approximation,
  per-stream α) live in this file and are architecturally non-trivial. "Wiring only" is
  not an accurate description.

- The LR non-tuning policy ("if mHC needs a different LR, that is a finding") is
  scientifically sound — it prevents silent per-architecture advantages. It is applied
  consistently. This is a principled choice that should be stated explicitly in §7.

- Phase 2's 9.7 BPC failure post-mortem and the resulting smoke test gate demonstrate
  honest self-documentation and engineering follow-through. The smoke test is a genuine
  contribution to research process hygiene.

---

## Genuine Strengths

1. **Capacity-matched controls.** Q1's baseline_wide and Q2's mol_single are
   well-motivated and correctly parameterized. The 3.2% gap in baseline_wide vs. mol is
   documented, quantified, and its directional implication is correctly stated.

2. **Verification pipeline.** Equations traced to primary sources, deviations from
   reference implementations documented with justification. Better than most independent
   ML research at this scale.

3. **Self-documenting failure.** The Phase 2 9.7 BPC plateau post-mortem and the smoke
   test gate it produced show honest engineering practice.

4. **LR lock policy.** Refusing to tune LR per-architecture eliminates a major
   confound and is applied consistently across all 17 configs.

---

## Prioritized Action List

| Priority | Item | File(s) to change | Effort |
|----------|------|-------------------|--------|
| FATAL | F1: Document and resolve mHC result provenance | README.md, mhc.md, CLAUDE.md; re-run mhc seed42 with go-mHC if needed | Medium |
| FATAL | F2: Add falsifiable hypothesis and interpretation framework for Q8 | STUDY_DESIGN.md §10 | Low |
| FATAL | F3: Reframe or fix MLA scaling study to answer an answerable question | STUDY_DESIGN.md §4, §9; add mla d=512 d_c=64 config if needed | Medium |
| MAJOR | M1: Update mhc.md grad norm explanation post-go-mHC migration | references/components/mhc.md | Low |
| MAJOR | M2: Add 3-seed requirement for outer_crl_fixed_stride | STUDY_DESIGN.md §5 | Low |
| MAJOR | M3: Complete FLOPs/active-params table for all 17 configs | STUDY_DESIGN.md §3 | Low |
| MAJOR | M4: Add convergence speed reporting for Study D/E (nGPT) | STUDY_DESIGN.md §6; phase1/train.py eval logging | Low–Medium |
| MAJOR | M5: Add resource allocation and prioritization section | STUDY_DESIGN.md §11 (new) | Low |
| MINOR | m1: Fix CLAUDE.md KromHC reference | CLAUDE.md | Low |
| MINOR | m2: Fix Q3 primary comparison in §1 table | STUDY_DESIGN.md §1 | Low |
| MINOR | m3: Specify checkpoint policy for multi-seed margin calculation | STUDY_DESIGN.md §5 | Low |
