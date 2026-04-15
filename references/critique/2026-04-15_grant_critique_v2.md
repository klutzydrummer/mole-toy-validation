# Grant Critique Report
**Date:** 2026-04-15
**Reviewer:** Dr. Evelyn Marsh, Program Officer, ML Foundations Grant Committee
**Project:** MoLE Toy Validation — Independent Research Proposal
**Verdict:** CONDITIONAL — MINOR REVISIONS REQUIRED

---

## Executive Summary

The prior FATAL issues have been resolved: mHC result provenance is documented and
superseded, the MLA scaling study question is correctly reframed, and Q8 now has a
falsifiable hypothesis. The study design is materially improved. What remains are three
MAJOR issues — one structural (Q4 cannot answer its stated question by design) and two
documentary (stale Sinkhorn-era diagnostics and a metric definition conflict) — plus six
MINOR housekeeping items. None of these are fatal to grant consideration, but the Q4
framing is scientifically indefensible in its current form and must be addressed before
any diff_attn result is reported.

---

## FATAL — Blocks grant recommendation

*None.*

---

## MAJOR — Substantive response required

### M1: Q4 is a scheduled experiment that cannot answer its stated question

**Issue:** Q4 asks "Does differential attention improve over baseline?" The designated
comparison is `diff_attn` vs. `baseline`. §3 documents that `diff_attn` has +2.1M more
parameters than `baseline` (doubled Q heads: 8 layers × d² extra Q projection params =
2,097,152). §1 explicitly states "BPC improvement reflects architecture + capacity
combined." The project then proceeds to list Q4 as a research question and schedule the
run. A comparison that the project itself has documented as uninterpretable cannot be a
research question.

**Impact:** Any positive result for `diff_attn` is attributable to architecture, capacity,
or both — the project has declared it cannot distinguish these. Any negative result is
equally uninterpretable: maybe the mechanism works but needs matched capacity to show it.
Q4 will produce a data point, not an answer. If this is reported as evidence for or against
differential attention as a mechanism, it is not scientifically defensible.

**Suggested fix:** Choose one of two paths:
1. Add `diff_attn_matched` — a differential attention config with the Q projection
   reduced to maintain the same total params as `baseline` (~27.8M). This is the controlled
   comparison Q4 requires.
2. Reframe Q4 explicitly as "exploratory observation: combined effect of differential
   attention mechanism + +25% attention capacity" and remove it from the list of hypothesis-
   driven research questions. Add an interpretation note: any diff_attn result reflects
   architecture + capacity and cannot attribute improvement to mechanism alone.

Path 2 requires only documentation changes. Path 1 requires a new config and additional
training. The project should choose explicitly; the current state — scheduling an
uncontrolled comparison as a research question — is not acceptable.

---

### M2: §9 retains a Sinkhorn-era mHC diagnostic with no current theoretical grounding

**Issue:** §9 execution order includes a pre-scheduled mHC diagnostic:
```
mHC diagnostic (prerequisite for 3-seed A, C, E):
  python phase1/train.py --config mhc --max_lr 1.5e-4 --total_steps 25000
  python phase1/train.py --config mhc --max_lr 7.5e-5 --total_steps 25000
```
These specific LR values (1.5e-4 and 7.5e-5, i.e., 0.5× and 0.25× the base LR) were
motivated by the hypothesis that Sinkhorn approximation error was causing the rising grad
norm. That hypothesis has been abandoned. The project now uses go-mHC, which is exactly
doubly stochastic by construction. If go-mHC shows a rising grad norm, the cause is
unknown. Prescribing two specific LR sweeps for an unknown cause is not a diagnostic — it
is an undirected search dressed up as a diagnostic.

§8 also retains: "That mHC is definitively broken — the diagnostic runs at 1.5e-4 and
7.5e-5 LR are required first" in the "Cannot conclude" section. This language is premised
on the same abandoned theory.

**Impact:** If go-mHC shows rising grad norm, these LR sweeps will run with no clear
interpretation criterion. If the grad norm improves at 1.5e-4, that tells you something
about LR sensitivity but nothing about the cause. If it does not improve, you have consumed
compute without diagnosis. Studies C and E are currently gated on this ill-specified
procedure.

**Suggested fix:** Replace the scheduled LR diagnostic with conditional language:
"If go-mHC mhc runs show rising grad norm (trend > 0.1 units over 100k steps), pause
Studies C and E and investigate. Possible causes include: LR sensitivity (compare 1.5e-4
and 7.5e-5), n=4 stream initialization variance, or a fundamental instability in the
Cayley parameterization at this scale. Determine the cause before prescribing a fix."
Remove the unconditional prerequisite. Update §8 accordingly.

---

### M3: §5 and §6 define conflicting primary metrics

**Issue:** §6 states "`best_val_bpc` — primary metric." §5 states "Use **final-step BPC**
(not best-checkpoint BPC) when computing cross-seed standard deviation and the margin >
3× std criterion." These are different metrics. A result could be borderline on one and
clear on the other. As written, the document has two "primary metrics" with different
values for the same run. Any multi-seed comparison will require specifying which metric
to apply the 3σ criterion to, and the answer is ambiguous.

**Impact:** When Study A multi-seed runs complete, the analyst must decide: do I compute
std over best-BPC or final-step BPC? The document says both are the primary metric. This
ambiguity will either produce an undocumented ad hoc choice or require returning to this
document to resolve a conflict that should have been resolved in advance.

**Suggested fix:** §6 "Per run (final values)" should read:
- `final_step_bpc` — **primary metric for statistical comparisons** (used in §5 margin
  calculations)
- `best_val_bpc` — secondary metric; reported for reference and to track
  convergence trajectory

The two-sentence clarification in §5 about checkpoint policy should cross-reference §6.

---

## MINOR — Should address

### m1: §1 says "seven research questions" but the project has eight

The introductory paragraph reads "The project is organized into seven research questions
across five study groups." Q1 through Q8 = 8 questions. Q3 was added later (Phase 2) and
Q8 was added with Study E. The count is simply wrong.
**Fix:** Change "seven" to "eight" in §1.

---

### m2: mhc.md "n=2" sections are entirely stale — project now uses n=4

The "Scale limitations" section contains a subsection titled **"n=2 is below the validated
minimum"** which states: "n=2 (used in this project for parameter-budget reasons)". The
project does not use n=2. All mHC configs use `n_streams=4`. The follow-up work section
also has a paragraph "For n=2 (this project's n_streams=2): mHC-lite is trivially exact..."
referring to n_streams=2 as the current configuration. The verification checklist item 5
also references "n_streams=2". These are three stale references to a parameter value the
project abandoned.

**Fix:** Update the subsection title and body to note that n=2 was a prior consideration
and the current implementation uses n=4. Remove or retitle the n=2 concern as historical
context rather than a current limitation.

---

### m3: mhc.md "Scale limitations" cites superseded results as evidence

The "No published results below 1B parameters" section states: "The Phase 1 result
(mHC: 3.5736 vs baseline: 3.5875) — mHC losing to baseline — is consistent with HC gains
being primarily a depth/scale phenomenon." These results have been explicitly superseded
and declared unreproducible from the current codebase. Citing them as evidence for a
scaling hypothesis is circular: the results are from an unconfirmed implementation, and
they are now being used to frame expectations for the go-mHC re-run. Remove the specific
BPC values from this section or clearly mark them as superseded observations pending
re-confirmation.
**Fix:** Replace the specific BPC values with: "(pending confirmation from go-mHC runs)."

---

### m4: phase1/components/CLAUDE.md component table still references KromHCResidual

The Phase B' verification report (phase_b_prime_mhc.md, Issue 2) explicitly flagged this:
"`phase1/components/CLAUDE.md` component table row for `mhc.py` reads `KromHCResidual,
HyperConnection` — should be updated to `GoMHCResidual, HyperConnection`." The verification
report was written on 2026-04-07. This is 2026-04-15. It has not been fixed.
**Fix:** One-line edit to `phase1/components/CLAUDE.md`.

---

### m5: §11 resource table shows "Seed42 complete" for Studies A and B

The §11 resource allocation table reads:
- "Study A — MoLE core (Q1, Q2) | ... | Seed42 complete"
- "Study B — Attention variants (Q4, Q5) | ... | Seed42 complete"

The rest of the document (§4, §9, README.md, CLAUDE.md) declares all prior results
superseded and all runs as "pending re-run." The status column in §11 is inconsistent with
every other statement in the project documentation.
**Fix:** Update both rows to "pending re-run (prior results superseded)."

---

### m6: run_experiments.sh phase1_scaling comment retains old MLA framing

`run_experiments.sh` line 383: comment reads "Goal: determine whether MLA's KV compression
failure is a ratio or absolute dimension problem." This is the framing that STUDY_DESIGN.md
§4 explicitly corrected to: "How does each architecture's BPC deficit scale with model
size?" The script comment contradicts the updated study design.
**Fix:** Update the comment in run_experiments.sh to match §4.

---

## OBSERVATIONS

- The verification pipeline covers the novel ngpt_mhc_a/c compositions through the
  Phase B' nGPT report, which verifies per-stream alpha shape and norm1/norm2 application
  in the branch function. However, the full Fréchet mean approximation logic in
  `_forward_ngpt_mhc_a` (the `L2Norm(mixed_i + α_i·post_i·(h − mixed_i))` update) is
  documented in the spec but not line-by-line verified against the transformer_block.py
  implementation in the report. The report only verifies alpha shape and branch norm;
  the multi-sphere LERP math itself has no verified line reference. This is the highest-
  risk unverified code path in the project.

- The NVIDIA reference implementation uses `dim=0` (column normalization) for weight
  normalization while the paper and our implementation use `dim=-1` (row normalization).
  The project follows the paper. This is documented and defensible. However: the published
  nGPT speedup results (4–20× convergence) were produced by NVIDIA with their reference
  implementation, which uses the opposite normalization convention. Whether the convergence
  speedup claim holds under row normalization is not established. This does not invalidate
  Q7 — it makes the expected effect size less predictable.

- The LR lock policy (no per-architecture LR tuning; mHC instability at 3e-4 is a finding)
  is explicitly stated in §7 and correctly applied. This remains a principled choice and
  eliminates a major confound.

- Phase 2 gates (smoke test + hash verification before any Phase 2 run) are well-engineered.
  The 9.7 BPC plateau post-mortem is honestly documented and the gate it produced is a
  genuine contribution to research process discipline.

---

## Genuine Strengths

1. **Provenance issue resolved cleanly.** The decision to supersede all prior results and
   re-run from scratch eliminates the implementation ambiguity rather than papering over it.
   This is the scientifically correct choice.

2. **Parameter table now complete.** All 17 configs have documented total/active params and
   FFN FLOPs, with explicit capacity confound warnings. This is the level of accounting that
   architecture comparisons require.

3. **Q8 interpretation framework.** The Q8 table in §10 is properly structured: directional
   prediction, theoretical motivation from published precedents, three-outcome interpretation
   table, and an honest "exploratory" label. This is how novel compositions should be framed.

4. **Resource prioritization.** §11 correctly identifies the minimum viable result set and
   decouples it from the exploratory extensions. The compute risk flags are accurate and
   appropriately conservative.

---

## Prioritized Action List

| Priority | Item | File(s) to change | Effort |
|----------|------|-------------------|--------|
| MAJOR | M1: Resolve Q4 — add controlled diff_attn variant OR reframe as exploratory observation | STUDY_DESIGN.md §1, §3, §4; phase1/model.py + run_experiments.sh if new config added | Low (reframe) or Medium (new config) |
| MAJOR | M2: Replace Sinkhorn-era mHC LR diagnostic with conditional investigation protocol | STUDY_DESIGN.md §8, §9 | Low |
| MAJOR | M3: Reconcile primary metric — final-step BPC for stats (§5) vs. best_val_bpc for reporting (§6) | STUDY_DESIGN.md §5, §6 | Low |
| MINOR | m1: Fix "seven" → "eight" research questions in §1 | STUDY_DESIGN.md §1 | Low |
| MINOR | m2: Update mhc.md n=2 sections — project uses n=4, n=2 is historical | references/components/mhc.md | Low |
| MINOR | m3: Remove superseded BPC values from mhc.md scale limitations | references/components/mhc.md | Low |
| MINOR | m4: Fix phase1/components/CLAUDE.md: KromHCResidual → GoMHCResidual | phase1/components/CLAUDE.md | Low |
| MINOR | m5: Update §11 status column — "Seed42 complete" → "pending re-run" | STUDY_DESIGN.md §11 | Low |
| MINOR | m6: Update run_experiments.sh phase1_scaling comment to match §4 framing | run_experiments.sh | Low |

*Effort: Low = documentation only, Medium = new ablation or config, High = multi-seed
training campaign or architectural change.*
