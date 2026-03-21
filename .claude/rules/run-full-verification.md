---
description: "Triggered by: 'run full verification'. Defines the Phase B' agent verification process."
---

# Full Verification Protocol

When the user says "run full verification" (or any clear variant), execute the Phase B' verification pipeline below. Do NOT skip steps. Do NOT record a pass without completing all checks.

---

## What verification means

The verification library has three phases:

| Phase | Location | Contents |
|-------|----------|----------|
| A | `references/sources/papers/` + `references/sources/code/` | Primary sources: paper summaries and verbatim reference implementations |
| B | `references/components/` | Component specs: authoritative equations, reference code snippets, and documented deviations for our implementation |
| B' | `references/verification/reports/` | Verification reports: Phase B specs cross-checked against Phase A sources AND our implementation |

**Phase A and B already exist.** Full verification means running Phase B': dispatching agents to validate that every claim in every Phase B component file is traceable to a Phase A source, and that our implementation in `phase1/model.py` / `phase2/model.py` matches the spec (or has a documented, intentional deviation).

---

## Phase B' procedure — run for each component

For each component spec in `references/components/`, dispatch a subagent that:

1. **Reads** the component spec (`references/components/<component>.md`)
2. **Reads** every source file cited in the spec's "Sources" table (`references/sources/papers/<paper>.md` and `references/sources/code/<file>.py`)
3. **Reads** the implementation section of `phase1/model.py` and/or `phase2/model.py` indicated by the spec's "Our implementation" pointers
4. **Cross-checks**:
   - Every equation in "Authoritative equations" — is it verbatim from the cited paper?
   - Every code snippet in "Reference implementation" — is it verbatim from the cited source file?
   - Every claim in "Our implementation" — does the actual code at the cited line numbers match?
   - Every item in "Intentional deviations" — is the deviation accurately described, and is it actually present in the code?
   - Every item in the "Verification checklist" — does the code satisfy the check?
5. **Writes a Phase B' report** to `references/verification/reports/phase_b_prime_<component>.md` with:
   - Overall verdict: PASS / FAIL / PASS with issues
   - Per-claim status: VERIFIED / INCORRECT / NOT FOUND
   - For any FAIL: exact location of the discrepancy (spec line vs code line vs source line)

### Components to verify

| Component spec | Sources | Implementation |
|---------------|---------|----------------|
| `causal_recurrence.md` | `griffin_2402.19427.md`, `griffin_rglru.py` | `phase2/model.py` — CausalRecurrenceLayer, _parallel_scan |
| `zone_ed_pipeline.md` | `hnet_2507.07955.md`, `hnet_boundary.py`, `dlcm.md` | `phase2/model.py` — ZoneE, ZoneD, BoundaryRouter |
| `boundary_router.md` | `hnet_2507.07955.md`, `hnet_boundary.py`, `dlcm.md` | `phase2/model.py` — BoundaryRouter |
| `mol_ffn.md` | `deepseek_v3_2412.19437.md`, `deepseek_v3_moe.py` | `phase1/model.py` — MoLFFN; `phase2/model.py` — InnerTransformer |
| `mhc.md` | `mhc_2512.24880.md`, `mhc_hyper_connections.py` | `phase1/model.py` — mHCLayer |
| `attention_rope_norms.md` | `rope_2104.09864.md`, `rmsnorm_1910.07467.md`, `swiglu_2002.05202.md`, `rope.py`, `rmsnorm.py`, `swiglu.py` | `phase1/model.py`, `phase2/model.py` — attention blocks |

---

## After agents complete

1. Review the reports in `references/verification/reports/`
2. If all pass (or pass-with-issues where issues are acceptable): `python utils/verify.py update --result pass --report references/verification/reports/`
3. If any fail: fix the root cause in the spec or implementation, then re-run
4. `git commit` including the updated `references/verification/last_verified.json` and reports

**DO NOT run `verify.py update --result pass` if any component has a FAIL verdict.**

---

## Scope

"Full verification" covers all 6 component files and both model files.

If the user says "verify causal_recurrence" or names a specific component, run Phase B' for that component only, but still follow all 5 steps above for that component.
