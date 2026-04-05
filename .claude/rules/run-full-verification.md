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

**Phase A and B already exist.** Full verification means running Phase B': dispatching agents to validate that every claim in every Phase B component file is traceable to a Phase A source, and that our implementation in the component files under `phase1/components/` and `phase2/components/` matches the spec (or has a documented, intentional deviation).

---

## ⚠️ Important: spec line number updates belong HERE, not as standalone operations

When a component's implementation files move or change, spec line number pointers become stale.
**Do NOT fix stale line numbers as a separate mass-update pass.** The correct place to fix them
is step 3 of the Phase B' procedure below: the verification agent reads the current implementation,
finds the code, and updates any stale pointers in the spec as part of writing its report.

Rationale: a line number update done in isolation is unverified — it asserts a pointer without
confirming the code at that location still matches the spec's mathematical claims. Only inside a
full verification run is the pointer update coupled with an actual correctness check.

---

## Phase B' procedure — run for each component

For each component spec in `references/components/`, dispatch a subagent that:

1. **Reads** the component spec (`references/components/<component>.md`)
2. **Reads** every source file cited in the spec's "Sources" table (`references/sources/papers/<paper>.md` and `references/sources/code/<file>.py`)
3. **Reads** the component implementation file(s) listed in the table below
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

| Component spec | Sources | Implementation files |
|---------------|---------|---------------------|
| `causal_recurrence.md` | `griffin_2402.19427.md`, `griffin_rglru.py` | `phase2/components/causal_recurrence.py` |
| `zone_ed_pipeline.md` | `hnet_2507.07955.md`, `hnet_boundary.py`, `dlcm.md` | `phase2/components/zone_e.py`, `phase2/components/zone_d.py` |
| `boundary_router.md` | `hnet_2507.07955.md`, `hnet_boundary.py`, `dlcm.md` | `phase2/components/boundary_router.py` |
| `mol_ffn.md` | `deepseek_v3_2412.19437.md`, `deepseek_v3_moe.py` | `phase1/components/mol_ffn.py`, `phase2/components/zone_e.py` |
| `mhc.md` | `mhc_2512.24880.md`, `mhc_hyper_connections.py` | `phase1/components/mhc.py` |
| `attention_rope_norms.md` | `rope_2104.09864.md`, `rmsnorm_1910.07467.md`, `swiglu_2002.05202.md`, `rope.py`, `rmsnorm.py`, `swiglu.py` | `phase1/components/attention_rope_norms.py`, `phase1/components/_shared.py` |
| `mla_attention.md` | `mla_deepseek_v2_2405.04434.md`, `mla_attention.py` | `phase1/components/mla_attention.py` |
| `diff_attention.md` | `diff_attn_v1_2410.05258.md`, `diff_attn_v2_2026_01.md`, `mla_deepseek_v2_2405.04434.md`, `mla_attention.py` | `phase1/components/diff_attention.py` |

---

## After agents complete

1. Review the reports in `references/verification/reports/`
2. If all pass (or pass-with-issues where issues are acceptable): `python utils/verify.py update --result pass --report references/verification/reports/ <component> [...]`
   Use the component names from the table above (e.g., `causal_recurrence`, `mla_attention`, etc.).
3. If any fail: fix the root cause in the spec or implementation, then re-run
4. `git commit` including the updated `references/verification/last_verified.json` and reports

**DO NOT run `verify.py update --result pass` if any component has a FAIL verdict.**

---

## Scope

"Full verification" covers all 8 components (see table above). The facade model files
(`phase1/model.py`, `phase2/model.py`) contain no math and are not separately verified —
verification targets the component files they import from.

If the user says "verify causal_recurrence" or names a specific component, run Phase B'
for that component only, but still follow all 5 steps above for that component.

Component names accepted by `verify.py update`:
`attention_rope_norms`, `mla_attention`, `diff_attention`, `mhc`, `mol_ffn`,
`causal_recurrence`, `zone_ed_pipeline`, `boundary_router`
