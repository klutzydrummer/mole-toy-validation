---
paths:
  - "references/components/**"
  - "references/verification/**"
  - "phase1/components/**"
  - "phase2/components/**"
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

1. **Reads** the component spec: `references/components/<component>.md`
   - The spec contains a "Sources" table — this tells you which Phase A files to read next.
2. **Reads** every source file listed in the spec's "Sources" table:
   `references/sources/papers/<paper>.md` and `references/sources/code/<file>.py`
3. **Reads** the implementation files for this component:
   Run `python utils/verify.py status` — the `TRACKED_COMPONENTS` dict in `utils/verify.py`
   maps each component name to its implementation files.
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
   - Any stale line number pointers found: note the correct current location

---

## After agents complete

1. Review the reports in `references/verification/reports/`
2. If all pass (or pass-with-issues where issues are acceptable): `python utils/verify.py update --result pass --report references/verification/reports/ <component> [...]`
   Run `python utils/verify.py status` to see all tracked component names and current state.
3. If any fail: fix the root cause in the spec or implementation, then re-run
4. `git commit` including the updated `references/verification/last_verified.json` and reports

**DO NOT run `verify.py update --result pass` if any component has a FAIL verdict.**

---

## Scope

"Full verification" covers all 8 components (see table above). The facade model files
(`phase1/model.py`, `phase2/model.py`) contain no math and are not separately verified —
verification targets the component files they import from.

If a specific component is named, run Phase B' for that component only, but still follow
all 5 steps above for it.

Valid component names are the keys of `TRACKED_COMPONENTS` in `utils/verify.py`.
Run `python utils/verify.py status` to see the current list and verification state.
