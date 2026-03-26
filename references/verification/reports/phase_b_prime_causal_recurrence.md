# Phase B' Verification: causal_recurrence
Date: 2026-03-26
Verdict: PASS with issues — All issues are line-number drift only; no logic or formula errors found.

---

## Per-claim status

### Authoritative Equations (vs. griffin_2402.19427.md)
- Eq. (1)–(4), Section 2.4: VERIFIED verbatim
- Eq. (6) log-space, Appendix A: VERIFIED verbatim
- sqrt(1−a_t²) norm-preservation identity: VERIFIED verbatim

### Reference Code Snippets (vs. griffin_rglru.py)
- Gate/coeff block (spec lines 304–319): VERIFIED — matches at source lines 303–319
- Sequential scan (spec lines 153–157): VERIFIED — body matches at lines 153–155
- rnn_param_init (spec lines 169–177): VERIFIED — content matches; softplus guard omitted editorially
- SqrtBoundDerivative (spec lines 182–196): VERIFIED — verbatim match

### Our Implementation Claims (vs. phase2/model.py)
- CausalRecurrenceLayer at :83 — INCORRECT (line ref only): actual line 82; content confirmed
- _parallel_scan at :44 — INCORRECT (line ref only): actual line 43; content confirmed
- ZoneE ModuleList at :281 — INCORRECT (line ref only): actual line 288; content confirmed
- ZoneD ModuleList at :398 — INCORRECT (line ref only): actual line 406; content confirmed

### Intentional Deviations
- Dev 1 (parallel scan replaces sequential): VERIFIED
- Dev 2 (log-space cumsum, clamped [-80,0]): VERIFIED
- Dev 3 (float32 sigmoid for decay) — VERIFIED behavior; spec line :139 stale → actual :145
- Dev 4 (sqrt clamp 1e-6) — VERIFIED behavior; spec line :145 stale → actual :151
- Dev 5 (depthwise conv kernel=4) — VERIFIED behavior; spec :102–127 stale → actual :110–135
- Dev 6 (log_a_init ZoneE=3.0, ZoneD=0.0): VERIFIED
- Dev 7 (out_proj + RMSNorm) — VERIFIED behavior; spec :115–116,149 stale → actual :122–124,155

### Verification Checklist (10 items)
- Item 1 (sqrt(1−a_t²) present): VERIFIED — stale line ref
- Item 2 (float32 promotion for a_base): VERIFIED — exact line
- Item 3 (parallel scan, no Python loop): VERIFIED — stale line ref
- Item 4 (log clamped to [-80, 0]): VERIFIED — stale line ref
- Item 5 (h0 initial state handled): VERIFIED — exact line
- Item 6 (dtype cast-back after scan): VERIFIED — stale line ref
- Item 7 (ZoneE log_a_init=3.0): VERIFIED — stale line ref
- Item 8 (ZoneD log_a_init=0.0): VERIFIED — stale line ref
- Item 9 (depthwise conv causal pad): VERIFIED — stale line ref
- Item 10 (1000-step smoke test gate): VERIFIED — exact line

---

## Issues

All issues are documentation drift only — spec line references are off by 6–8 lines throughout due to code edits since last verification. No logic, formula, or behavioral errors.

| Location | Spec says | Actual |
|----------|-----------|--------|
| CausalRecurrenceLayer | :83 | :82 |
| _parallel_scan | :44 | :43 |
| ZoneE recurrence ModuleList | :281 | :288 |
| ZoneD recurrence ModuleList | :398 | :406 |
| Dev 3 float32 sigmoid | :139 | :145 |
| Dev 4 sqrt clamp | :145 | :151 |
| Dev 5 depthwise conv | :102–127 | :110–135 |
| Dev 7 out_proj+RMSNorm | :115–116,149 | :122–124,155 |

---

## Summary

No logic, formula, or behavioral errors found. The implementation correctly applies sqrt(1−a_t²) normalization, float32 promotion for the decay path, float32 accumulation in the parallel scan with correct dtype cast-back, and role-specific log_a_init (3.0 for Zone E, 0.0 for Zone D). The only finding is pervasive line-number drift (~6–8 lines) in the spec's implementation pointers — documentation maintenance only, no correctness impact.
