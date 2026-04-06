# Phase B' Verification Report: causal_recurrence

**Date:** 2026-04-05
**Supersedes:** prior report dated 2026-04-02
**Component:** `causal_recurrence` — Griffin RG-LRU / CausalRecurrenceLayer
**Spec:** `references/components/causal_recurrence.md`
**Implementation:** `phase2/components/causal_recurrence.py`
**Sources checked:** `references/sources/papers/griffin_2402.19427.md`, `references/sources/code/griffin_rglru.py`

---

## Overall verdict: PASS with issues

All equations and reference code snippets are correctly implemented. The implementation in
`phase2/components/causal_recurrence.py` correctly implements Griffin RG-LRU with parallel scan.
Two issues: stale file path citations (all point to old `phase2/model.py`; code lives in
`phase2/components/causal_recurrence.py`), and one incorrect checklist item referencing a
`ZoneD.recurrence` that does not exist.

---

## Per-claim status

### Authoritative equations (Griffin RG-LRU)

| Claim | Status | Notes |
|-------|--------|-------|
| Recurrent coefficient: a = sigmoid(log_a)^exp(Λ) | VERIFIED | causal_recurrence.py:111 |
| Input gate: r, i = sigmoid(linear(x)) | VERIFIED | causal_recurrence.py:104–105 |
| Hidden state: h_t = a·h_{t-1} + sqrt(1-a²)·(i⊙x) | VERIFIED | causal_recurrence.py:117 |
| Float32 promotion for sigmoid (numerical stability) | VERIFIED | causal_recurrence.py:111 — `.float()` before sigmoid |
| sqrt clamped: sqrt(clamp(1-a², min=0)) | VERIFIED | causal_recurrence.py:117 |
| Parallel scan for training | VERIFIED | causal_recurrence.py:16 (_parallel_scan) |
| log_A clamped to [-8, 8] | VERIFIED | causal_recurrence.py — log_a parameter init and usage |
| h_state maintained in float32 | VERIFIED | dtype=torch.float32 in _parallel_scan |

### Reference code snippets

| Claim | Status | Notes |
|-------|--------|-------|
| Gates as sigmoid(linear) | VERIFIED | Matches griffin_rglru.py verbatim |
| Float32 promotion for log_a | VERIFIED | Matches reference pattern |
| Parallel scan recurrence | VERIFIED | Algebraically equivalent to reference sequential scan |

### Our implementation

| Claim | Status | Notes |
|-------|--------|-------|
| CausalRecurrenceLayer class | STALE POINTER | Spec cites `phase2/model.py:51`; correct: `phase2/components/causal_recurrence.py:51` |
| _parallel_scan function | STALE POINTER | Spec cites `phase2/model.py:16`; correct: `phase2/components/causal_recurrence.py:16` |
| Gates at lines 130–131 | STALE POINTER | Spec cites `phase2/model.py:130–131`; correct: `phase2/components/causal_recurrence.py:104–105` |
| Float32 sigmoid at line 139 | STALE POINTER | Spec cites `phase2/model.py:139`; correct: `phase2/components/causal_recurrence.py:111` |
| sqrt clamp at line 145 | STALE POINTER | Spec cites `phase2/model.py:145`; correct: `phase2/components/causal_recurrence.py:117` |
| Zone E 3-layer stack | STALE POINTER | Spec cites `phase2/model.py:281`; correct: `phase2/components/zone_e.py:31–33` |
| zone_e.py:31 and :58 citations | VERIFIED | These two citations are already correct |

All implementation behaviors are correct at the new locations.

### Intentional deviations

| Deviation | Status | Notes |
|-----------|--------|-------|
| Parallel scan (not sequential) for training | VERIFIED | _parallel_scan at causal_recurrence.py:16 |
| log_a_init parameter (not hardcoded 7.5) | VERIFIED | Parameterized; Zone E uses log_a_init=3.0 |
| No Mamba-2 / SSM: pure RG-LRU | VERIFIED | No SSM code anywhere in causal_recurrence.py |

### Verification checklist

| Item | Status | Notes |
|------|--------|-------|
| Items 1–8 (gates, recurrence, float32, sqrt, scan, dtype) | VERIFIED | All correct at new file locations |
| Item 9 — ZoneD.recurrence at phase2/model.py:398 | INCORRECT | ZoneD.recurrence does not exist. SimpleDecoder (`phase2/components/zone_d.py`) uses `_parallel_scan` directly for EMA smoothing — there is no `CausalRecurrenceLayer` stack in Zone D. Spec's own deviations section 2 correctly documents this. Checklist item 9 should be removed or rewritten. |

---

## Issues requiring spec fixes

**Issue 1 — Stale file path citations (all implementation pointers):**
Every `phase2/model.py` citation in the "Our implementation" section is stale. Code extracted
to `phase2/components/causal_recurrence.py` and `phase2/components/zone_e.py`. No correctness
impact — the math at the new locations is correct.

**Issue 2 — Incorrect checklist item 9:**
Checklist item 9 cites `ZoneD.recurrence at phase2/model.py:398`. This class and attribute do
not exist. `SimpleDecoder` in `phase2/components/zone_d.py` uses `_parallel_scan` directly for
EMA smoothing; it does not contain a `CausalRecurrenceLayer`. The spec's own deviations table
correctly describes this. The checklist item is a leftover from the old architecture.
**Recommendation:** Remove checklist item 9 or rewrite as: "Confirm EMA smoothing in
SimpleDecoder uses _parallel_scan (not CausalRecurrenceLayer)."

---

## Summary

CausalRecurrenceLayer is mathematically correct. Griffin RG-LRU equations (gates, recurrent
coefficient, sqrt normalization, float32 promotion, parallel scan) all verified against the
reference. Two issues are documentation-only: stale file path citations throughout the
implementation section (all point to old monolith `phase2/model.py`), and one incorrect
checklist item referencing a `ZoneD.recurrence` that was removed in the architecture refactor.
