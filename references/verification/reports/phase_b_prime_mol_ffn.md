# Phase B' Verification Report: mol_ffn

**Date:** 2026-04-05
**Supersedes:** prior report dated 2026-04-02
**Component:** `mol_ffn` — Mixture-of-LoRAs FFN
**Spec:** `references/components/mol_ffn.md`
**Implementation:** `phase1/components/mol_ffn.py`, `phase2/components/zone_e.py`
**Sources checked:** `references/sources/papers/deepseek_v3_2412.19437.md`, `references/sources/code/deepseek_v3_moe.py`

---

## Overall verdict: PASS with issues

All routing logic is correct and all equations trace verbatim to the DeepSeek-V3 paper.
The issues are documentation-only (stale line numbers) plus one tracking configuration
concern about zone_e.py being in TRACKED_COMPONENTS for this component.

---

## Per-claim status

### Authoritative equations (DeepSeek-V3 Eq. 12–16 + bias update)

| Claim | Status | Notes |
|-------|--------|-------|
| Affinity scores via sigmoid (not softmax): s_i = sigmoid(u·e_i) | VERIFIED | mol_ffn.py:118 uses torch.sigmoid |
| Bias added for top-k selection only: s'_i = s_i + b_i | VERIFIED | mol_ffn.py:119 adds bias before topk |
| Unbiased scores for weight normalization: normalize s_i not s'_i | VERIFIED | mol_ffn.py:122 normalizes unbiased scores |
| B matrix zero-initialized | VERIFIED | mol_ffn.py:24: nn.Parameter(torch.zeros(...)) |
| Load-balance bias update: sign rule, no_grad | VERIFIED | mol_ffn.py:159/165, torch.no_grad() |
| Shared adapters always active (not gated) | VERIFIED | mol_ffn.py:144–145, 155 |

### Reference implementation cross-check

| Claim | Status | Notes |
|-------|--------|-------|
| One-hot for differentiable gradient flow | VERIFIED | mol_ffn.py:125 |
| SiLU activation in LoRAAdapter | VERIFIED | mol_ffn.py:148 |
| scale = 1 / sqrt(rank) | VERIFIED | mol_ffn.py:25 |
| Batched einsum for expert composition | VERIFIED | mol_ffn.py — mathematically equivalent to per-expert loop |

### Intentional deviations

| Deviation | Status | Notes |
|-----------|--------|-------|
| Batched einsum instead of per-expert loop | VERIFIED | Algebraically equivalent, confirmed |
| SiLU not GeLU activation | VERIFIED | mol_ffn.py:148 |
| Rank default ambiguity | INCORRECT (docs) | Spec says "rank 4" in deviations table but MoLFFN.__init__ defaults rank=8. "Rank 4" referred to InnerTransformer which no longer exists. Needs spec update. |

### Verification checklist

| Item | Status | Notes |
|------|--------|-------|
| Sigmoid affinity (not softmax) | VERIFIED | mol_ffn.py:118 |
| Bias used only for top-k, not normalization | VERIFIED | mol_ffn.py:119–122 |
| B zero-initialized | VERIFIED | mol_ffn.py:24 |
| Load-balance sign rule | VERIFIED | mol_ffn.py:159–165 |
| Shared adapters active for all tokens | VERIFIED | mol_ffn.py:144–145, 155 |
| All checklist line citations | STALE POINTERS | All 12 items reference old `phase1/model.py` line numbers. Correct file is `phase1/components/mol_ffn.py`. Key mappings: sigmoid→118, bias/topk→119–121, normalize→122, B-zero→24, one-hot→125, no_grad→159/165, silu→148, scale→25, shared→144–145/155, router→106 |

---

## Issues requiring spec fixes

**Issue 1 — Stale line numbers (documentation only):**
All checklist items cite `phase1/model.py` line numbers. Code now at `phase1/components/mol_ffn.py`.
No correctness impact.

**Issue 2 — zone_e.py tracking concern:**
`zone_e.py` is listed in `TRACKED_COMPONENTS["mol_ffn"]` and tracked by this component, but no
Phase 2 config currently passes `use_mol=True` to `TransformerEncoder`. The `use_mol` flag exists
in `TransformerBlock` but is never activated in any `zone_e.py` encoder. Changes to `zone_e.py`
for unrelated reasons (encoder architecture) will incorrectly flag `mol_ffn` as stale.
**Recommendation:** Remove `zone_e.py` from `TRACKED_COMPONENTS["mol_ffn"]` until a Phase 2 config
actually uses MoLFFN in an encoder.

**Issue 3 — Rank default mismatch (documentation only):**
Spec's intentional deviations table references "rank 4" in the context of InnerTransformer, which
no longer exists. `MoLFFN.__init__` defaults `rank=8`. The "rank 4" description is stale.

---

## Summary

MoLFFN implementation is correct. All DeepSeek-V3 routing equations verified. Three documentation
issues: stale line numbers (all checklist items), a tracking configuration mismatch (zone_e.py
in TRACKED_COMPONENTS but no active use_mol config), and a stale rank default description.
The tracking concern is the most actionable — it will cause false staleness alerts.
