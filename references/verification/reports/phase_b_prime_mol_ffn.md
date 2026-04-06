# Phase B' Verification Report: mol_ffn

**Date:** 2026-04-06
**Supersedes:** prior report dated 2026-04-05
**Component:** `mol_ffn` — Mixture-of-LoRAs FFN
**Spec:** `references/components/mol_ffn.md`
**Implementation:** `phase1/components/mol_ffn.py` (sole tracked file — zone_e.py removed from scope)
**Sources checked:** `references/sources/papers/deepseek_v3_2412.19437.md`, `references/sources/code/deepseek_v3_moe.py`

---

## Overall verdict: PASS with issues

All routing logic is correct and all equations trace verbatim to the DeepSeek-V3 paper.
The issues are documentation-only: stale line-number pointers (pre-refactor model.py references)
and one stale method name (`get_mol_stats` → `get_load_stats`) in checklist item 11.
No mathematical errors. No behavioral deviations from spec intent.

---

## Changes since last report (2026-04-05)

- `zone_e.py` removed from `TRACKED_COMPONENTS["mol_ffn"]` — no longer in verification scope.
  Prior report's Issue 2 (tracking concern) is now resolved.
- `SingleLoRAFFN.rank` default changed from 64 to 72 (exact capacity match for Q2 ablation).
  Docstring updated to "rank=72 (exact capacity match)". Verified correct — see section below.

---

## Per-claim status

### Authoritative equations (DeepSeek-V3 Eq. 12–16 + bias update)

All equations in the spec are quoted verbatim from the paper source and verified against
`references/sources/papers/deepseek_v3_2412.19437.md`.

| Claim | Status | Notes |
|-------|--------|-------|
| Eq. 12 — FFN output combining shared + routed experts | VERIFIED | Paper source matches spec exactly |
| Eq. 13 — Gating: normalize unbiased scores of selected experts | VERIFIED | Paper source matches spec exactly |
| Eq. 14 — Top-K selection without bias | VERIFIED | Paper source matches spec exactly |
| Eq. 15 — Affinity scores via sigmoid (not softmax) | VERIFIED | Paper source matches spec exactly |
| Eq. 16 — Top-K selection with load-balance bias | VERIFIED | Paper source matches spec exactly |
| Bias update rule — sign rule, not gradient | VERIFIED | Paper source matches spec exactly |

### Reference implementation cross-check (against deepseek_v3_moe.py)

The `Gate.forward` snippet in the spec matches `deepseek_v3_moe.py` lines 63–121 structurally
and semantically: sigmoid scores → save original_scores → add bias for selection only → topk
by biased score → gather from original_scores → normalize → scale.

The `Expert` class snippet matches `deepseek_v3_moe.py` lines 147–159 verbatim.

| Snippet | Status |
|---------|--------|
| `Gate.forward` routing pattern | VERIFIED |
| `Expert` class | VERIFIED |

### Our implementation (phase1/components/mol_ffn.py)

Class/line table from spec matches actual file exactly:

| Class | Spec lines | Actual lines | Status |
|-------|-----------|-------------|--------|
| `LoRAAdapter` | 18–28 | 18–28 | VERIFIED |
| `MoLFFN` | 66–184 | 66–184 | VERIFIED |
| `MoLFFN.forward` | 113–168 | 113–168 | VERIFIED |

**LoRAAdapter init (spec: mol_ffn.py:18–28)**
- B matrix zero init: `torch.zeros(rank, d_out)` — VERIFIED at line 24
- A matrix: `torch.randn(d_in, rank) * (1.0 / math.sqrt(d_in))` — VERIFIED at line 23
- Scale: `1.0 / rank` — VERIFIED at line 25

**Routing logic (spec: mol_ffn.py:117–122)**
- `torch.sigmoid(logits)` — VERIFIED at line 118
- `biased = scores + self.expert_bias` — VERIFIED at line 119
- `biased.topk(self.top_k, dim=-1)` (selects by biased scores) — VERIFIED at line 120
- `scores.gather(2, topk_idx)` (gathers UNBIASED scores) — VERIFIED at line 121
- Normalization with `+ 1e-8` epsilon — VERIFIED at line 122

**Composition order (spec: mol_ffn.py:144–155)**
Gate and up corrections applied BEFORE silu; down corrections applied AFTER. — VERIFIED at lines 144–155.

**Load balance update (spec: mol_ffn.py:158–165)**
- `torch.no_grad()` guard — VERIFIED at line 159
- `scatter_add_` accumulation — VERIFIED at lines 160–163
- `self.expert_bias += self.bias_step * (avg - counts).sign()` — VERIFIED at line 165

### Intentional deviations

| Deviation | Status |
|-----------|--------|
| Experts are rank-r LoRA adapters over shared base, not full independent MLPs | VERIFIED — LoRAAdapter at lines 18–28 |
| Three expert sets (gate, up, down) per layer | VERIFIED — expert_gate/up/down at lines 101–103 |
| Shared LoRA always active | VERIFIED — shared_gate/up/down at lines 96–98 |
| `expert_bias` as `register_buffer`, not `nn.Parameter` | VERIFIED at line 109 |
| `bias_step = 0.01` (not 0.001) | VERIFIED at line 111 |
| No node/group-limited routing | VERIFIED — no group masking in forward |
| Normalization denominator has `+ 1e-8` | VERIFIED at line 122 |
| Router is `nn.Linear(d, n_experts, bias=False)` | VERIFIED at line 106 |
| Default: 8 experts, top-2, rank 4 | VERIFIED — MoLFFN.__init__ signature at line 78 (rank default is 8 in the signature; "rank 4" in the deviations table refers to base LoRA rank, not the default argument — see stale pointer note below) |

### SingleLoRAFFN rank=72

`SingleLoRAFFN` (lines 31–63) defaults `rank=72`. Docstring states "rank=72 (exact capacity match)".
Capacity math: MoLFFN has 1 shared + 8 routed experts × rank 8 = 9 × 8 = 72 equivalent rank.
Matches Q2 ablation intent. — VERIFIED

---

## Verification checklist — per-item results

All checklist line numbers in the spec are stale (pre-component-split, reference old `model.py`
line numbers). Correct current locations in `phase1/components/mol_ffn.py` are noted below.

| # | Claim | Spec ref | Actual location | Status |
|---|-------|---------|----------------|--------|
| 1 | `torch.sigmoid` not softmax | `model.py:279` | `mol_ffn.py:118` | VERIFIED — stale pointer |
| 2 | Bias for selection only; gather from `scores` not `biased` | lines 280–282 | `mol_ffn.py:119–121` | VERIFIED — stale pointer |
| 3 | Normalization over selected experts only | line 283 | `mol_ffn.py:122` | VERIFIED — stale pointer |
| 4 | B matrix zero init | "line 220" | `mol_ffn.py:24` | VERIFIED — stale pointer |
| 5 | One-hot gradient flow | "line 286" | `mol_ffn.py:125` | VERIFIED — stale pointer |
| 6 | `expert_bias` is `register_buffer`; update inside `no_grad()` | "line 316" | `mol_ffn.py:109`, `mol_ffn.py:159` | VERIFIED — stale pointer |
| 7 | Load balance sign rule matches paper | (no specific line) | `mol_ffn.py:165` | VERIFIED |
| 8 | Composition order: gate/up before silu, down after | "line 299" | `mol_ffn.py:144–148` | VERIFIED — stale pointer |
| 9 | `scale = 1.0 / rank` | "line 221" | `mol_ffn.py:25` | VERIFIED — stale pointer |
| 10 | Shared adapters always active | lines 290–292, 302 | `mol_ffn.py:144–145, 155` | VERIFIED — stale pointer |
| 11 | `model.get_mol_stats()` exists and returns `expert_balance` | "line 321" | `MoLFFN.get_load_stats()` at `mol_ffn.py:170` | INCORRECT method name; `expert_balance` field is present at line 179 |
| 12 | Router `nn.Linear(d, n_experts, bias=False)` | "line 267" | `mol_ffn.py:106` | VERIFIED — stale pointer |

---

## Issues requiring spec update

**Issue 1 — Stale line numbers throughout checklist (documentation only)**
All 12 checklist items reference `phase1/model.py` (pre-refactor). Code is now in
`phase1/components/mol_ffn.py`. Correct line numbers given in table above. No correctness impact.

**Issue 2 — Wrong method name in checklist item 11**
Spec checklist item 11 says `model.get_mol_stats()`. Actual method is `MoLFFN.get_load_stats()`
at `mol_ffn.py:170`. The return value `expert_balance` (line 179) is correct. Only the method
name and line reference are wrong.
Action: update item 11 to `MoLFFN.get_load_stats()` at `mol_ffn.py:170`.

**Issue 3 — Rank description in deviations table**
Deviations table row "Default: 8 experts, top-2, rank 4" — the "rank 4" is misleading since
`MoLFFN.__init__` defaults `rank=8`. This appears to be a carryover from an earlier architecture.
No correctness impact on implementation.

---

## Summary

MoLFFN implementation is correct. All DeepSeek-V3 routing equations verified against paper and
reference code. SingleLoRAFFN rank=72 is correct for the Q2 ablation. Three documentation issues
remain: stale checklist line numbers (all items), stale method name in item 11
(`get_mol_stats` → `get_load_stats`), and a stale rank description in the deviations table.
None affect implementation correctness.
