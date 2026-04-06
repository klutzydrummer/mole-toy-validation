# Phase B' Verification Report: mol_ffn

**Date:** 2026-04-06
**Supersedes:** prior report dated 2026-04-06
**Component:** `mol_ffn` — Mixture-of-LoRAs FFN
**Spec:** `references/components/mol_ffn.md`
**Implementation:** `phase1/components/mol_ffn.py` (sole tracked file — zone_e.py removed from scope)
**Sources checked:** `references/sources/papers/deepseek_v3_2412.19437.md`, `references/sources/code/deepseek_v3_moe.py`

---

## Overall verdict: PASS

All routing logic is correct and all equations trace verbatim to the DeepSeek-V3 paper.
Documentation issues identified in the prior report have been corrected in the spec:
stale line-number pointers updated to `phase1/components/mol_ffn.py`, method name corrected
(`get_mol_stats` → `get_load_stats`), and rank description corrected (rank 4 → rank 8).
No mathematical errors. No behavioral deviations from spec intent.

---

## Changes since last report (2026-04-06)

Spec corrections applied to `references/components/mol_ffn.md`:

1. **Description line**: "rank-4 LoRA adapter" corrected to "rank-8 LoRA adapter" (matches
   `MoLFFN.__init__` default `rank=8` at `mol_ffn.py:78`).

2. **Deviations table**: "Default: 8 experts, top-2, rank 4" corrected to "rank 8".

3. **Checklist items 1–12**: All stale `model.py` line references updated to correct
   `mol_ffn.py` locations. Specific updates:
   - Item 1: `model.py:279` → `mol_ffn.py:118`
   - Item 2: lines 280–282 → `mol_ffn.py:119–121`
   - Item 3: line 283 → `mol_ffn.py:122`
   - Item 4: line 220 → `mol_ffn.py:24`
   - Item 5: line 286 → `mol_ffn.py:125`
   - Item 6: line 316 → `mol_ffn.py:109` (register_buffer) and `mol_ffn.py:159` (no_grad block)
   - Item 8: line 299 → `mol_ffn.py:144–148`
   - Item 9: line 221, "rank=4, scale=0.25" → `mol_ffn.py:25`, "rank=8, scale=0.125"
   - Item 10: lines 290–292, 302 → `mol_ffn.py:144–145, 155`
   - Item 11: `model.get_mol_stats()` line 321 → `MoLFFN.get_load_stats()` at `mol_ffn.py:170`
   - Item 12: line 267 → `mol_ffn.py:106`

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
| Default: 8 experts, top-2, rank 8 | VERIFIED — MoLFFN.__init__ signature at line 78 |

### SingleLoRAFFN rank=72

`SingleLoRAFFN` (lines 31–63) defaults `rank=72`. Docstring states "rank=72 (exact capacity match)".
Capacity math: MoLFFN has 1 shared + 8 routed experts × rank 8 = 9 × 8 = 72 equivalent rank.
Matches Q2 ablation intent. — VERIFIED

---

## Verification checklist — per-item results

All checklist items verified against `phase1/components/mol_ffn.py`. Line numbers in spec are now current.

| # | Claim | Spec ref | Actual location | Status |
|---|-------|---------|----------------|--------|
| 1 | `torch.sigmoid` not softmax | `mol_ffn.py:118` | `mol_ffn.py:118` | VERIFIED |
| 2 | Bias for selection only; gather from `scores` not `biased` | `mol_ffn.py:119–121` | `mol_ffn.py:119–121` | VERIFIED |
| 3 | Normalization over selected experts only | `mol_ffn.py:122` | `mol_ffn.py:122` | VERIFIED |
| 4 | B matrix zero init | `mol_ffn.py:24` | `mol_ffn.py:24` | VERIFIED |
| 5 | One-hot gradient flow | `mol_ffn.py:125` | `mol_ffn.py:125` | VERIFIED |
| 6 | `expert_bias` is `register_buffer`; update inside `no_grad()` | `mol_ffn.py:109`, `mol_ffn.py:159` | `mol_ffn.py:109`, `mol_ffn.py:159` | VERIFIED |
| 7 | Load balance sign rule matches paper | `mol_ffn.py:165` | `mol_ffn.py:165` | VERIFIED |
| 8 | Composition order: gate/up before silu, down after | `mol_ffn.py:144–148` | `mol_ffn.py:144–148` | VERIFIED |
| 9 | `scale = 1.0 / rank` | `mol_ffn.py:25` | `mol_ffn.py:25` | VERIFIED |
| 10 | Shared adapters always active | `mol_ffn.py:144–145, 155` | `mol_ffn.py:144–145, 155` | VERIFIED |
| 11 | `MoLFFN.get_load_stats()` exists and returns `expert_balance` | `mol_ffn.py:170` | `mol_ffn.py:170`, `expert_balance` at line 179 | VERIFIED |
| 12 | Router `nn.Linear(d, n_experts, bias=False)` | `mol_ffn.py:106` | `mol_ffn.py:106` | VERIFIED |

---

## Summary

MoLFFN implementation is correct. All DeepSeek-V3 routing equations verified against paper and
reference code. SingleLoRAFFN rank=72 is correct for the Q2 ablation. All three documentation
issues from the prior report have been resolved in `references/components/mol_ffn.md`:
stale checklist line numbers (all items updated to mol_ffn.py), stale method name in item 11
(`get_mol_stats` → `get_load_stats`), and stale rank description (rank 4 → rank 8).
No implementation changes were required.
