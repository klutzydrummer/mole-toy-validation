# Phase B' Verification: mol_ffn
Date: 2026-03-26
Verdict: PASS with issues — All routing logic and batched-einsum composition correct; three documentation-only issues carried forward from previous report; one new issue added (spec composition code block is now stale).

---

## Scope of this run

This run re-verifies the full mol_ffn component following a refactor of `MoLFFN.forward` that replaced the per-expert loop with batched einsum matmuls. The mathematical result is asserted identical; this report confirms that claim and verifies all checklist items against the new code.

---

## Sources consulted

- `references/components/mol_ffn.md` (spec)
- `references/sources/papers/deepseek_v3_2412.19437.md`
- `references/sources/code/deepseek_v3_moe.py`
- `phase1/model.py` — `LoRAAdapter` (lines 234–244), `MoLFFN` (lines 282–403)
- `phase2/model.py` — `InnerTransformer` (lines 327–371)

---

## Authoritative Equations (vs. deepseek_v3_2412.19437.md)

| Claim | Status |
|---|---|
| Eq. 15: sigmoid (not softmax) for affinity scores | VERIFIED — `torch.sigmoid(logits)` at line 334 |
| Eq. 16: bias added for top-k selection only; unbiased scores used for weights | VERIFIED — `biased = scores + self.expert_bias` used only in `biased.topk()`; `scores.gather()` pulls from unbiased `scores` |
| Eq. 13: normalization over selected experts only | VERIFIED — `topk_scores.sum(dim=-1, keepdim=True)` sums only the top-k scores |
| Eq. 12: shared always active + weighted routed sum | VERIFIED — base + shared always added; expert contribution via weighted sum |
| Load-balance sign rule | VERIFIED — `bias += bias_step * (avg - counts).sign()` at line 384; underloaded → positive, overloaded → negative, matching paper |

All equations in the spec are traceable verbatim to `deepseek_v3_2412.19437.md` Section 2.1.

---

## Reference Code Snippets (vs. deepseek_v3_moe.py)

| Claim | Status |
|---|---|
| Gate.forward routing pattern (sigmoid → bias selection → unbiased gather → normalize) | VERIFIED verbatim |
| B matrix zero-init | VERIFIED — `torch.zeros(rank, d_out)` at line 240 |
| expert_bias as register_buffer | VERIFIED — `register_buffer("expert_bias", ...)` at line 325 |
| Bias update inside torch.no_grad() | VERIFIED — lines 378–384 |

---

## Batched Einsum Correctness (new in this run)

The refactored forward replaces a per-expert loop:

```python
# Old loop (removed):
for e in range(self.n_experts):
    w_e = expert_weights[:, :, e].unsqueeze(-1)
    gate = gate + w_e * self.expert_gate[e](x)
```

with batched einsum:

```python
A_gate = torch.stack([e.A for e in self.expert_gate])  # [E, d, rank]
B_gate = torch.stack([e.B for e in self.expert_gate])  # [E, rank, d_ff]
xA_gate  = torch.einsum("bld,edr->bler", x, A_gate)
xAB_gate = torch.einsum("bler,erd->bled", xA_gate, B_gate) * scale
gate = self.base_gate(x) + self.shared_gate(x) + (xAB_gate * w).sum(dim=2)
```

where `w = expert_weights.unsqueeze(-1)` is `[B, L, E, 1]`.

**Equivalence proof:**

For each expert `e`, the old loop contributed: `w_e * (x @ A_gate[e] @ B_gate[e]) * scale`

The batched code computes `xAB_gate[b,l,e,:] = (x[b,l,:] @ A_gate[e] @ B_gate[e]) * scale` for all `e` simultaneously, then `(xAB_gate * w).sum(dim=2)` sums over `e` with the corresponding weight — exactly `Σ_e w_e * (x @ A_e @ B_e) * scale`.

The operations are mathematically identical. The only difference is computation order and kernel dispatch count (3×2 batched GEMMs vs 3×2×n_experts sequential launches).

**Scale factor**: `scale = self.expert_gate[0].scale` is used (line 348). This assumes all experts have the same scale. Since all `LoRAAdapter` instances are created with the same `rank`, this is correct — `scale = 1.0 / rank` is identical for every expert.

**Verdict: VERIFIED — batched einsum is mathematically equivalent to the per-expert loop.**

---

## Our Implementation Claims (vs. phase1/model.py)

| Claim | Status |
|---|---|
| LoRAAdapter at lines 214–224 | INCORRECT (line ref only) — actual lines 234–244; content confirmed |
| MoLFFN at lines 227–332 | INCORRECT (line ref only) — actual lines 282–403; content confirmed |
| MoLFFN.forward at lines 274–318 | INCORRECT (line ref only) — actual lines 329–387; content confirmed |
| Composition code snippet in spec (per-expert loop) | INCORRECT (stale) — spec shows the old per-expert loop; actual code uses batched einsum. Mathematical result is identical but the implementation body no longer matches the spec snippet. |
| InnerTransformer use_mol=True, rank=4 | VERIFIED — line 344 `use_mol=True`, line 337 `mol_rank: int = 4` |
| Shared adapters always active | VERIFIED — `shared_gate`, `shared_up`, `shared_down` added unconditionally at lines 363–374 |
| Router: nn.Linear(d, n_experts, bias=False) | VERIFIED — line 322 |
| expert_bias as register_buffer (not nn.Parameter) | VERIFIED — line 325 |

---

## Intentional Deviations

| Deviation | Status |
|---|---|
| Experts are LoRA adapters, not full MLPs | VERIFIED |
| Three expert sets (gate/up/down) | VERIFIED |
| Shared LoRA always active | VERIFIED |
| expert_bias as register_buffer | VERIFIED |
| bias_step = 0.01 | VERIFIED — line 327 |
| No node/group-limited routing | VERIFIED |
| 1e-8 epsilon in normalization | VERIFIED — line 338 |
| Router is nn.Linear with bias=False | VERIFIED |
| Default 8 experts, top-2, rank 8 (Phase 1) / rank 4 (Phase 2 InnerTransformer) | INCORRECT (spec says "rank 4" without distinguishing phases; Phase 1 MoLFFN `__init__` defaults `rank=8` at line 295; Phase 2 InnerTransformer defaults `mol_rank=4` at line 337) |

---

## Verification Checklist

| Item | Status | Notes |
|---|---|---|
| 1. Sigmoid not softmax at model.py:279 | VERIFIED (stale line) | Actual line 334; `torch.sigmoid(logits)` confirmed |
| 2. Bias for selection only at lines 280–282 | VERIFIED (stale lines) | Actual lines 335–337; pattern confirmed |
| 3. Normalization over selected experts at line 283 | VERIFIED (stale line) | Actual line 338; `topk_scores.sum()` confirmed |
| 4. B matrix zero init at line 220 | VERIFIED (stale line) | Actual line 240; `torch.zeros(rank, d_out)` confirmed |
| 5. One-hot gradient flow at line 286 | VERIFIED (stale line) | Actual line 341; `F.one_hot(topk_idx, self.n_experts)` confirmed |
| 6. Bias not gradient-trained: register_buffer + no_grad | VERIFIED | Lines 325 and 378 |
| 7. Load balance sign rule | VERIFIED | `bias_step * (avg - counts).sign()` at line 384 |
| 8. Composition order (gate/up before silu, down after) | VERIFIED | Lines 363–374 — gate and up accumulated before `F.silu(gate) * up` at line 367 |
| 9. Scale factor = 1/rank | VERIFIED | `self.scale = 1.0 / rank` at line 241; used as `scale = self.expert_gate[0].scale` in batched path |
| 10. Shared adapter always active | VERIFIED | Lines 363–364 and 374 |
| 11. expert_balance > 0.7 (runtime check) | NOT VERIFIED (runtime) | `model.get_mol_stats()` on ToyTransformer calls through to `get_load_stats()` on MoLFFN — method chain confirmed correct; runtime value depends on training |
| 12. No bias in router linear | VERIFIED | `nn.Linear(d, n_experts, bias=False)` at line 322 |

---

## Issues

### Issue 1 (documentation-only): Stale line number citations
All line number citations in the spec are off by 20–65 lines. Code grew after the spec was written. No behavioral impact. The spec "Our implementation" table should be updated to reflect current line numbers.

### Issue 2 (documentation-only): Composition code snippet is stale
The spec's "MoLFFN composition" section shows a per-expert for-loop. The actual implementation now uses batched einsum (lines 344–374). The mathematical result is identical (verified above), but a reader comparing the spec snippet to the code will find no match. The spec code block should be updated to show the batched form.

### Issue 3 (documentation-only): Rank default ambiguity
The "Intentional deviations" table says "Default: rank 4" without distinguishing Phase 1 vs Phase 2. Phase 1 `MoLFFN.__init__` defaults `rank=8`; Phase 2 `InnerTransformer.__init__` defaults `mol_rank=4`. Should be documented separately.

---

## Summary

All routing logic is functionally correct and traceable to the DeepSeek-V3 paper: sigmoid affinity scores, bias used only for top-k selection, unbiased scores for weight normalization, B matrix zero-initialized at init, load-balance sign rule matching paper description. The batched einsum refactor is mathematically equivalent to the prior per-expert loop — verified by algebraic expansion. `InnerTransformer` in `phase2/model.py` correctly instantiates MoLFFN blocks via `use_mol=True` with `rank=4`. The three issues found are all documentation-only: stale line numbers, a stale code snippet (behavior unchanged), and an undifferentiated rank default.

**Overall verdict: PASS with issues** (issues are documentation-only; no behavioral discrepancy found).
