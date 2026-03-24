# Phase B' Verification: mol_ffn
Date: 2026-03-23
Verdict: PASS with issues — All routing logic correct; three documentation-only issues found.

---

## Per-claim status

### Authoritative Equations (vs. deepseek_v3_2412.19437.md)
- Sigmoid routing scores (not softmax): VERIFIED
- Bias added for selection only, unbiased scores used for weights: VERIFIED
- Normalized top-k weights: VERIFIED
- Load-balance bias update sign rule: VERIFIED
- MoL composition order (gate/up before silu, down after): VERIFIED

### Reference Code Snippets (vs. deepseek_v3_moe.py)
- Gate class (sigmoid scores + bias selection): VERIFIED verbatim
- Expert weight normalization: VERIFIED verbatim
- B matrix zero-init in LoRA: VERIFIED verbatim
- expert_bias as register_buffer updated in torch.no_grad(): VERIFIED verbatim

### Our Implementation Claims (vs. phase1/model.py MoLFFN, phase2/model.py InnerTransformer)
- LoRAAdapter at :214–224 — INCORRECT (line ref only): actual lines 234–244; content confirmed
- MoLFFN at :227–332 — INCORRECT (line ref only): actual starts at 282; content confirmed
- MoLFFN.forward at :274–318 — INCORRECT (line ref only): actual starts at 329; content confirmed
- InnerTransformer use_mol=True, rank-4: VERIFIED
- Shared adapters always active: VERIFIED
- expert_bias update inside torch.no_grad(): VERIFIED

### Intentional Deviations
- Sigmoid not softmax for routing scores: VERIFIED
- Bias for selection only: VERIFIED
- Shared LoRA always active: VERIFIED
- Default rank=8 (Phase 1) / rank=4 (Phase 2) — INCORRECT (spec conflates both): spec says "Default: rank 4" without distinguishing Phase 1 vs Phase 2 contexts. Phase 1 MoLFFN defaults rank=8; rank=4 is InnerTransformer only.

### Verification Checklist (11 items)
- Items 1–10: VERIFIED — all stale line refs (off 20–65 lines); behavior confirmed
- Item 11 (model.get_mol_stats()) — INCORRECT: actual method on MoLFFN is `get_load_stats()` at line 376; calling spec's name would raise AttributeError

---

## Issues

1. **All line number citations are stale by 20–65 lines.** Code grew after spec was written. No behavioral impact.

2. **Checklist item 11 wrong method name.** Spec says `model.get_mol_stats()` but the method on `MoLFFN` is `get_load_stats()` (line 376). Note: `ToyTransformer` and `InnerTransformer` *do* expose `get_mol_stats()` which calls through to `get_load_stats()` on each block — so at the model level the spec name is correct; it is incorrect only as a description of the MoLFFN method itself.

3. **Rank default ambiguity.** Deviation table says "Default: rank 4" without distinguishing: MoLFFN in phase1 defaults rank=8; InnerTransformer in phase2 defaults rank=4. Should be documented separately.

---

## Summary

All routing logic is functionally correct — sigmoid not softmax, bias used only for selection, unbiased scores for weight normalization, B matrix zero-initialized, load-balance sign rule matching paper. InnerTransformer correctly uses MoLFFN via use_mol=True with rank-4 adapters. The three issues are documentation-only: stale line numbers, a method name that is correct at the model level but ambiguous at the MoLFFN level, and an undifferentiated rank default.
