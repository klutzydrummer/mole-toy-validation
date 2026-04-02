---
paths:
  - "references/components/**"
---

When editing any file in references/components/, cross-check the corresponding implementation before finalizing the change:

| Spec file | Implementation |
|-----------|---------------|
| causal_recurrence.md | phase2/model.py — CausalRecurrenceLayer, _parallel_scan |
| zone_ed_pipeline.md | phase2/model.py — ZoneE, ZoneD, BoundaryRouter |
| boundary_router.md | phase2/model.py — BoundaryRouter |
| mol_ffn.md | phase1/model.py — MoLFFN; phase2/model.py — InnerTransformer |
| mhc.md | phase1/model.py — mHCLayer |
| attention_rope_norms.md | phase1/model.py and phase2/model.py — attention blocks |
| mla_attention.md | phase1/model.py — MLACausalAttention |
| diff_attention.md | phase1/model.py — DifferentialCausalAttention, DiffMLAAttention |

Any factual claim in a spec file (equations, init values, expected ranges) must be verified against the implementation AND the primary source paper cited in that spec file. If the spec and implementation disagree, document the deviation explicitly in the spec's "Intentional deviations" section.
