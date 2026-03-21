---
paths:
  - "phase2/model.py"
  - "phase2/train.py"
---

When working with any Phase 2 file, you MUST read the following reference specs before making or reviewing any changes:

- `references/components/causal_recurrence.md` — CausalRecurrenceLayer (RG-LRU): equations, init values, deviations, verification checklist
- `references/components/zone_ed_pipeline.md` — Zone E / Zone D full pipeline: encoder, router, concept tokens, EMA, plug-back, gated residual
- `references/components/boundary_router.md` — BoundaryRouter: cosine vs learned, position-0 rule, EMA clamp
- `references/components/mol_ffn.md` — MoL FFN (inner transformer): routing correctness, load balance
- `references/components/attention_rope_norms.md` — Attention, RoPE, RMSNorm

Read ALL of these files before proposing or reviewing any change to phase2/model.py or phase2/train.py. Do not rely on memory of prior reads — always re-read when the file has been modified.
