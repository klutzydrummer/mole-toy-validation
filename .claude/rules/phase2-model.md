---
paths:
  - "phase2/model.py"
  - "phase2/train.py"
  - "phase2/components/causal_recurrence.py"
  - "phase2/components/zone_e.py"
  - "phase2/components/zone_d.py"
  - "phase2/components/boundary_router.py"
---

When working with any Phase 2 file, you MUST read the relevant reference spec(s)
before making or reviewing any changes. Read the spec for the file(s) you are
touching — do NOT rely on memory of prior reads.

## Component → spec mapping

| File | Read before touching |
|------|---------------------|
| `causal_recurrence.py` | `references/components/causal_recurrence.md` |
| `zone_e.py` | `references/components/zone_ed_pipeline.md` AND `references/components/mol_ffn.md` |
| `zone_d.py` | `references/components/zone_ed_pipeline.md` |
| `boundary_router.py` | `references/components/boundary_router.md` |
| `model.py` | Specs for any component whose config or wiring is changing |
| `train.py` | Specs for any component the change touches |

## Critical invariants to check before any edit

- `causal_recurrence.md`: sqrt(1-a²) normalization, float32 sigmoid (not float16), log_a_init=3.0 (Zone E) / 0.0 (Zone D). **Never use default 7.5 in production** — causes 9.7 BPC plateau failure.
- `zone_ed_pipeline.md`: EMA over concept tokens (not raw encoder output), cumsum plug-back, gated residual, STE rounding in SimpleDecoder, smoke test item 10
- `boundary_router.md`: adjacent key k_{t-1} (not k_t), p_0=1.0 always (BOS is always a boundary), top-M selection, dummy slot to prevent slot-0 scatter collision
- `mol_ffn.md`: sigmoid routing, unbiased weight normalization, load-balance sign — applies to MoLFFN inside zone_e TransformerEncoder blocks

## After modifying any component file

The component's verification hash will be stale. Before pushing:
```bash
python utils/verify.py check          # shows which components are stale
# Tell Claude: 'run full verification for <component>'
python utils/verify.py update --result pass --report <report_path> <component_name>
```

`verify.py check` blocks cloud training until updated with a passing result.

**Note:** `zone_e.py` is tracked by both `mol_ffn` and `zone_ed_pipeline` components.
Changing it marks BOTH stale — verify both before pushing.
