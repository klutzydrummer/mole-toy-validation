---
paths:
  - "references/components/**"
---

When editing any file in `references/components/`, cross-check the corresponding
implementation before finalizing the change. Any factual claim (equations, init
values, expected ranges) must be verified against BOTH the implementation AND the
primary source paper cited in that spec. If spec and implementation disagree,
document the deviation explicitly in the spec's "Intentional deviations" section.

## Spec → implementation mapping

| Spec file | Implementation file(s) |
|-----------|------------------------|
| `attention_rope_norms.md` | `phase1/components/attention_rope_norms.py`, `phase1/components/_shared.py` |
| `mla_attention.md` | `phase1/components/mla_attention.py` |
| `diff_attention.md` | `phase1/components/diff_attention.py` |
| `mhc.md` | `phase1/components/mhc.py` |
| `mol_ffn.md` | `phase1/components/mol_ffn.py`, `phase2/components/zone_e.py` |
| `causal_recurrence.md` | `phase2/components/causal_recurrence.py` |
| `zone_ed_pipeline.md` | `phase2/components/zone_e.py`, `phase2/components/zone_d.py` |
| `boundary_router.md` | `phase2/components/boundary_router.py` |

## After editing a spec

If the edit changes a claim about the implementation (equation, init value, behavior):
1. Verify the implementation file still matches the updated spec.
2. If it does not, fix the implementation too — specs and code must agree.
3. The component's verification hash will be stale after any implementation change.
   Run: `python utils/verify.py check` to see which components need re-verification.
