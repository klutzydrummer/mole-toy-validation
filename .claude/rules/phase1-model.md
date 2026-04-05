---
paths:
  - "phase1/model.py"
  - "phase1/train.py"
  - "phase1/components/_shared.py"
  - "phase1/components/attention_rope_norms.py"
  - "phase1/components/mla_attention.py"
  - "phase1/components/diff_attention.py"
  - "phase1/components/mhc.py"
  - "phase1/components/mol_ffn.py"
  - "phase1/components/transformer_block.py"
---

When working with any Phase 1 file, you MUST read the relevant reference spec(s)
before making or reviewing any changes. Read the spec for the file(s) you are
touching — do NOT rely on memory of prior reads.

## Component → spec mapping

| File | Read before touching |
|------|---------------------|
| `_shared.py` | `attention_rope_norms.md` (RMSNorm, RoPE, SwiGLU live here) |
| `attention_rope_norms.py` | `references/components/attention_rope_norms.md` |
| `mla_attention.py` | `references/components/mla_attention.md` |
| `diff_attention.py` | `references/components/diff_attention.md` |
| `mhc.py` | `references/components/mhc.md` |
| `mol_ffn.py` | `references/components/mol_ffn.md` |
| `transformer_block.py` | No spec (wiring only) — but read specs for any component it calls |
| `model.py` | All specs relevant to configs being changed |
| `train.py` | Specs for any component the change touches |

## Critical invariants to check before any edit

- `mhc.md`: H_res/H_pre/H_post init, KromHC Kronecker factorization for H_res (not Sinkhorn — replaced by arXiv:2601.21579), softmax (not softplus) for H_pre/H_post
- `mol_ffn.md`: sigmoid routing scores, unbiased weight normalization, load-balance loss sign
- `attention_rope_norms.md`: RoPE rotation pairs, RMSNorm formula, SwiGLU gate
- `mla_attention.md`: shared KV latent (d_c=d//4), separate Q latent (d_c_q=d//2), RoPE on full d_head, no RMSNorm on latents
- `diff_attention.md`: doubled Q heads, GQA pairing via repeat_interleave (not blocked split), sigmoid λ (not exp), no per-head RMSNorm, W_lambda bias=True

## After modifying any component file

The component's verification hash will be stale. Before pushing:
```bash
python utils/verify.py check          # shows which components are stale
# Tell Claude: 'run full verification for <component>'
python utils/verify.py update --result pass --report <report_path> <component_name>
```

`verify.py check` blocks cloud training until updated with a passing result.
