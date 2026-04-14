# phase1/components/

Each file in this directory is a **verifiable component** — it corresponds to an authoritative spec in `references/components/` and is individually hashed by `utils/verify.py`.

## Component → spec mapping

| File | Spec | Contains |
|------|------|----------|
| `_shared.py` | (dependency only) | RMSNorm, precompute_rope, apply_rope, SwiGLU |
| `attention_rope_norms.py` | `references/components/attention_rope_norms.md` | CausalSelfAttention |
| `mla_attention.py` | `references/components/mla_attention.md` | MLACausalAttention |
| `diff_attention.py` | `references/components/diff_attention.md` | DifferentialCausalAttention, DiffMLAAttention |
| `mhc.py` | `references/components/mhc.md` | GoMHCResidual, HyperConnection |
| `mol_ffn.py` | `references/components/mol_ffn.md` | LoRAAdapter, SingleLoRAFFN, MoLFFN |
| `transformer_block.py` | (wiring only — no spec) | TransformerBlock |

## Before modifying any file here

1. **Read the component spec** for that file (see table above).
2. **Read the primary sources** cited in the spec's "Sources" table.
3. Fix root causes — never suppress errors or work around spec requirements.

## After modifying any file here

The component's hash will be stale. Before pushing:

```bash
# Tell Claude: 'run full verification' for the changed component, or:
python utils/verify.py update --result pass --report <report_path> <component_name>
```

`verify.py check` will block cloud training until the hash is updated with a passing result.

## Import rules

Components may import from:
- `torch`, `torch.nn`, `torch.nn.functional`, `math` — always fine
- `phase1.components._shared` — shared primitives (RMSNorm, RoPE, SwiGLU)
- `phase1.components.attention_rope_norms` — CausalSelfAttention
- Other `phase1.components.*` files — for inter-component dependencies

Components must NOT import from:
- `phase1.model` — circular (model.py is a facade that imports from here)
- `phase2.*` — phase1 components are phase2-agnostic

## What belongs here vs. in model.py

- **Here:** mathematical implementations — attention mechanisms, FFN variants, routing logic, recurrence.
- **In `phase1/model.py`:** `ToyTransformer`, `CONFIGS`, the `forward` wiring. No math.
- **In `transformer_block.py`:** `TransformerBlock` — the per-layer combinator. Wiring, but shared across phases so it lives here.
