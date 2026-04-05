# phase2/components/

Each file in this directory is a **verifiable component** — it corresponds to an authoritative spec in `references/components/` and is individually hashed by `utils/verify.py`.

## Component → spec mapping

| File | Spec | Contains |
|------|------|----------|
| `causal_recurrence.py` | `references/components/causal_recurrence.md` | _parallel_scan, CausalRecurrenceLayer |
| `zone_e.py` | `references/components/zone_ed_pipeline.md` + `mol_ffn.md` | CRLEncoder, CRLEncoderFull, TransformerEncoder, DiffAttnEncoder, MLAEncoder, IdentityEncoder |
| `boundary_router.py` | `references/components/boundary_router.md` | BoundaryRouter |
| `zone_d.py` | `references/components/zone_ed_pipeline.md` | SimpleDecoder |

**Note:** `zone_e.py` is tracked by two specs — `zone_ed_pipeline.md` and `mol_ffn.md`. A change here will flag both components stale and require verifying both.

## Before modifying any file here

1. **Read the component spec(s)** for that file (see table above). If the file maps to two specs, read both.
2. **Read the primary sources** cited in the spec's "Sources" table.
3. Changing `causal_recurrence.py` is especially high-risk — the `_parallel_scan` is a mathematically faithful Griffin RG-LRU implementation. Verify the sqrt(1-a²) normalization term before any edit.

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
- `phase1.components._shared` — RMSNorm, RoPE, SwiGLU
- `phase1.components.attention_rope_norms` — CausalSelfAttention
- `phase1.components.transformer_block` — TransformerBlock
- Other `phase2.components.*` files — for inter-component dependencies

Components must NOT import from:
- `phase2.model` — circular (model.py is a facade that imports from here)
- `phase1.model` — use `phase1.components.*` directly

## What belongs here vs. in model.py

- **Here:** mathematical implementations — recurrence cells, encoder architectures, routing logic, decoder reconstruction.
- **In `phase2/model.py`:** `OuterModel`, `CONFIGS`, the `forward` wiring. No math.
- `log_a_init` defaults: Zone E encoders use `log_a_init=3.0`, Zone D uses `log_a_init=0.0`. Never use the `CausalRecurrenceLayer` default of 7.5 in production — it causes the 9.7 BPC plateau failure (see project memory).
