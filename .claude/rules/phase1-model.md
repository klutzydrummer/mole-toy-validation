---
paths:
  - "phase1/model.py"
  - "phase1/train.py"
---

When working with any Phase 1 file, you MUST read the following reference specs before making or reviewing any changes:

- `references/components/mhc.md` — mHC (Hyper-Connections): sinkhorn, H_res/H_pre/H_post init, activation correctness
- `references/components/mol_ffn.md` — MoL FFN: routing equations, load balance sign, unbiased weights
- `references/components/attention_rope_norms.md` — Attention, RoPE, RMSNorm

Read ALL of these files before proposing or reviewing any change to phase1/model.py or phase1/train.py. Do not rely on memory of prior reads — always re-read when the file has been modified.
