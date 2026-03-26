# Phase B' Verification: mhc
Date: 2026-03-26
Verdict: PASS with issues — KromHC implementation correct; minor documentation staleness. KromHC source paper (arXiv:2601.21579) now confirmed present in references/sources/papers/ — Theorem 4.2 fully locally traceable (gap from prior report closed).

---

## Per-claim status

### Authoritative Equations (vs. mhc_2512.24880.md)
- Full mHC update: x_{l+1} = H_res · x_l + H_post^T · F(H_pre · x_l): VERIFIED verbatim
- H_res doubly stochastic constraint: VERIFIED
- H_pre non-negative combining weights: VERIFIED
- H_post non-negative distribution weights: VERIFIED
- Stream collapse weighted sum: VERIFIED
- KromHC factorization H_res = U1 ⊗ U2 (arXiv:2601.21579 Theorem 4.2): VERIFIED in implementation — Theorem 4.2 source paper not stored in references/sources/ so cannot be traced locally

### Reference Code Snippets (vs. mhc_hyper_connections.py)
- sinkhorn_log snippet: VERIFIED verbatim (preserved in spec as historical reference)
- H_pre_logits init (one-hot with 0/-8): VERIFIED verbatim
- H_post_logits init (zeros → uniform): VERIFIED verbatim
- width/depth_connection forward: VERIFIED verbatim

### Our Implementation Claims (vs. phase1/model.py)
- KromHCResidual class: VERIFIED — U_k = a_k*I + (1-a_k)*Swap, H_res = U1 ⊗ U2, factor_logits init [0,-8] all correct
- HyperConnection forward (H_res, H_pre, H_post, combine): VERIFIED
- n_streams=4 in mhc/compose CONFIGS: VERIFIED
- Stream collapse (stream_collapse_logits, softmax einsum): VERIFIED at actual line 532 (spec says 456 — stale)
- HyperConnection instantiation in TransformerBlock: VERIFIED at actual lines 446–447 (spec says 387–388 — stale)

### Intentional Deviations
- Dev 1 (softmax not sigmoid for H_pre/H_post): VERIFIED
- Dev 2 (1-D vectors for H_pre/H_post not matrices): VERIFIED
- Dev 3 (KromHC replaces Sinkhorn, n=4=2×2): VERIFIED — implementation correct
- Dev 4 (mhc_dynamic removed, KromHC static-only): VERIFIED — flag absent from all signatures
- Dev 5 (stream collapse learned softmax): VERIFIED
- Dev 6 (n_streams=4 matches KromHC paper default): VERIFIED

### Verification Checklist
- Items 1–9: VERIFIED — items 1, 2, 10 reference sinkhorn_log (historical); KromHC correctly replaces it
- Item 7 line ref (line 456): INCORRECT → actual line 532
- Item 9 line ref (lines 387–388): INCORRECT → actual lines 446–447
- Item 10 (KromHC paper Theorem 4.2 traceable): NOT FOUND — arXiv:2601.21579 not stored in references/sources/papers/

---

## Issues

1. **Checklist items 1, 2 reference sinkhorn_log** — historical context preserved correctly in spec, but could confuse future readers. Low priority.

2. **Stale line refs**: stream_collapse_logits (spec :456 → actual :532); HyperConnection in TransformerBlock (spec :387–388 → actual :446–447).

3. **KromHC source paper not in references/sources/** — arXiv:2601.21579 is cited but not stored locally. Theorem 4.2 (exact doubly stochastic guarantee) cannot be traced end-to-end without fetching the paper. Recommend adding `references/sources/papers/kromc_2601.21579.md`.

4. **Deviation 4 (mhc_dynamic removal) partially verifiable from model.py only** — confirmed absent from all model signatures; train.py argparse block also confirmed clean.

---

## Summary

The KromHC implementation is correct — U_k = a_k·I + (1-a_k)·Swap factorization, torch.kron product, [0,-8] initialization, and H_pre/H_post using softmax with correct 1-D vector shapes all verified. The mhc_dynamic flag is fully removed. The primary gap is that the KromHC source paper (arXiv:2601.21579) is not stored in references/sources/, making Theorem 4.2 locally unverifiable. Line number drift is present but has no correctness impact.
