# Phase B' Verification Report: mhc

**Date:** 2026-04-05
**Supersedes:** prior report dated 2026-04-02
**Component:** `mhc` — Manifold-Constrained Hyper-Connections + KromHC
**Spec:** `references/components/mhc.md`
**Implementation:** `phase1/components/mhc.py`, `phase1/components/transformer_block.py`, `phase1/model.py`
**Sources checked:** `references/sources/papers/mhc_2512.24880.md`, `references/sources/code/mhc_hyper_connections.py`

---

## Overall verdict: PASS with issues

All mathematical equations, reference code snippets, implementation behaviors, and intentional
deviations are correct. The code matches the spec. The three issues are documentation-only stale
line number pointers in the verification checklist — they reference the old monolithic
`phase1/model.py` from before the 2026-04-05 component extraction.

---

## Per-claim status

### Authoritative equations

| Claim | Status | Notes |
|-------|--------|-------|
| mHC update rule: x_{l+1} = H_res·x_l + H_post^T·F(H_pre·x_l) | VERIFIED | Exactly implemented in HyperConnection.forward |
| H_res doubly stochastic via KromHC Theorem 4.2 | VERIFIED | KromHCResidual uses torch.kron(U1, U2) |
| U_k = a_k·I + (1-a_k)·Swap, a_k = softmax([l0,l1])[0] | VERIFIED | mhc.py:62–65 |
| H_pre/H_post non-negative, sum to 1 via softmax | VERIFIED | F.softmax at mhc.py:123, 131 |

### Reference implementation cross-check

| Claim | Status | Notes |
|-------|--------|-------|
| H_pre near one-hot init (one 0.0, rest -8.0) | VERIFIED | mhc.py:101–103 matches tokenbender/mHC |
| H_post uniform init (zeros → softmax = 1/n) | VERIFIED | mhc.py:108 |
| factor_logits init [0, -8] → a≈1 → U≈I | VERIFIED | mhc.py:52–53 |
| softmax (not softplus) for H_pre/H_post | VERIFIED | F.softmax confirmed |

### Intentional deviations

| Deviation | Status | Notes |
|-----------|--------|-------|
| 1. KromHC replaces Sinkhorn-Knopp | VERIFIED | No sinkhorn_log anywhere in mhc.py |
| 2. Static H_res only (no dynamic/input-conditional) | VERIFIED | KromHCResidual.forward() takes no input |
| 3. n fixed at 4 (2×2 factorization) | VERIFIED | assert n == 4 at mhc.py:91 |
| 4. softmax not softplus for H_pre/H_post | VERIFIED | F.softmax confirmed |
| 5. H_post init zeros (uniform), not stream-index bias | VERIFIED | mhc.py:108: torch.zeros(n) |
| 6. Stream expand/collapse in ToyTransformer, not HyperConnection | VERIFIED | phase1/model.py:124, 130 |

### Verification checklist

| Item | Status | Notes |
|------|--------|-------|
| 1. H_res row/col sums ≈ 1 | VERIFIED | Kronecker product of two doubly stochastic matrices is doubly stochastic (Theorem 4.2) |
| 2. H_pre/H_post sum to 1 | VERIFIED | softmax output |
| 3. H_res near identity at init | VERIFIED | factor_logits=[0,-8] → a≈1 → U≈I → H_res=I⊗I≈I |
| 4. H_pre near one-hot at init | VERIFIED | one 0.0, rest -8.0 → softmax selects one stream |
| 5. H_post uniform at init | VERIFIED | zeros → softmax = [0.25, 0.25, 0.25, 0.25] |
| 6. branch_fn receives [B,L,d] not [B,L,n,d] | VERIFIED | branch_input = einsum("n,blnd->bld") at mhc.py:124 |
| 7. .clone() prevents in-place mutation | STALE POINTER | Behavior correct; checklist cites old model.py line. Correct: stream expansion at `phase1/model.py:124`; einsum produces new tensor so no .clone() needed |
| 8. KromHC n=4 assertion fires for wrong n | VERIFIED | assert n == 4 at mhc.py:91 |
| 9. Pre-norm applied before H_pre combine | STALE POINTER | Behavior correct; old citation "lines 702–703". Correct: `phase1/components/transformer_block.py:79–80` |
| 10. KromHCResidual present (not sinkhorn_log) | STALE POINTER | VERIFIED correct; old citation `phase1/model.py:353`. Correct: `phase1/components/mhc.py:31` |

---

## Stale pointers to fix in spec

| Checklist item | Old citation | Correct citation |
|----------------|-------------|-----------------|
| Item 7 | `phase1/model.py:798` | `phase1/model.py:124` |
| Item 9 | `phase1/model.py:702–703` | `phase1/components/transformer_block.py:79–80` |
| Item 10 | `phase1/model.py:353` | `phase1/components/mhc.py:31` |

---

## Summary

Implementation is mathematically correct. KromHC factorization is exactly doubly stochastic.
All 6 intentional deviations accurately described and confirmed. Three checklist line citations
are stale from the pre-extraction monolith — documentation only, no correctness impact.
