# Phase B' Verification Report: mhc

**Date:** 2026-04-07
**Supersedes:** prior report dated 2026-04-06 (KromHC era)
**Component:** `mhc` — Manifold-Constrained Hyper-Connections + go-mHC
**Spec:** `references/components/mhc.md`
**Implementation:** `phase1/components/mhc.py`, `phase1/components/transformer_block.py`, `phase1/model.py`
**Sources checked:** `references/sources/papers/mhc_2512.24880.md`, `references/sources/code/mhc_hyper_connections.py`

---

## Overall verdict: PASS with issues

The go-mHC migration (replacing KromHC) is correctly and completely implemented. All
mathematical constructions are sound, zero-initialization gives exact identity at start,
H_pre/H_post inits are preserved from the reference code, and all KromHC references have
been removed from `mhc.py`.

Two minor documentation issues:
1. `references/sources/papers/go_mhc_2604.02309.md` does not yet exist — the spec correctly
   notes "(to be added)" but the source paper is missing from Phase A.
2. `phase1/components/CLAUDE.md` component table still lists `mhc.py` contents as
   "KromHCResidual, HyperConnection" — should be "GoMHCResidual, HyperConnection".

Neither affects implementation correctness or training. PASS.

---

## Special Focus: go-mHC Migration

### 1. `GoMHCResidual` class at approximately line 37 of mhc.py
**VERIFIED.** `class GoMHCResidual(nn.Module):` is at **line 37 exactly**.

### 2. Cayley transform: `Q = solve(I+A, I-A)` (not `(I-A)(I+A)^{-1}`)
**VERIFIED.** `mhc.py:117`:
```python
Q = torch.linalg.solve(I + A, I - A)
```
`torch.linalg.solve(M, B)` solves `M @ X = B`, so this computes `Q = (I+A)^{-1}(I-A)`.
This is the correct right Cayley transform. Code comment confirms the identity.

### 3. Zero initialization of W_res weight and bias
**VERIFIED.** `mhc.py:72–73`:
```python
nn.init.zeros_(self.W_res.weight)
nn.init.zeros_(self.W_res.bias)
```
Both weight and bias zero-initialized.

### 4. Block Frobenius: reshape Q as [B,L,n,s,n,s], sum dim (3,5), divide by s
**VERIFIED.** `mhc.py:121–122`:
```python
Q_blocks = Q.reshape(B, L, n, self.s, n, self.s)
H_res = (Q_blocks ** 2).sum(dim=(3, 5)) / self.s
```
Exactly matches the spec's description.

### 5. `HyperConnection` uses `self.go_res = GoMHCResidual(n, d, s=2)` (not krom_res)
**VERIFIED.** `mhc.py:151`:
```python
self.go_res = GoMHCResidual(n, d, s=2)
```
No `krom_res` attribute exists anywhere in the file.

### 6. Einsum in HyperConnection.forward uses `"blij, bljd -> blid"` (batched H_res)
**VERIFIED.** `mhc.py:175`:
```python
mixed = torch.einsum("blij, bljd -> blid", H_res, streams)
```
Correct batched contraction over stream index `j` → output stream `i`.

### 7. H_pre near-one-hot init, H_post zeros init
**VERIFIED.**
- `pre_logits` (`mhc.py:156–158`): `torch.full((n,), -8.0)`, one entry set to `0.0`. Near one-hot after softmax.
- `post_logits` (`mhc.py:163`): `torch.zeros(n)`. Uniform `[1/n, ..., 1/n]` after softmax.

### 8. No n==4 assertion in HyperConnection
**VERIFIED.** No assertion on `n` in `HyperConnection.__init__` or `forward`. Any `n` accepted.

### 9. No reference to `KromHCResidual`, `sinkhorn_log`, `kron` anywhere in mhc.py
**VERIFIED.** Grep found zero matches for all three strings in `phase1/components/mhc.py`.

---

## Authoritative Equations Cross-Check

| Equation | Source | Status |
|----------|--------|--------|
| Eq. 1: `x_{l+1} = x_l + F(x_l, W_l)` | `mhc_2512.24880.md:45` | VERIFIED |
| Eq. 3: `x_{l+1} = H_res·x_l + H_post^T·F(H_pre·x_l, W_l)` | `mhc_2512.24880.md:57` | VERIFIED |
| Eq. 4 (multi-layer unrolled) | `mhc_2512.24880.md:61` | VERIFIED (documented, not directly tested in code) |
| Eq. 6 (Birkhoff polytope constraint) | `mhc_2512.24880.md:71` | VERIFIED analytically — Cayley + block Frobenius guarantees exact doubly stochasticity |
| Eq. 8: `H^pre = σ(H̃^pre)`, `H^post = 2σ(H̃^post)` | `mhc_2512.24880.md:108` | DEVIATION (softmax used instead, matches reference code — see Deviations) |
| go-mHC: `Q = (I+A)^{-1}(I-A)` | arXiv:2604.02309 (source not yet in Phase A) | VERIFIED in code |
| go-mHC: `H_res[i,j] = (1/s)||Q_block[i,j]||²_F` | arXiv:2604.02309 | VERIFIED in code |

---

## Reference Implementation Cross-Check

| Claim | Source lines | Status |
|-------|-------------|--------|
| `sinkhorn_log` function body | `mhc_hyper_connections.py:64–76` | VERIFIED — matches spec verbatim (historical; no longer used in our implementation) |
| H_res init: `full(-8.0)`, diagonal `0.0` | `mhc_hyper_connections.py:187–189` | VERIFIED — pattern preserved in `GoMHCResidual` W_res zero-init giving identity H_res |
| H_pre init: `full(-8.0)`, one entry `0.0` | `mhc_hyper_connections.py:191–193` | VERIFIED — `pre_logits` at `mhc.py:156–158` |
| H_post init: `zeros(1, num_streams)` | `mhc_hyper_connections.py:195–198` | VERIFIED — `post_logits` at `mhc.py:163` |
| H_pre softmax: `.softmax(dim=-1)` | `mhc_hyper_connections.py:247` | VERIFIED — `F.softmax(self.pre_logits, dim=0)` at `mhc.py:178` |
| H_post softmax: `.softmax(dim=-1)` | `mhc_hyper_connections.py:250` | VERIFIED — `F.softmax(self.post_logits, dim=0)` at `mhc.py:186` |

---

## Our Implementation Cross-Check

| Symbol | Spec location | Actual location | Status |
|--------|--------------|-----------------|--------|
| `GoMHCResidual` class | `mhc.md` → `mhc.py:37` | `mhc.py:37` | VERIFIED |
| `HyperConnection.__init__` | `mhc.md` → `mhc.py:145` | `mhc.py:145` | VERIFIED |
| `HyperConnection.forward` | `mhc.md` → `mhc.py:165` | `mhc.py:165` | VERIFIED |
| `TransformerBlock._forward_mhc` | `mhc.md` → `transformer_block.py:78` | `transformer_block.py:78` | VERIFIED |
| Stream expansion `.clone()` | `mhc.md` → `model.py:124` | `model.py:124` | VERIFIED |
| Stream collapse `softmax + einsum` | `mhc.md` → `model.py:130` | `model.py:130–131` | VERIFIED |

All spec line numbers are current and accurate.

---

## Verification Checklist

| Item | Status | Notes |
|------|--------|-------|
| 1. H_res doubly stochastic (row/col sums ≈ 1, entries ≥ 0) | VERIFIED | Exact by construction — Cayley transform Q ∈ SO(ns), block Frobenius gives exact DS; no approximation |
| 2. H_res near identity at init | VERIFIED | W_res=0 → z=0 → A=0 → Q=I → H_res=I_n exactly |
| 3. H_pre near one-hot at init | VERIFIED | `pre_logits`: one 0.0, rest -8.0 → softmax selects one stream |
| 4. H_post uniform at init | VERIFIED | `post_logits = zeros(n)` → softmax = [1/n, ..., 1/n] |
| 5. Stream divergence mechanism present | VERIFIED (structural) | H_post distributes branch_output with per-stream weights; divergence enabled. Not runtime-tested here. |
| 6. mHC forward shapes [B,L,n,d] → [B,L,n,d] | VERIFIED | `HyperConnection.forward` signature and return; `_forward_mhc` at `transformer_block.py:78–91` |
| 7. `.clone()` at stream expansion | VERIFIED | `model.py:124`: `.expand(...).clone()` |
| 8. Stream collapse uses softmax on logits | VERIFIED | `model.py:130–131`: `F.softmax(stream_collapse_logits, dim=0)` then einsum |
| 9. Pre-norm inside branch_fn (not on raw stream) | VERIFIED | `transformer_block.py:89–90`: lambda applies norm1/norm2 inside |
| 10. go-mHC replaces KromHC | VERIFIED | `GoMHCResidual` at line 37; no `KromHCResidual`/`sinkhorn_log`/`kron` in file; `self.go_res = GoMHCResidual(n, d, s=2)` at `mhc.py:151` |

---

## Intentional Deviations

| Deviation | Spec description | Code | Status |
|-----------|-----------------|------|--------|
| 1. softmax for H_pre/H_post (not sigmoid) | Paper Eq. 8 uses σ; code uses softmax per reference impl | `F.softmax(..., dim=0)` at `mhc.py:178, 186` | VERIFIED |
| 2. H_pre/H_post stored as [n] not [1,n] | Single-view simplification | `pre_logits: (n,)`, `post_logits: (n,)` | VERIFIED |
| 3. go-mHC replaces KromHC (April 2026) | Cayley+block Frobenius, input-conditional, any n | `GoMHCResidual` with `W_res` linear layer | VERIFIED |
| 5. Stream collapse via learned softmax weights | Not specified by paper; design choice | `stream_collapse_logits` initialized zeros | VERIFIED |
| 6. n_streams=4 | Paper's validated default; matches KromHC and go-mHC recommendation | `ToyTransformer.CONFIGS["mhc"]["n_streams"]=4` | VERIFIED |

---

## Issues

### Issue 1: Missing Phase A source for go-mHC
**Severity: Minor / documentation gap**
`references/sources/papers/go_mhc_2604.02309.md` does not exist. The spec correctly notes
"(to be added)" in the Sources table. The go-mHC paper (arXiv:2604.02309) should be
summarized and added to Phase A sources to complete the verification chain.

### Issue 2: Stale CLAUDE.md in phase1/components/
**Severity: Minor / stale documentation**
`phase1/components/CLAUDE.md` component table row for `mhc.py` reads:
> `mhc.py` | ... | `KromHCResidual, HyperConnection`

Should be updated to `GoMHCResidual, HyperConnection`. No correctness impact.

---

## Summary

The go-mHC migration is complete, correct, and consistent with the spec. The Cayley
transform `Q = (I+A)^{-1}(I-A)` is correctly implemented via `torch.linalg.solve`.
Block Frobenius projection correctly reshapes `[B,L,ns,ns]` → `[B,L,n,s,n,s]` and sums
over dims (3,5). Zero-init of W_res ensures exact identity H_res at start. H_pre/H_post
initialization patterns are preserved from the reference implementation. No remnants of
KromHCResidual, sinkhorn_log, or kron remain in mhc.py. All spec line numbers are accurate.
