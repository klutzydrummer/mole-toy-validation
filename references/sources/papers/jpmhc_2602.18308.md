# JPmHC: Dynamical Isometry via Orthogonal Hyper-Connections

**arXiv:** 2602.18308
**Published:** February 2026
**Authors:** Biswa Sengupta, Jinhua Wang, Leo Brunswic
**HTML:** https://arxiv.org/abs/2602.18308

---

## Summary

Replaces mHC's doubly-stochastic H_res with a **Stiefel-manifold-constrained orthogonal
mixer** (not doubly stochastic, but norm-preserving). Uses Cayley transform to enforce
orthogonality, ensuring `||H_res||₂ = 1` exactly and thus dynamical isometry of the
stream-mixing Jacobian.

Key difference from go-mHC:
- go-mHC (arXiv:2604.02309): H_res ∈ Birkhoff polytope (doubly stochastic, non-negative)
- JPmHC: H_res ∈ O(n) or SO(n) (orthogonal group)

Both use Cayley transform. JPmHC generalizes to the full orthogonal group (allows signed
entries), go-mHC constrains to the doubly-stochastic subset via block Frobenius projection.

---

## Dynamical isometry argument

For a network with Jacobian J = ∂x_L/∂x_0, dynamical isometry requires all singular values
of J to be ≈ 1. Orthogonal H_res provides this: `||H_res x||₂ = ||x||₂` for any x, so
each stream-mixing step is norm-preserving.

Doubly-stochastic H_res (mHC/go-mHC) satisfies `||H_res x||₁ = ||x||₁` but NOT
necessarily `||H_res x||₂ = ||x||₂` — so DS mixing can still contract or expand L2 norms.

---

## Relevance to nGPT + mHC

For `ngpt_mhc_a` (multi-sphere), H_res mixing leaves the sphere. We renorm after mixing:
```
mixed_i = Σ_j H_res[i,j] * x_j    # weighted sum, leaves sphere
mixed_i = L2Norm(mixed_i)          # project back
```

JPmHC's orthogonal mixer would interact differently with this renorm:
- Orthogonal H_res is norm-preserving in L2: `||mixed_i||₂ = ||x_i||₂ = 1`
  IF all streams have equal weight (orthogonality doesn't preserve per-component norms)
- In practice, post-mixing L2Norm is still needed for individual stream components

**Potential advantage for multi-sphere:** Orthogonal (JPmHC-style) H_res, combined with
per-stream L2Norm renorm, might give better gradient flow than DS-constrained go-mHC,
since the orthogonal group is the natural symmetry group of S^{d-1}.

---

## Status in this codebase

Not implemented. Included as a reference for the nGPT+mHC composition design space.
If `ngpt_mhc_a` underperforms, replacing go-mHC's DS constraint with an orthogonal
(Stiefel) constraint may be worth exploring as a follow-up.

Tested on: ARC-AGI benchmark. Achieves faster convergence, higher accuracy, lower
computational cost vs bistochastic (mHC) baselines.
