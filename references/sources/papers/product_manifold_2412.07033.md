# Product Manifold Machine Learning for Physics

**arXiv:** 2412.07033
**Published:** December 2024
**Authors:** (University affiliation, particle physics application)
**HTML:** https://arxiv.org/abs/2412.07033

---

## Relevance to this codebase

Provides the closest published precedent for the **multi-sphere (Option A)** composition
implemented in `ngpt_mhc_a`. Key contribution: explicit treatment of multi-head attention
operations on **product of Riemannian manifolds** in tangent space.

---

## Core concepts

**Product manifold:** Cartesian product of constant-curvature Riemannian manifolds.
For n spheres: `M = S^{d-1} × S^{d-1} × ... × S^{d-1}` (n times)

Each "stream" or "head" lives on its own copy of S^{d-1}, and operations are performed
independently on each factor, with the product structure enabling cross-manifold interactions.

**PM-Transformer:** Multi-head attention where each head operates on its factor manifold.
Attention scores computed in tangent space at the current point on the manifold.
Mixing between factors happens via learned "transport" operations.

**Tangent space operations:**
- Geodesic interpolation on sphere: SLERP (spherical linear interpolation)
- For small steps: LERP ≈ SLERP (same approximation nGPT uses)
- Exponential map: from tangent space back to manifold
- Logarithmic map: from manifold to tangent space at base point

---

## Connection to ngpt_mhc_a (Option A)

Our implementation uses LERP + L2Norm (Fréchet mean approximation) for H_res mixing:
```python
mixed = einsum(H_res, x)    # weighted sum of unit vectors — leaves sphere
mixed = l2_norm(mixed)      # project back — approximates Fréchet mean
```

This is exactly the LERP + retraction pattern used in PM-Transformer, where:
- The LERP step = H_res linear mixing
- The retraction (projection) = L2Norm renorm
- Fréchet mean on S^{d-1} is approached iteratively but LERP+retract is a first-order approx

**When is the approximation tight?**
The Fréchet mean LERP approximation is most accurate when:
1. Mixing weights are near-uniform (go-mHC's Cayley init gives identity → uniform at start)
2. Stream vectors are not antipodal (handled by the L2Norm stability of training)

---

## Key gap from existing literature

The PM-Transformer paper focuses on product of manifolds with **different curvatures**
(spherical, hyperbolic, Euclidean factors) for physics applications. It does NOT address:
- Doubly-stochastic mixing constraints between sphere factors (our H_res constraint)
- Composition with nGPT's per-dim alpha eigenlearning rates
- Language modeling applications

Our `ngpt_mhc_a` is therefore a novel composition with no direct published precedent:
it combines nGPT's per-stream S^{d-1} constraint, go-mHC's DS-constrained mixing,
and per-stream alpha interpolation rates.

---

## Related: Fréchet mean on spheres (arXiv:2109.13512)

Neural Networks in Fréchet Spaces (arXiv:2109.13512) provides theoretical grounding for
differentiable computation through Fréchet means on arbitrary Riemannian manifolds.
The LERP+retract approximation used in `ngpt_mhc_a` is justified there as a first-order
approximation to the Fréchet mean that is differentiable and stable.
