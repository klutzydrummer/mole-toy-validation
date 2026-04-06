# Phase B' Verification Report: zone_ed_pipeline

**Date:** 2026-04-06
**Supersedes:** report dated 2026-04-05
**Component:** `zone_ed_pipeline` — Zone E encoder + Zone D decoder + ratio loss
**Spec:** `references/components/zone_ed_pipeline.md`
**Implementation:** `phase2/components/zone_e.py`, `phase2/components/zone_d.py`
**Sources checked:** `references/sources/papers/hnet_2507.07955.md`, `references/sources/code/hnet_boundary.py`

**Change triggering this re-verification:**
`CRLEncoderFull` in `zone_e.py` had a redundant `norm_out = RMSNorm(d)` removed.
Each `CausalRecurrenceLayer` already ends with `return self.norm(self.out_proj(out))` (line 121
of `causal_recurrence.py`). `CRLEncoderFull` now returns `x` directly after the recurrence stack,
matching `CRLEncoder`'s structure. This re-verification confirms the removal is correct and does
not violate any spec claim.

---

## Overall verdict: PASS

All H-Net equations are correctly implemented. All reference code snippets match verbatim.
All intentional deviations are present and accurately described. The `norm_out` removal from
`CRLEncoderFull` is correct: `CausalRecurrenceLayer` provides its own output norm at line 121
of `causal_recurrence.py`, and the spec contains no claim requiring an extra outer norm on
`CRLEncoderFull`. `CRLEncoder` (bottlenecked) also has no outer norm, making both variants
consistent. Minor stale line number pointers noted below.

---

## Key finding: CRLEncoderFull norm_out removal is correct

**Claim being verified:** Was the removal of `norm_out = RMSNorm(d)` from `CRLEncoderFull` correct?

**Evidence:**

1. `causal_recurrence.py` line 121: `return self.norm(self.out_proj(out))` — each
   `CausalRecurrenceLayer` applies `RMSNorm` to its output unconditionally before returning.

2. `CRLEncoder` (bottlenecked variant, `zone_e.py` lines 19–40) has no outer norm either —
   after `self.up_proj(h)` it returns directly. Both variants are now structurally consistent.

3. `CRLEncoderFull` docstring (zone_e.py line 60): "No norm_out: each CausalRecurrenceLayer
   already ends with self.norm(out_proj(out)). CRLEncoder also has no extra outer norm —
   both variants are consistent." This accurately describes the implementation.

4. The spec (`zone_ed_pipeline.md`) describes Zone E encoders as: "All implement
   `forward(x: [B, L, d]) -> encoder_out [B, L, d]`." It makes no claim that encoders must
   apply an outer norm. The H-Net paper (hnet_2507.07955.md) places RMSNorm at the
   "end of each network component (ℰˢ, 𝒟ˢ, and 𝓜)" — since `CausalRecurrenceLayer`
   itself ends with RMSNorm, the encoder component as a whole satisfies this requirement.

**Verdict:** CORRECT. The removal introduces no spec violation and aligns both CRL
encoder variants.

---

## Per-claim status

### Authoritative equations (H-Net Eq. 1–10)

| Claim | Status | Notes |
|-------|--------|-------|
| Eq. 1 — main flow: x̂ = ℰ(x), ẑ = 𝓜(x̂), z = 𝒟(ẑ) | VERIFIED | OuterModel.forward (model.py:175–197); 𝓜 = identity in Phase 2 outer |
| Eq. 2 — chunking: (x^{s+1}, p^s) = chunk(x̂^s) | VERIFIED | BoundaryRouter produces M concept tokens + boundary_probs [B,L] |
| Eq. 3 — gated residual: z = dechunk(ẑ, p) + linear(x̂) | VERIFIED | zone_d.py:114; residual_proj zero-init at model.py:162 |
| Eq. 4 — boundary prob: p_t = (1/2)(1 - cos_sim(q_t, k_{t-1})) | VERIFIED | boundary_router.py; p_0=1.0 via F.pad |
| Eq. 5 — EMA: z̄_t = p_t·ẑ_t + (1-p_t)·z̄_{t-1} | VERIFIED | zone_d.py:64–84 via _parallel_scan; p_at_bounds.clamp(min=0.1) at line 73 |
| Eq. 6 — confidence: c_t = p_t if b_t=1, (1-p_t) if b_t=0 | VERIFIED | zone_d.py:104 |
| Eq. 7 — STE: ste(c_t) = c_t + stopgrad(1-c_t) = 1.0 in fwd | VERIFIED | zone_d.py:106 |
| Eq. 8 — plug-back: z̃_t = z̄_{cumsum(b)_t - 1} | VERIFIED | zone_d.py:92–96 via cumsum-1 gather |
| Eq. 9 — upsampler: ste(c_t) · z̃_t | VERIFIED | zone_d.py:109 |
| Eq. 10 — ratio loss: (N/(N-1))·((N-1)·F·G + (1-F)·(1-G)) | VERIFIED | phase2/train.py:95–108; F=hard fraction, G=soft average |

### Reference code snippets

| Claim | Status | Notes |
|-------|--------|-------|
| W_q/W_k identity init (hnet_boundary.py lines 79–83) | VERIFIED | Snippet in spec matches verbatim; implementation re-applies eye_ at model.py:157–159 |
| Boundary probability computation (hnet_boundary.py lines 113–123) | VERIFIED | Snippet matches verbatim; boundary_router.py implements same logic |
| Plug-back cumsum+gather (hnet_boundary.py lines 337–342) | VERIFIED | Snippet matches verbatim; zone_d.py:92–96 matches exactly |

### Our implementation — Zone E (zone_e.py)

| Claim | Status | Notes |
|-------|--------|-------|
| `CRLEncoder` lines 20–41 (spec) | STALE — minor | Class starts line 19, ends line 40. Off by 1. Content correct. |
| `CRLEncoderFull` lines 44–67 (spec) | STALE — minor | Class starts line 43. Content correct. |
| `TransformerEncoder` lines 70–88 | VERIFIED | Exact match |
| `DiffAttnEncoder` lines 91–110 | VERIFIED | Exact match |
| `MLAEncoder` lines 113–132 | VERIFIED | Exact match |
| `IdentityEncoder` lines 135–143 | VERIFIED | Exact match |
| All encoders: forward(x:[B,L,d]) → [B,L,d] | VERIFIED | All six classes satisfy this contract |
| CRLEncoder: down_proj → 3×CRL(d//4, log_a_init=3.0) → up_proj | VERIFIED | zone_e.py:29–40 |
| CRLEncoderFull: 3×CRL(d, log_a_init=3.0), no bottleneck | VERIFIED | zone_e.py:57–67; no norm_out (confirmed correct — see key finding above) |
| Transformer variants: 4 TransformerBlock layers + norm_out | VERIFIED | TransformerEncoder/DiffAttnEncoder/MLAEncoder each have self.norm_out = RMSNorm(d) and call it in forward |
| IdentityEncoder: returns x unchanged | VERIFIED | zone_e.py:141–143 |

### Our implementation — Zone D (zone_d.py)

| Claim | Status | Notes |
|-------|--------|-------|
| `SimpleDecoder.forward` lines 52–116 (spec) | VERIFIED | Accurate |
| Step 1: EMA smoothing lines 70–84 (spec) | STALE — minor | EMA block starts at line 64 (not 70); ends at line 84 correct. The comment block before line 70 is part of the EMA setup. Confirmed from prior report. |
| p_at_bounds.clamp(min=0.1) at line 73 | VERIFIED | Exact match |
| Step 2: plug-back lines 91–96 | VERIFIED | Exact match |
| Step 3: confidence + STE lines 103–109 | VERIFIED | All three formulas (Eq. 6/7/9) at the cited lines |
| Step 4: residual line 114 | VERIFIED | `out = upsampled + self.residual_proj(encoder_out)` |

### Our implementation — model.py / train.py

| Claim | Status | Notes |
|-------|--------|-------|
| OuterModel.forward lines 174–196 (spec) | STALE — minor | Forward is lines 175–197. Off by 1. Content verified correct. |
| W_q/W_k eye_ re-init after _init_weights, lines 156–158 (spec) | STALE — minor | Actual lines 157–159. Off by 1. Content correct. |
| residual_proj zero init line 161 (spec) | STALE — minor | Actual line 162. Off by 1. Content correct. |
| ratio_loss() in train.py lines 95–108 | VERIFIED | Exact match to H-Net Eq. 10 |
| alpha=0.03, outer_crl uses alpha=0 | VERIFIED | Confirmed in train.py |

### Intentional deviations

| Deviation | Status | Notes |
|-----------|--------|-------|
| Griffin RG-LRU (CRL) instead of Mamba-2 for encoder | VERIFIED | Present in CRLEncoder and CRLEncoderFull |
| No width expansion in encoder | VERIFIED | d stays constant throughout Zone E |
| W_q/W_k re-init via nn.init.eye_ (not _no_reinit flag) | VERIFIED | model.py:157–159; achieves same outcome |
| p_at_bounds.clamp(min=0.1) vs H-Net [1e-4, 1-1e-4] | VERIFIED | zone_d.py:73; stronger guard accurately described in spec |

### Verification checklist

| Item | Status | Notes |
|------|--------|-------|
| 1. Position-0 boundary for cosine_rule/learned_e2e | VERIFIED | boundary_router.py; F.pad(p, (1,0), value=1.0) |
| 2. Variable M per sequence | VERIFIED | BoundaryRouter: threshold p≥0.5, variable M, padded to M_max |
| 3. EMA initialized at h0 = concept_tokens[:,0] | VERIFIED | zone_d.py:75: h0 = concept_tokens[:,0] |
| 4. p_at_bounds.clamp(min=0.1) prevents EMA collapse | VERIFIED | zone_d.py:73 |
| 5. Plug-back correctness via cumsum | VERIFIED | cumsum-1 gather at zone_d.py:92–96 |
| 6. STE is identity in forward pass | VERIFIED | zone_d.py:106: c_t + (1-c_t).detach() = 1.0 in fwd |
| 7. residual_proj weight=0 at init | VERIFIED | model.py:162: nn.init.zeros_(self.decoder.residual_proj.weight) |
| 8. W_q/W_k identity init survives _init_weights | VERIFIED | model.py:157–159: eye_ re-applied after self.apply(_init_weights) |
| 9. Ratio loss minimum at target | VERIFIED | train.py:95–108; formula analytically gives 1.0 at F=G=1/N |
| 10. EMA collapse detection — MANDATORY PRE-FLIGHT CHECK | VERIFIED | utils/smoke_test.py enforced as hard gate in run_experiments.sh |
| 11. Ratio loss drives compression toward target_rate | VERIFIED (structure) | Code path confirmed; runtime convergence requires cloud training |
| 12. boundary_entropy decreases over training | VERIFIED (structure) | Logged as hdc/boundary_entropy in train.py; runtime requires cloud |

---

## Stale line number summary

All stale pointers are off-by-one shifts with no correctness impact:

| Spec claim | Actual location | Impact |
|------------|-----------------|--------|
| CRLEncoder "lines 20–41" | lines 19–40 | None — content verified correct |
| CRLEncoderFull "lines 44–67" | lines 43–67 | None — content verified correct |
| EMA block start "line 70" | line 64 | None — documented in prior report |
| OuterModel.forward "lines 174–196" | lines 175–197 | None — content verified correct |
| W_q/W_k re-init "lines 156–158" | lines 157–159 | None — content verified correct |
| residual_proj zero init "line 161" | line 162 | None — content verified correct |

---

## Summary

Zone E / Zone D pipeline is correctly implemented. All 10 H-Net equations verified. All
reference code snippets match verbatim. All intentional deviations are present and accurately
described. The `CRLEncoderFull` `norm_out` removal is confirmed correct: each `CausalRecurrenceLayer`
provides its own output RMSNorm internally (causal_recurrence.py:121), the spec imposes no
outer-norm requirement on encoders, and both CRL variants are now structurally consistent.
Six minor off-by-one stale line number pointers noted; none affect correctness.
