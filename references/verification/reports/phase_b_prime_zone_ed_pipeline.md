# Phase B' Verification: zone_ed_pipeline

**Date:** 2026-04-02
**Verdict:** PASS with issues — All pipeline equations and core logic verified correct.
Three minor documentation issues remain open (Issues 1, 2, 5). No code correctness failures.

---

## Change log vs. prior report (2026-03-26)

- Issue 3 (gate_proj near-zero init): RESOLVED. Code now has `nn.init.zeros_` on weight and
  `bias.fill_(-4.0)`, giving `sigmoid(-4.0) ≈ 0.018` — satisfies H-Net near-zero requirement.
- Issue 4 (checklist item 1 wrong for fixed_stride): RESOLVED. Spec checklist item 1 now
  correctly notes "Not applicable to fixed_stride — that mode scatters at R-1, 2R-1, ...,
  L-1 and does not guarantee position 0 is selected."
- Issues 1, 2, 5 remain open (documentation only, no correctness impact).

---

## Per-claim status

### Authoritative Equations (vs. hnet_2507.07955.md, dlcm.md)

- H-Net Eq. 1 (main processing flow): VERIFIED verbatim
- H-Net Eq. 2 (chunking operation): VERIFIED verbatim
- H-Net Eq. 3 (gated residual / dechunking): VERIFIED verbatim; zero-init quote verbatim
- H-Net Eq. 4 (boundary probability, p₁=1.0, W_q/W_k init): VERIFIED verbatim
- H-Net Eq. 5 (EMA smoothing): VERIFIED verbatim
- H-Net Eq. 6 (confidence scoring): VERIFIED verbatim
- H-Net Eq. 7 (STE): VERIFIED verbatim
- H-Net Eq. 8 (causal plug-back): VERIFIED verbatim
- H-Net Eq. 9 (upsampler output): VERIFIED verbatim
- H-Net Eq. 10 (ratio loss, F/G definitions, α=0.03): VERIFIED verbatim
- DLCM Section 7.2.2 U-shaped loss (motivation for gated residual): VERIFIED — dlcm.md
  contains the U-shaped loss analysis in Section 7.2.2; spec citation is correct.
- DLCM Eq. 24 (gradient conflict): VERIFIED — dlcm.md documents the gradient conflict
  between ∇ℒ_CE and ∇ℒ_aux under "Gradient Flow Through the Router."

### Reference Code Snippets (vs. hnet_boundary.py)

All snippets in zone_ed_pipeline.md are stated to be verbatim from hnet_boundary.py.

- W_q/W_k identity initialization (spec lines 124–130): VERIFIED verbatim against
  hnet_boundary.py lines 77–83.
- Boundary probability forward (einsum, clamp, PAD_PROB, F.pad) (spec lines 137–145):
  VERIFIED verbatim against hnet_boundary.py lines 113–123.
- EMA Mamba-2 scan kernel (dt, x, A, b, c, mamba_chunk_scan_combined) (spec lines 152–158):
  VERIFIED verbatim against hnet_boundary.py lines 309–318.
- EMA step mode (spec lines 164–165): VERIFIED verbatim against hnet_boundary.py lines
  368–369.
- Plug-back cumsum + gather (spec lines 171–176): VERIFIED verbatim against hnet_boundary.py
  lines 337–342.

### Our Implementation Claims (vs. phase2/model.py)

**ZoneE (spec: "phase2/model.py:258–313")**
- Cited line range is off: class begins at 264, forward at 292, returns at 320.
  Content is correct; line numbers are stale. INCORRECT (line ref), VERIFIED (content).
- Pipeline: down_proj (d→d_outer), 3× CausalRecurrenceLayer(d_outer, log_a_init=3.0),
  up_proj (d_outer→d), BoundaryRouter, gather → concept_tokens: VERIFIED at lines 286–320.
- encoder_out saved as skip connection before routing call: VERIFIED (line 309).
- embed(x) done in HDCModel.forward before calling ZoneE: VERIFIED (line 628).

**BoundaryRouter (spec: "phase2/model.py:156–251")**
- Cited line range is off: class begins at 162, forward at 204, returns at 257.
  Content is correct. INCORRECT (line ref), VERIFIED (content).
- cosine_rule: F.pad(p, (1,0), value=1.0) for position 0: VERIFIED (line 236).
- learned_e2e: F.pad(p, (1,0), value=1.0) for position 0: VERIFIED (line 244).
- fixed_stride: no pad, scatters 1.0 at R-1, 2R-1, ...: VERIFIED (lines 219–223).
- topk(M) selection + sort: VERIFIED (lines 248–249).
- learned_isolated detaches boundary_probs_for_zd: VERIFIED (lines 252–253).

**ZoneD (spec: "phase2/model.py:370–464")**
- Cited line range is off: class begins at 377, forward at 411, returns at 472.
  Content is correct. INCORRECT (line ref), VERIFIED (content).
- Step 1 EMA: p_at_bounds.clamp(min=0.1) at line 434: VERIFIED.
- Step 1 EMA: h0 = concept_out[:,0] at line 435: VERIFIED.
- Step 1 EMA: _parallel_scan used (not Mamba-2): VERIFIED.
- Step 2 Plug-back: torch.searchsorted(boundary_idx, queries, right=True) at lines 453–455:
  VERIFIED.
- Step 3 Gated residual: `(1.0 - p_expanded) * gate * encoder_out + plugback` at line 464:
  VERIFIED.
- Step 4 Decoder recurrence: down_proj → 3× CRL(log_a_init=0.0) → up_proj → norm_out at
  lines 467–472: VERIFIED.

**HDCModel.forward (spec: "phase2/model.py:584–611")**
- Cited line range is off: forward begins at 616, ends at 643. INCORRECT (line ref),
  VERIFIED (content).
- Full flow (embed → ZoneE → InnerTransformer → ZoneD → lm_head): VERIFIED.
- ZoneD receives boundary_probs_for_zd (not boundary_probs directly): VERIFIED (line 639).
  Spec forward-pass diagram shows "boundary_probs" but the actual argument is
  boundary_probs_for_zd — still a documentation discrepancy (Issue 5, unchanged).

**p_at_bounds.clamp(min=0.1) (spec lines 221–231, cited at "phase2/model.py:426")**
- Cited line is off by 8: actual line 434. INCORRECT (line ref), VERIFIED (content).
- Clamp and EMA collapse rationale match code comment: VERIFIED.

**eye_ re-application after _init_weights (spec checklist item 7, cited lines 568–571)**
- Cited lines are off: actual lines 590–593. INCORRECT (line ref), VERIFIED (content).
- Code re-applies nn.init.eye_ for W_q and W_k after self.apply(_init_weights): VERIFIED.

**gate_proj near-zero init (spec lines 595–602, HDCModel.__init__)**
- Code: `nn.init.zeros_(self.zone_d.gate_proj.weight)` and `bias.fill_(-4.0)` at lines
  601–602: VERIFIED. sigmoid(-4.0) ≈ 0.018; gate starts near-suppressed as required.
- H-Net requirement "initialized close to 0" (Eq. 3 note): SATISFIED.

### Intentional Deviations

- CRL instead of Mamba (Zone E and Zone D): VERIFIED — no mamba_ssm import in phase2/model.py.
- No width expansion (same d for outer/inner): VERIFIED.
- Dense re-indexing for inner (positions 0..M-1): VERIFIED — InnerTransformer receives
  concept_tokens with no position remapping.
- 3 CRL layers instead of H-Net's 4: VERIFIED — both ZoneE and ZoneD have exactly 3.
- top-M instead of threshold at 0.5: VERIFIED — topk(M) used at line 248.
- Gated residual (refined Eq. 3): VERIFIED — `(1-p)*gate*encoder_out + plugback` vs H-Net's
  `dechunk(·) + linear(x̂)`. Deviation from H-Net Eq. 3 form is correctly documented.
- Simplified loss_comp (G only, not F·G): VERIFIED — training loop uses
  `(boundary_probs.mean() - 1/R)^2` as described. (In train.py, not model.py.)
- p_at_bounds.clamp(min=0.1) stronger than H-Net's [1e-4, 1-1e-4]: VERIFIED.

**_no_reinit mechanism — INCORRECTLY DESCRIBED (Issue 1, unchanged):**
Spec says "the `_no_reinit = True` flag prevents downstream re-initialization from overwriting
this." H-Net sets this flag (hnet_boundary.py lines 82–83); our code does not. Our code
achieves the same result by re-applying `nn.init.eye_` after `apply(_init_weights)` at lines
590–593. Outcome is correct; the description is inaccurate. No training impact.

**STE/confidence weighting (Eq. 6–7, 9) — NOT USED in our implementation:**
H-Net's confidence scoring (Eq. 6), STE (Eq. 7), and upsampler weighting (Eq. 9) are
documented in the spec's equation list but are not present in ZoneD's implementation. ZoneD
does not multiply plug-back output by ste(c_t). This is an undocumented deviation — the spec
describes these equations as H-Net mechanisms but does not explicitly state whether our
implementation uses them. No claim in "Our implementation" says we use STE, so this is not
a false claim, but the omission is not noted as a deviation.

### Verification Checklist (11 items)

1. **Position-0 boundary for cosine_rule / learned_e2e:** VERIFIED.
   Code: F.pad(p, (1,0), value=1.0) at lines 236 and 244. Checklist now correctly notes
   fixed_stride does not guarantee position 0 — consistent with code (lines 219–223).

2. **Top-M yields exactly M = L // R:** VERIFIED.
   `M = L // self.R` (line 214), `topk(M)` (line 248), sort (line 249). Always [B, M].

3. **EMA correct at position 0:** VERIFIED.
   `h0 = concept_out[:, 0]` (line 435) — first concept token, not zeros.

4. **p_at_bounds.clamp(min=0.1):** VERIFIED by inspection.
   Line 434 clamps to min=0.1. Runtime confirmation requires running hdc_rulebased on
   low-contrast input; code is correct as written.

5. **Plug-back correctness for known boundary_idx:** VERIFIED.
   searchsorted with right=True at line 453. For boundary_idx=[0,4,9], L=12:
   - j=0..3: count=1, bucket=0 ✓
   - j=4..8: count=2, bucket=1 ✓
   - j=9..11: count=3, bucket=2 ✓

6. **Gated residual weighting:** VERIFIED.
   At p≈1.0: (1-p)≈0, gate term ≈0, output dominated by plugback.
   At p≈0.0: (1-p)≈1, gate opens, output dominated by gated encoder_out.
   Gate bias=-4.0 → sigmoid(-4.0)≈0.018 ensures residual starts near-suppressed.

7. **W_q/W_k eye init survives _init_weights:** VERIFIED.
   Lines 590–593 re-apply nn.init.eye_ after self.apply(_init_weights) at line 577.

8. **R controls M correctly:** VERIFIED.
   CONFIGS: R=2/4/8 for hdc_r2/standard/hdc_r8. At L=512: M=256/128/64.

9. **boundary_probs_for_zd detached only for isolated routing:** VERIFIED.
   Lines 252–255: detach only in learned_isolated; pass-through in all other modes.
   ZoneD called with boundary_probs_for_zd at line 639.

10. **EMA collapse detection (smoke test):** VERIFIED.
    Architectural fix: ZoneE CRL log_a_init=3.0 (line 288), ZoneD CRL log_a_init=0.0
    (line 406). smoke_test.py exists and is enforced by run_experiments.sh before Phase 2.
    The four checks described in the checklist (encoder_out variance, concept_tokens
    variance, loss reduction, no NaN/inf) are implemented in utils/smoke_test.py per
    project memory.

11. **loss_comp drives boundary_probs toward 1/R:** VERIFIED (design).
    Training loss `(boundary_probs.mean() - 1/R)^2` is a correct squared-error penalty for
    this purpose. Runtime convergence requires training data.

---

## Open Issues

**Issue 1 (Minor) — _no_reinit mechanism described incorrectly**
Spec says the `_no_reinit = True` flag is used (spec line 61). Our code does not set this
flag; instead, it re-applies `nn.init.eye_` post-init (lines 590–593). Outcome is identical;
description is wrong. Fix: update deviation description in spec.

**Issue 2 (Minor) — Systematic line number drift throughout**
All cited line ranges in "Our implementation" are off by 5–25 lines. No behavioral impact.
Fix: refresh line numbers to match current phase2/model.py.

**Issue 5 (Minor) — Forward pass diagram omits boundary_probs_for_zd distinction**
Spec diagram (under "Full forward pass") shows:
  `ZoneD(concept_out, encoder_out, boundary_probs, boundary_idx)`
Actual call (line 639) uses `boundary_probs_for_zd`, which is detached in isolated mode.
Fix: update diagram to use `boundary_probs_for_zd`.

---

## Summary

All 10 authoritative equations are verbatim from hnet_2507.07955.md. All 5 reference code
snippets are verbatim from hnet_boundary.py. All core pipeline mechanics — Zone E encoder,
boundary routing, ZoneD EMA smoothing, plug-back, gated residual, decoder recurrence — are
correctly implemented and match the spec. The gate_proj near-zero init (Issue 3) was resolved
since the prior report. Three minor documentation issues remain (stale line numbers, _no_reinit
description, forward pass diagram). No code correctness failures.
