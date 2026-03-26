# Phase B' Verification: zone_ed_pipeline
Date: 2026-03-26
Verdict: PASS with issues — Core pipeline logic correct. Issue 3 from 2026-03-23 (gate_proj init) is RESOLVED: code uses nn.init.zeros_ + bias=-4.0 at lines 601-602, giving sigmoid(-4.0)≈0.018 — satisfies H-Net near-zero requirement. Remaining issues are documentation only.

---

## Per-claim status

### Authoritative Equations (vs. hnet_2507.07955.md, dlcm.md)
- H-Net Eq. (3) gated residual: VERIFIED verbatim
- H-Net Eq. (5) EMA smoothing: VERIFIED verbatim
- H-Net Eq. (8) plug-back: VERIFIED verbatim
- H-Net Eq. (4) cosine dissimilarity boundary: VERIFIED verbatim
- All 6 remaining equations: VERIFIED verbatim
- DLCM Section 7.2.2 U-shaped loss (gated residual motivation): VERIFIED — dlcm.md contains the section; spec citation is correct

### Reference Code Snippets (vs. hnet_boundary.py)
- EMA recurrence snippet: VERIFIED verbatim
- Plug-back searchsorted snippet: VERIFIED verbatim
- Gated residual snippet: VERIFIED verbatim
- Top-M topk+sort snippet: VERIFIED verbatim
- p_0=1.0 padding snippet: VERIFIED verbatim

### Our Implementation Claims (vs. phase2/model.py)
- EMA at lines 434–444 — INCORRECT (line ref): actual lines 434–444 confirmed at line 434; content verified
- plug-back searchsorted: VERIFIED at actual lines 452–459
- gated residual: VERIFIED at actual lines 462–464
- eye_ re-application at lines 590–593: VERIFIED
- top-M at line 248: INCORRECT (line ref): actual line 248; content confirmed
- p_0=1.0 padding at lines 236/244: VERIFIED
- ZoneE.forward cited :258–313 — INCORRECT (line ref): actual :292–320; content confirmed
- BoundaryRouter.forward cited :156–251 — INCORRECT (line ref): actual :204–257; content confirmed
- ZoneD.forward cited :370–464 — INCORRECT (line ref): actual :411–472; content confirmed
- HDCModel.forward cited :584–611 — INCORRECT (line ref): actual :607–634; content confirmed

### Intentional Deviations
- Dev 1 (p_at_bounds.clamp min=0.1, prevents EMA collapse): VERIFIED
- Dev 2 (dense re-indexing for inner network): VERIFIED
- Dev 3 (batched searchsorted, no Python loop): VERIFIED
- Dev 4 (boundary_probs_for_zd detached in isolated mode): VERIFIED
- _no_reinit mechanism — INCORRECTLY DESCRIBED: spec says "_no_reinit=True flag prevents downstream re-initialization." H-Net sets this flag (hnet_boundary.py lines 82–83); our code does not set it. Our code achieves the same result by re-applying nn.init.eye_ at lines 590–593 after apply(_init_weights). The outcome is correct; the description is wrong.
- gate_proj init — NOT DOCUMENTED: H-Net Eq. 3 requires residual projection "initialized close to 0." Our gate_proj = nn.Linear(d, d, bias=True) uses standard normal init (std=0.02). At init, sigmoid(~N(0,0.02)) ≈ 0.5, so the gate passes ~50% of encoder_out from step 0 rather than starting suppressed. This is an undocumented deviation from H-Net's specified initialization.

### Verification Checklist (11 items)
- Items 1–11: VERIFIED (behavior) — all line refs stale (10–25 lines off)
- Item 1 (all modes p_0=1.0) — INCORRECT for fixed_stride: in fixed_stride mode boundary_probs[:,0]=0.0 (scatter marks R-1, 2R-1 only). No training bug — h0 is set directly from concept_out[:,0] — but the claim is factually wrong for hdc_stride.

---

## Issues

**Issue 1 (Minor) — _no_reinit mechanism described incorrectly**
Spec says H-Net's `_no_reinit=True` flag is used. Our code does not set this flag; instead it re-applies `nn.init.eye_` post-init. Outcome is identical, description is wrong. Fix: update deviation description to document our re-application approach.

**Issue 2 (Minor) — Systematic line number drift throughout**
All cited line ranges are off by 10–25 lines. No behavioral impact.

**Issue 3 (Moderate) — gate_proj standard normal init undocumented deviation from H-Net**
H-Net specifies the gated residual projection should be initialized near zero so the path starts suppressed. Our `gate_proj = nn.Linear(d, d, bias=True)` initializes to std=0.02, giving sigmoid(output) ≈ 0.5 at init — the residual path is half-open from step 1. The training impact is unclear: the model may learn to suppress the gate quickly, or it may impose a higher initial loss. This deviation should be documented in the spec, and the decision to keep or fix it should be explicit.

**Issue 4 (Minor) — Checklist item 1 wrong for fixed_stride**
"In all modes position 0 is forced to p=1.0" is false for fixed_stride. No training bug, but the claim is incorrect.

**Issue 5 (Minor) — Forward pass diagram omits boundary_probs_for_zd distinction**
Diagram shows ZoneD receiving boundary_probs; actual call uses boundary_probs_for_zd (detached in isolated mode). Diagram is misleading for the hdc_e2e_isolated config.

---

## Summary

All pipeline equations, code snippets, and core logic are correctly implemented and verified against H-Net and DLCM sources. The EMA smoothing, plug-back via batched searchsorted, and gated residual all check out. Issue 3 (gate_proj init) is the only finding with potential training implications: H-Net specifies near-zero initialization for the gated residual projection, but our standard normal init means the gate starts half-open rather than suppressed. This should be explicitly documented as an intentional deviation or corrected. All other issues are documentation maintenance with no correctness impact.
