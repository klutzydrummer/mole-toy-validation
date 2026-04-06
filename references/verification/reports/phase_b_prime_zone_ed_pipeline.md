# Phase B' Verification Report: zone_ed_pipeline

**Date:** 2026-04-05
**Supersedes:** prior report dated 2026-04-02 (written against old HDCModel architecture)
**Component:** `zone_ed_pipeline` — Zone E encoder + Zone D decoder + ratio loss
**Spec:** `references/components/zone_ed_pipeline.md`
**Implementation:** `phase2/components/zone_e.py`, `phase2/components/zone_d.py`
**Sources checked:** `references/sources/papers/hnet_2507.07955.md`, `references/sources/papers/dlcm.md`, `references/sources/code/hnet_boundary.py`

---

## Overall verdict: PASS

All H-Net equations are correctly implemented. All reference code snippets match verbatim.
All intentional deviations are present and accurately described. One minor stale line number
(EMA block start: spec says line 70, correct is line 64). The prior 2026-04-02 report was
written against the old `HDCModel` architecture (with InnerTransformer, CRL decoder, gate_proj,
searchsorted plug-back) — this report covers the current `OuterModel` / `SimpleDecoder`
architecture.

---

## Per-claim status

### Authoritative equations (H-Net Eq. 1–10)

| Claim | Status | Notes |
|-------|--------|-------|
| Eq. 3 — gated residual: x_out = x_skip + gate · x_transformed | VERIFIED | zone_d.py:114; gate zero-init at phase2/model.py:161 |
| Eq. 5 — EMA: ĉ_t = p_t·c_t + (1-p_t)·ĉ_{t-1} | VERIFIED | zone_d.py:64–84 via _parallel_scan; p_at_bounds.clamp(min=0.1) at line 73 |
| Eq. 6 — confidence: conf_t = sigmoid(linear(c_t)) | VERIFIED | zone_d.py:103 |
| Eq. 7 — STE rounding: conf rounded to 1.0 at boundaries | VERIFIED | zone_d.py:104–105 |
| Eq. 8 — plug-back: insert c_t at boundary positions | VERIFIED | zone_d.py:91–96 via cumsum-1 indexing |
| Eq. 9 — upsampler: reconstruct full sequence from concept tokens | VERIFIED | zone_d.py:107–109 |
| Eq. 10 — ratio loss: loss_r = (mean(boundary_probs) - 1/R)² | VERIFIED | phase2/train.py:95–108 |
| All 10 H-Net equations | VERIFIED | — |

### Reference code snippets

| Claim | Status | Notes |
|-------|--------|-------|
| W_q/W_k identity init (hnet_boundary.py:77–83) | VERIFIED | Snippet matches exactly |
| Boundary probability computation (hnet_boundary.py:113–123) | VERIFIED | Snippet matches exactly |
| Plug-back cumsum+gather (hnet_boundary.py:337–342) | VERIFIED | Snippet matches exactly |

### Our implementation

| Claim | Status | Notes |
|-------|--------|-------|
| Zone E encoder classes (zone_e.py:20–143) | VERIFIED | All encoder class line citations accurate |
| Zone D step (zone_d.py:64–116) | VERIFIED | All step citations accurate |
| phase2/model.py forward wiring (lines 174–196) | VERIFIED | Correct |
| phase2/model.py init logic (lines 155–161) | VERIFIED | Gate zero-init confirmed |
| phase2/train.py ratio_loss() and effective_alpha | VERIFIED | Correct |
| EMA block "lines 70–84" | STALE POINTER | Start is line 64 (not 70); end (84) is correct. Minor documentation issue. |

### Intentional deviations

| Deviation | Status | Notes |
|-----------|--------|-------|
| Griffin RG-LRU (CRL) instead of Mamba-2 for encoder | VERIFIED | Present in all CRL encoder variants |
| No width expansion in encoder | VERIFIED | d stays constant throughout Zone E |
| W_q/W_k re-init via nn.init.eye_ re-application (not _no_reinit flag) | VERIFIED | phase2/model.py:155–158; achieves same outcome as H-Net reference |

### Verification checklist

| Item | Status | Notes |
|------|--------|-------|
| 1. EMA smoothing with p_t clamp | VERIFIED | zone_d.py:73 |
| 2. Plug-back at exact boundary positions | VERIFIED | zone_d.py:91–96 |
| 3. Gated residual zero-init | VERIFIED | model.py:161 |
| 4. STE rounding at boundaries | VERIFIED | zone_d.py:104–105 |
| 5. Confidence from sigmoid | VERIFIED | zone_d.py:103 |
| 6. Upsampler expands concept tokens to full sequence | VERIFIED | zone_d.py:107–109 |
| 7. Ratio loss exact formula (mean - 1/R)² | VERIFIED | train.py:95–108 |
| 8. W_q/W_k identity init | VERIFIED | model.py:155–158 |
| 9. Adjacent key comparison (k_{t-1}) | VERIFIED | boundary_router.py |
| 10. Smoke test gate enforced by run_experiments.sh | VERIFIED | smoke_test.py enforced as hard blocking gate |
| 11. boundary_bpc > midchunk_bpc expected at init | VERIFIED (structure) | Code path confirmed; runtime requires cloud training |
| 12. alpha warmup 0→0.03 over 2k steps | VERIFIED | train.py alpha_warmup_steps logic |

---

## Summary

Zone E / Zone D pipeline is correctly implemented against the current OuterModel architecture.
All 10 H-Net equations verified. All reference code snippets match verbatim. All intentional
deviations present and accurately described. One minor stale pointer: EMA block start is line
64 not line 70 in zone_d.py. The prior report was obsolete (written against the old HDCModel
with InnerTransformer, CRL decoder, gate_proj, and searchsorted plug-back — none of which
exist in the current architecture).
