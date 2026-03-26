# Phase B' Verification: boundary_router
Date: 2026-03-26
Verdict: PASS with issues — All routing logic correct; stale line refs and one undocumented deviation.

---

## Per-claim status

### Authoritative Equations (vs. hnet_2507.07955.md, hnet_boundary.py, dlcm.md)
- H-Net Eq. (4) cosine dissimilarity: p_t = (1 - cos_sim(enc_t, enc_{t-1})) / 2: VERIFIED
- Adjacent key comparison (k_{t-1} not k_ema): VERIFIED
- p_0 = 1.0 (certain boundary at position 0): VERIFIED
- Top-M hard selection: VERIFIED
- Identity init for W_q/W_k: VERIFIED
- loss_comp = (boundary_probs.mean() - 1/R)^2: VERIFIED at phase2/train.py:340

### Reference Code Snippets (vs. hnet_boundary.py)
- Cosine similarity adjacent tokens: VERIFIED verbatim
- F.pad(p, (1, 0), value=1.0): VERIFIED verbatim
- topk + sort ascending: VERIFIED verbatim
- nn.init.eye_ for W_q/W_k: VERIFIED verbatim

### Our Implementation Claims (vs. phase2/model.py BoundaryRouter)
- cosine_rule mode (F.normalize, adjacent sim, pad): VERIFIED
- learned_e2e mode (W_q/W_k, adjacent q·k, pad): VERIFIED
- fixed_stride mode (arange, scatter 1.0): VERIFIED
- learned_isolated mode (detach before ZoneD): VERIFIED at :253
- Top-M via topk + sort: VERIFIED
- loss_comp at phase2/model.py:480 — INCORRECT: actual location is phase2/train.py:340; model.py:480 is HDCModel class docstring

### Intentional Deviations
- Dev 1 (adjacent k_{t-1} not k_ema): VERIFIED
- Dev 2 (loss_comp quadratic, no STE): VERIFIED
- Dev 3 (fixed_stride as lower bound): VERIFIED
- Dev 4 (learned_isolated detach): VERIFIED
- _no_reinit=True on W_q/W_k — NOT DOCUMENTED: H-Net reference sets this flag; our implementation omits it. Not listed in deviations.

### Verification Checklist
- All items: VERIFIED (behavior) — all line references stale (off ~6 lines throughout)
- loss_comp file location: INCORRECT in spec (cites model.py, lives in train.py)

---

## Issues

1. **Systematic line number drift** — all spec citations off by ~6 lines. Most critically, `phase2/train.py:592` is cited for loss_comp but the file has 513 lines. Actual loss_comp is at train.py:340.

2. **Wrong file for loss_comp** — Deviation 2 cites `phase2/model.py:480` for the loss_comp formula. Line 480 in model.py is the HDCModel class docstring. The formula is in `phase2/train.py:340`.

3. **Undocumented deviation: _no_reinit omitted** — The H-Net reference code sets `_no_reinit = True` on W_q/W_k to prevent the parent module's weight init from overwriting the identity init. Our code re-applies `nn.init.eye_` after `self.apply(_init_weights)` in HDCModel.__init__ (line 590–593), which achieves the same effect via a different mechanism. The outcome is correct but the deviation is not documented.

---

## Summary

All four routing modes are correctly implemented and verified against H-Net and DLCM sources. The p_0=1.0 invariant, adjacent key comparison, top-M selection, and identity init for learned routers all check out. The loss_comp formula and file location is wrong in the spec (cites model.py when it lives in train.py). The _no_reinit mechanism differs from reference but achieves the same outcome via post-init eye_ re-application — should be documented as an intentional deviation.
