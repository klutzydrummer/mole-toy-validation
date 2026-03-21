# Phase B' Validation Report — Zone E / Zone D Pipeline

**Component file:** `references/components/zone_ed_pipeline.md`
**Sources checked:**
- `references/sources/papers/hnet_2507.07955.md` (H-Net arXiv:2507.07955)
- `references/sources/code/hnet_boundary.py` (goombalab/hnet `dc.py`, MIT license)

**Date:** 2026-03-21 (re-verified; original 2026-03-17)
**Validator:** Phase B' validation agent

**Re-verification note (2026-03-21):** Full re-check including implementation cross-check against actual line numbers. Implementation is substantially correct — no training bugs. Three issues found (all documentation):

1. **Line numbers systematically ~9 off (Low):** Every cited line range in "Our implementation" is off by ~9 lines. ZoneE.forward (cited 258-313, actual 293-321), BoundaryRouter.forward (cited 156-251, actual 205-258), ZoneD.forward (cited 370-464, actual 412-473), EMA block (cited 426-436, actual 435-445), plug-back (cited 444-451, actual 453-460), gated residual (cited 453-456, actual 462-465), decoder recurrence (cited 458-463, actual 467-473), HDCModel.forward (cited 584-611, actual 596-623).

2. **`_no_reinit` mechanism description inaccurate (Low):** Spec says "`_no_reinit = True` flag prevents re-initialization." Our code does NOT set that flag. Instead `HDCModel.__init__` lines 580-583 re-apply `nn.init.eye_` explicitly after `self.apply(_init_weights)`. Same net result, different mechanism.

3. **Checklist item 1 over-claims for fixed_stride (Low):** States "in all modes position 0 is forced to p=1.0." False for `fixed_stride` — `boundary_probs[:,0] = 0.0` in that mode. No training bug because `h0 = concept_out[:,0]` is set directly at line 436, bypassing the p formula. But the checklist claim is inaccurate for `hdc_stride`.

4. **Checklist item 10 smoke test thresholds — PASS, exact match:** `ENCODER_DIVERSITY_MIN=1e-4`, `CONCEPT_DIVERSITY_MIN=1e-4`, `LOSS_REDUCTION_REQUIRED=0.90`, NaN/inf check all confirmed against `utils/smoke_test.py`.

---

## Overall Verdict: PASS (with noted deviations)

All authoritative equations and code snippets are traceable to the cited sources. Deviations from the source that are introduced in the "Our implementation" section are clearly labeled as intentional. No contradictions were found between the component file and the source material. One minor discrepancy in equation indexing notation is flagged below as acceptable (mathematically equivalent).

---

## Verified Claims

### Equations (Section "Authoritative equations")

**Eq. 1 — Main processing flow**
- Claim: `x̂ˢ = ℰˢ(xˢ), ẑˢ = 𝓜(xˢ), ẑˢ = 𝒟ˢ(zˢ)`
- Source: `hnet_2507.07955.md` Equation (1), lines 28–30. Verbatim match.
- Status: VERIFIED

**Eq. 2 — Chunking operation**
- Claim: `(xˢ⁺¹, pˢ) = chunk(x̂ˢ)`
- Source: `hnet_2507.07955.md` Equation (2), lines 37–38. Verbatim match.
- Status: VERIFIED

**Eq. 3 — Gated residual / dechunking with residual connection**
- Claim: `zˢ = dechunk(ẑˢ⁺¹, pˢ) + linear(x̂ˢ)`
- Claim (quote): "we adopt the first approach – adding a projection (linear) only to the residual connection."
- Claim (zero init quote): "this residual connection is initialized close to 0; earlier versions of H-Net found this to be an important detail, but it may be less important when combined with additional techniques such as LR modulation."
- Source: `hnet_2507.07955.md` Equation (3) and surrounding text, lines 44–53. Verbatim match on equation and both quoted strings.
- Status: VERIFIED

**Eq. 4 — Boundary probability computation**
- Claim: Full four-line formula for qₜ, kₜ, pₜ, bₜ.
- Claim (quote): "p₁ = 1.0 by definition, ensuring the sequence begins with a boundary."
- Claim (W_q/W_k init note): identity initialization and `_no_reinit = True`.
- Source: `hnet_2507.07955.md` Equation (4), lines 62–83. Verbatim match on all three elements.
- Status: VERIFIED

**Eq. 5 — EMA smoothing**
- Claim: `z̄ₜ = Pₜ ẑₜ + (1 − Pₜ) z̄ₜ₋₁`
- Claim (quote): "The smoothing module applies an exponential moving average (EMA) with the following definition."
- Source: `hnet_2507.07955.md` Equation (5), lines 106–109. Verbatim match.
- Status: VERIFIED

**Eq. 6 — Confidence scoring**
- Claim: `cₜ = pₜ^(bₜ) (1 − pₜ)^(1−bₜ)` with piecewise expansion.
- Claim (quote): "The coefficient c quantifies the routing module's confidence in its boundary decisions."
- Source: `hnet_2507.07955.md` Equation (6), lines 129–133. Verbatim match.
- Status: VERIFIED

**Eq. 7 — Straight-Through Estimator (STE)**
- Claim: `ste(cₜ) = cₜ + stopgradient(1 − cₜ)`
- Claim (quote): "The Straight-Through Estimator (STE)...rounds confidence scores to 1.0 in the forward pass while maintaining continuous gradients during backpropagation."
- Source: `hnet_2507.07955.md` Equation (7), lines 138–141. Verbatim match.
- Status: VERIFIED

**Eq. 8 — Causal plug-back**
- Claim: `z̃ₜ = z̄_{∑ₖ₌₁ᵗ bₖ}`
- Claim (quote): "The upsampling operation repeats each compressed vector until the next boundary position..."
- Source: `hnet_2507.07955.md` Equation (8), lines 146–149. Verbatim match.
- Status: VERIFIED

**Eq. 9 — Upsampler output**
- Claim: `upsampler(z̄, c)ₜ = ste(cₜ) · z̃ₜ`
- Claim (quote): "Multiplying upsampled vectors by their confidence scores incentivizes..."
- Source: `hnet_2507.07955.md` Equation (9), lines 164–167. Verbatim match.
- Status: VERIFIED

**Eq. 10 — Ratio loss**
- Claim: Full formula for ℒ_ratio with definitions of F and G. Combined objective `ℒ = ℒ_AR + α ∑ₛ ℒ_ratio^s`, α = 0.03.
- Source: `hnet_2507.07955.md` Equation (10), lines 176–190. Verbatim match on formula, definitions, and α value.
- Status: VERIFIED

---

### Code Snippets (Section "Reference implementation")

All code snippets in this section are claimed to be "verbatim from `sources/code/hnet_boundary.py`".

**RoutingModule — W_q / W_k identity initialization**
- Claim: 8-line block ending with `_no_reinit = True` on both weights.
- Source: `hnet_boundary.py` lines 77–83. Verbatim match (including `factory_kwargs` argument that appears in the source but is omitted from the component's snippet — this is an acceptable excerpt, not a discrepancy).
- Status: VERIFIED

**RoutingModule — boundary probability computation (forward pass)**
- Claim: `cos_sim` einsum, `boundary_prob` clamp, `PAD_PROB = 1.0`, `F.pad` call.
- Source: `hnet_boundary.py` lines 113–123. Verbatim match.
- Status: VERIFIED

**DeChunkLayer — EMA via Mamba-2 scan kernel (training)**
- Claim: Lines beginning `p = torch.clamp(...)`, `dt = torch.log(...)`, `x = (hidden_states / dt...)`, `A = -torch.ones(...)`, `b = p...`, `c = torch.ones_like(b)`, `out = mamba_chunk_scan_combined(...)`.
- Source: `hnet_boundary.py` lines 288–325. All lines verified verbatim. The component elides the intervening rearrange/repeat call arguments with `...` in the `mamba_chunk_scan_combined(...)` call, which is an acceptable excerpt.
- Status: VERIFIED

**DeChunkLayer — EMA step mode (inference)**
- Claim: `result = p * current_hidden_states + (1 - p) * inference_params.last_value` and `inference_params.last_value.copy_(result)`.
- Source: `hnet_boundary.py` lines 368–369. Verbatim match.
- Status: VERIFIED

**DeChunkLayer — plug-back (Eq. 8)**
- Claim: `plug_back_idx = torch.cumsum(boundary_mask, dim=1) - 1` followed by `torch.gather` call.
- Source: `hnet_boundary.py` lines 337–342. Verbatim match.
- Status: VERIFIED

---

### Source Attribution

- Component file states code is from `github.com/goombalab/hnet`, MIT license, retrieved 2026-03-17. The source file header (`hnet_boundary.py` lines 1–20) confirms the same repository, commit, license, and retrieval date.
- Status: VERIFIED

---

## Deviations from Source (Labeled in Component as Intentional)

The following items appear in the "Our implementation" section and are explicitly marked as deviations. They are not errors; they are design choices that differ from H-Net. Each is checked for internal consistency and cross-checked against the source.

**Deviation 1 — CausalRecurrenceLayer (RG-LRU) instead of Mamba-2 in Zones E and D**
- Component claim: "intentional substitution to avoid the `mamba_ssm` dependency in the outer zones; the role... is the same."
- Source reference: `hnet_2507.07955.md` lines 196–199 confirm H-Net uses Mamba-2 layers in encoder/decoder.
- The substitution is not contradicted by the source; it is an architectural choice.
- Status: DEVIATION — CLEARLY LABELED, CONSISTENT

**Deviation 2 — No width expansion (both zones d=256)**
- Component claim: H-Net requires D⁰ ≤ D¹ ≤ ... and appends a shared trainable vector; our outer/inner widths are both d=256.
- Source reference: `hnet_2507.07955.md` lines 198–201 confirm the monotonically non-decreasing width constraint and the append mechanism.
- The component correctly describes the H-Net constraint and accurately states it does not apply.
- Status: DEVIATION — CLEARLY LABELED, CONSISTENT

**Deviation 3 — Dense re-indexing (positions 0..M-1) instead of sparse original-position RoPE**
- Component claim: "H-Net's validated approach for causal LM. Sparse original-position indexing is left as ablation A-new."
- Source: Not explicitly specified in the paper source file provided (the paper source does not discuss RoPE indexing strategy). The claim "H-Net's validated approach for causal LM" is asserted without a direct citation to a specific paper line.
- Status: DEVIATION — LABELED BUT PARTIALLY UNSUPPORTED (see Unverified Claims)

**Deviation 4 — top-M selection instead of 0.5-threshold**
- Component claim: "H-Net selects boundaries by `argmax(boundary_prob) == 1` (soft threshold at 0.5), which yields a variable number of tokens per sequence."
- Source check: `hnet_boundary.py` line 134 confirms `boundary_mask = selected_idx == 1`, which is equivalent to threshold at 0.5 after argmax over the two-class softmax. This matches the component's characterization.
- The top-M substitution rationale ("our inner transformer requires a fixed sequence length") is an internal design constraint, not contradicted by the source.
- Status: DEVIATION — CLEARLY LABELED, CONSISTENT

**Deviation 5 — Gated residual refinement in Zone D**
- Component claim: residual weight is `(1 - p_j) * sigmoid(W_gate * encoder_out_j)` rather than a plain `linear`; cites DLCM Section 4.2 U-shaped loss.
- Source: Eq. 3 in `hnet_2507.07955.md` specifies `linear(x̂ˢ)`. The gated form is a documented enhancement. The DLCM reference is external and not included in the sources provided for this validation; the claim about "U-shaped loss described in DLCM Section 4.2" cannot be verified from the provided sources.
- Status: DEVIATION — LABELED; DLCM CITATION UNVERIFIABLE FROM PROVIDED SOURCES

**Deviation 6 — p_at_bounds.clamp(min=0.1) instead of H-Net's clamp(min=1e-4)**
- Component claim: H-Net clamps to `[1e-4, 1-1e-4]` (only prevents exactly-zero); our `min=0.1` is a stronger guard.
- Source check: `hnet_boundary.py` line 288 confirms `p = torch.clamp(boundary_prob[..., -1].float(), min=1e-4, max=1-(1e-4))`. Component's characterization of the H-Net value is correct.
- The rationale for the stronger clamp (EMA collapse prevention) is plausible and internally consistent.
- Status: DEVIATION — CLEARLY LABELED, CONSISTENT

**Deviation 7 — Simplified compression loss**
- Component claim: `loss_comp = lambda_comp * (boundary_probs.mean() - 1/R)^2` uses only G (average probability) not both F and G as in H-Net Eq. 10. Rationale: "sufficient for the top-M selection strategy."
- Source: `hnet_2507.07955.md` Equation (10) confirms H-Net uses both F and G. The simplification is correctly described and the rationale is internally consistent.
- Status: DEVIATION — CLEARLY LABELED, CONSISTENT

---

## Unverified / Unsupported Claims

**U1 — "Dense re-indexing... H-Net's validated approach for causal LM"**
- Location: "Our implementation" → BoundaryRouter section, Deviation 3.
- The phrase "H-Net's validated approach for causal LM" implies the paper validates dense sequential positions for RoPE in the inner network, but the paper source file (`hnet_2507.07955.md`) does not contain any discussion of RoPE indexing strategy. This claim has no traceable support in the provided sources.
- Severity: LOW — the deviation itself (using dense positions) is reasonable and the unsupported part is only the attribution of that choice to H-Net validation.

**U2 — "DLCM Section 4.2 U-shaped loss"**
- Location: "Our implementation" → Zone D section, step 3 (Gated residual).
- This is a citation to an external source (DLCM) that is not included in the references list of the component file and is not present as a source file. The claim is that a gated residual mitigates a "U-shaped loss described in DLCM Section 4.2."
- Severity: MEDIUM — this is a design justification claim with no verifiable backing from the provided sources. It is not a contradiction, but it is an unsupported assertion used to motivate a deviation from the canonical H-Net formula.

**U3 — "4 layers (or equivalent)" discrepancy**
- Location: "Our implementation" → Zone E, step 4 (Decoder recurrence), and Zone D section.
- The component specifies "3× CausalRecurrenceLayer" in both Zone E and Zone D, while `hnet_2507.07955.md` line 196 states H-Net "always stick[s] to four layers (or the equivalent of four Mamba layers) in each encoder/decoder network." The component does not acknowledge this numerical discrepancy (3 vs. 4 layers). It is labeled as a recurrence-type substitution but not as a depth change.
- Severity: LOW-MEDIUM — this is an implicit deviation (layer count differs from H-Net's stated practice) that is not flagged as an explicit deviation in the component file.

---

## Contradictions

No internal contradictions were found between the component file's claims and the cited source material.

One apparent tension is noted but is not a contradiction:

**T1 — Eq. 4 boundary index convention**
- The component's Eq. 4 states `pₜ = (1/2)(1 − (qₜᵀ kₜ₋₁) / ...)`, implying q at position t compared against k at position t-1.
- The source paper (`hnet_2507.07955.md`) states the same formula.
- However, the code (both in `hnet_2507.07955.md` lines 87–91 and `hnet_boundary.py` lines 113–116) computes `q` from `hidden_states[:, :-1]` and `k` from `hidden_states[:, 1:]`, meaning `cos_sim[t]` = cosine(q_t, k_{t+1}), which after the `F.pad(..., (1, 0))` shift gives `boundary_prob[t]` = dissimilarity between tokens t-1 and t.
- The component's Eq. 4 notation and the code's implementation are mathematically equivalent after accounting for the pad; this is consistent with the paper's own explanation. Not a contradiction.
- Status: NOTED — CONSISTENT (equivalent after index shift)

---

## Summary Table

| Item | Type | Status |
|---|---|---|
| Eq. 1 | Paper equation | VERIFIED |
| Eq. 2 | Paper equation | VERIFIED |
| Eq. 3 (formula + both quotes) | Paper equation + quotes | VERIFIED |
| Eq. 4 (formula + p₁ quote + init) | Paper equation + quotes | VERIFIED |
| Eq. 5 (formula + quote) | Paper equation + quote | VERIFIED |
| Eq. 6 (formula + quote) | Paper equation + quote | VERIFIED |
| Eq. 7 (formula + quote) | Paper equation + quote | VERIFIED |
| Eq. 8 (formula + quote) | Paper equation + quote | VERIFIED |
| Eq. 9 (formula + quote) | Paper equation + quote | VERIFIED |
| Eq. 10 (formula + α) | Paper equation | VERIFIED |
| Code: W_q/W_k identity init | Code snippet | VERIFIED |
| Code: boundary_prob computation | Code snippet | VERIFIED |
| Code: EMA Mamba-2 scan (training) | Code snippet | VERIFIED |
| Code: EMA step (inference) | Code snippet | VERIFIED |
| Code: plug-back gather | Code snippet | VERIFIED |
| Deviation: RG-LRU vs Mamba-2 | Labeled deviation | CONSISTENT |
| Deviation: no width expansion | Labeled deviation | CONSISTENT |
| Deviation: dense re-indexing | Labeled deviation | PARTIALLY UNSUPPORTED (U1) |
| Deviation: top-M vs threshold | Labeled deviation | CONSISTENT |
| Deviation: gated residual | Labeled deviation | DLCM CITE UNVERIFIABLE (U2) |
| Deviation: clamp(min=0.1) | Labeled deviation | CONSISTENT |
| Deviation: simplified loss | Labeled deviation | CONSISTENT |
| 3 layers vs H-Net's 4 | Implicit deviation | NOT FLAGGED (U3) |
