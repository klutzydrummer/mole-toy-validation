# Phase B' Verification Report: BoundaryRouter

**Component file:** `references/components/boundary_router.md`
**Verified against:**
- `references/sources/papers/hnet_2507.07955.md` (H-Net, arXiv:2507.07955)
- `references/sources/papers/dlcm.md` (DLCM, arXiv:2512.24617)
- `references/sources/code/hnet_boundary.py` (H-Net `RoutingModule` / `DeChunkLayer`, MIT)

**Date:** 2026-03-21 (re-verified; original 2026-03-17)
**Overall verdict: PASS with issues**

**Re-verification note (2026-03-21):** Full re-check performed including implementation cross-check against actual line numbers. All algorithmic invariants correct — no training bugs. Four issues found (all documentation):

1. **Line numbers systematically ~7 off (Low):** All `model.py` line citations in the spec are ~7 lines below actual. `BoundaryRouter` at 163 (spec: 156), `ZoneE` at 265 (spec: 258), identity init at 202-203 (spec: 195-196), p₀ pad at 237/245 (spec: 230/238), topk at 249-250 (spec: 242-243), detach at 254 (spec: 247).

2. **`loss_comp` citation wrong (Low-Moderate):** Spec cites `model.py:480` (docstring, not code) and `train.py:592` (file has only 455 lines). Actual location: `train.py:298`.

3. **`_no_reinit` flag absent (Informational):** H-Net uses `weight._no_reinit = True`. Our code instead re-applies `nn.init.eye_` explicitly after `_init_weights` at `model.py:581-583`. Same net result, different mechanism.

4. **Checklist item 9 framing imprecise (Low):** Says "confirm `W_q.weight.grad` is None with `learned_isolated`" — but `loss_comp` gradient does reach W_q/W_k in that mode. Only LM loss gradient is blocked. Spec prose is correct; checklist phrasing misleads.

All equations are traceable to their cited sources (verbatim or mathematically equivalent). All code snippets are present verbatim in the source file at the stated line ranges. All stated deviations are accurately described. No internal contradictions were found. One minor indexing note is documented below.

---

## Verified Claims

### 1. H-Net Eq. (4) — Boundary probability (learned routing)

**Claim:** Component cites H-Net (arXiv:2507.07955) Equation 4 and states:
```
qₜ = W_q x̂ₜ
kₜ = W_k x̂ₜ
pₜ = (1/2)(1 − (qₜᵀ kₜ₋₁) / (‖qₜ‖ ‖kₜ₋₁‖)) ∈ [0, 1]
bₜ = 𝟙{pₜ ≥ 0.5}
```
and `p₁ = 1.0` by definition.

**Verification:** VERIFIED. `hnet_2507.07955.md` § "Equation (4) — Boundary Probability Computation (Learned Routing)" contains this equation verbatim. The source file also quotes: "p₁ = 1.0 by definition, ensuring the sequence begins with a boundary."

**Source pointer:** `hnet_2507.07955.md` lines 59–70.

---

### 2. W_q / W_k identity initialization claim

**Claim:** "W_q and W_k are initialized to the identity matrix so that at init the router computes raw cosine similarity with no learned transformation applied."

**Verification:** VERIFIED. `hnet_2507.07955.md` lines 72–83 reproduce the initialization block from `dc.py`. `hnet_boundary.py` lines 79–83 contain the code verbatim:
```python
with torch.no_grad():
    self.q_proj_layer.weight.copy_(torch.eye(d_model))
    self.k_proj_layer.weight.copy_(torch.eye(d_model))
self.q_proj_layer.weight._no_reinit = True
self.k_proj_layer.weight._no_reinit = True
```

**Source pointer:** `hnet_boundary.py` lines 79–83; `hnet_2507.07955.md` lines 72–83.

---

### 3. `_no_reinit = True` claim

**Claim:** "`_no_reinit = True` prevents any downstream initialization sweep from overwriting the identity."

**Verification:** VERIFIED. The flag is set on both `q_proj_layer.weight` and `k_proj_layer.weight` at `hnet_boundary.py` lines 82–83, consistent with the paper source's annotation: "The `_no_reinit = True` flag prevents any downstream re-initialization from overwriting this."

**Source pointer:** `hnet_boundary.py` lines 82–83.

---

### 4. Code snippet: boundary probability computation (training, parallel mode)

**Claim:** Component attributes lines 113–123 of `hnet_boundary.py` to the parallel-mode forward pass.

**Verification:** VERIFIED. The code block in the component matches `hnet_boundary.py` lines 113–123 verbatim:
```python
cos_sim = torch.einsum(
    "b l d, b l d -> b l",
    F.normalize(self.q_proj_layer(hidden_states[:, :-1]), dim=-1),
    F.normalize(self.k_proj_layer(hidden_states[:, 1:]), dim=-1),
)
boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)
PAD_PROB = 1.0
boundary_prob = F.pad(boundary_prob, (1, 0), "constant", PAD_PROB)
```

**Source pointer:** `hnet_boundary.py` lines 113–123.

---

### 5. Code snippet: boundary mask (threshold at 0.5)

**Claim:** Component attributes lines 130–134 of `hnet_boundary.py` to the boundary mask computation.

**Verification:** VERIFIED. The code block matches `hnet_boundary.py` lines 130–134 verbatim:
```python
boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)
selected_idx = torch.argmax(boundary_prob, dim=-1)
boundary_mask = selected_idx == 1
```

**Source pointer:** `hnet_boundary.py` lines 130–134.

---

### 6. Code snippet: step (autoregressive inference)

**Claim:** Component attributes lines 169–192 of `hnet_boundary.py` to the step mode.

**Verification:** VERIFIED. The code block in the component matches `hnet_boundary.py` lines 169–192 verbatim (the `step` method body). The component's description — "compares the current token's k-projection against the stored last hidden state's q-projection, enforcing p=1.0 for the very first token seen" — is consistent with the code at lines 171–182.

**Source pointer:** `hnet_boundary.py` lines 168–192.

---

### 7. DLCM Eq. (6) — Boundary probability (independent formulation)

**Claim:** Component cites DLCM (arXiv:2512.24617) Equation 6 and states:
```
pₜ = (1 − cos(qₜ₋₁, kₜ)) / 2 = (1/2)(1 − (qₜ₋₁ᵀ kₜ) / (‖qₜ₋₁‖₂ ‖kₜ‖₂))
```
with the note that the index convention differs from H-Net by one position.

**Verification:** VERIFIED. `dlcm.md` § "Equation (6) — Boundary Probability via Cosine Dissimilarity" contains this equation verbatim. The source file also notes: "q is from the *previous* token (t−1), k is from the *current* token (t)." The component's characterization of the index difference as measuring the same adjacent-token dissimilarity is confirmed by `dlcm.md`'s comparison table at lines 254–267 ("same dissimilarity").

**Source pointer:** `dlcm.md` lines 68–74.

---

### 8. DLCM `p₁ = 1` claim

**Claim:** "DLCM also enforces `p₁ = 1`."

**Verification:** VERIFIED. `dlcm.md` lines 83 and 245 both state: "We enforce p₁ = 1 so that the first token always starts a new concept."

**Source pointer:** `dlcm.md` lines 83, 245.

---

### 9. H-Net Eq. (10) — Ratio loss

**Claim:** Component cites H-Net (arXiv:2507.07955) Equation 10 as:
```
ℒ_ratio = (N / (N−1)) · ((N−1) · F · G + (1−F) · (1−G))
F = (1/L) ∑ₜ bₜ
G = (1/L) ∑ₜ pₜ
```
with minimum 1.0 when F = G = 1/N, and α = 0.03 in all experiments.

**Verification:** VERIFIED. `hnet_2507.07955.md` § "Equation (10) — Targeted Compression Loss" contains this equation verbatim with the same definitions and the note "loss achieves minimum value 1.0 when F = G = 1/N." The α = 0.03 value is also confirmed at `hnet_2507.07955.md` line 190.

**Source pointer:** `hnet_2507.07955.md` lines 173–190.

---

### 10. DLCM Eq. (10) — Auxiliary loss

**Claim:** Component cites DLCM (arXiv:2512.24617) Equation 10 as:
```
ℒ_aux = (R / (R−1)) · [(R−1) · F_global · G_global + (1−F_global) · (1−G_global)] − 1
```
with the note that it is "structurally identical" to H-Net's loss, shifted by −1 so minimum is 0.

**Verification:** VERIFIED. `dlcm.md` § "Equation (10) — Auxiliary Load-Balancing Loss" contains this equation verbatim. The source file also confirms: "This loss is structurally identical to H-Net's ℒ_ratio (Eq. 10), with R in place of N. The −1 offset means the loss is 0 at the target operating point F = G = 1/R."

**Source pointer:** `dlcm.md` lines 114–122.

---

### 11. DLCM — Gradient conflict (Eq. 24)

**Claim:** Component cites DLCM (arXiv:2512.24617) Equation 24 for the gradient conflict:
```
∇_θ ℒ_total = ∇_θ ℒ_CE  +  λ · ∇_θ ℒ_aux
               (anti-compression)   (pro-compression)
```
and states: "DLCM names this 'Gradient Conflict in Learned Boundary Prediction' but does not apply a stop-gradient to resolve it. H-Net likewise has no stop-gradient through the boundary probability computation."

**Verification:** VERIFIED. `dlcm.md` lines 136–143 reproduce this equation and confirm the label "Gradient Conflict in Learned Boundary Prediction" and that "There is no stop-gradient or gradient isolation mentioned." `hnet_2507.07955.md` lines 211–215 confirm: "There is no mention of gradient isolation or detaching gradients through the boundary probability computation itself."

**Source pointer:** `dlcm.md` lines 136–143; `hnet_2507.07955.md` lines 211–215.

---

### 12. H-Net STE (Eq. 7) — No STE in our implementation

**Claim:** "H-Net wraps confidence scores with `ste(c) = c + stopgradient(1 − c)` (H-Net Eq. 7)... We do not apply STE because top-M selection is not used as a multiplicative weight on the token representations."

**Verification:** VERIFIED. `hnet_2507.07955.md` § "Equation (7) — Straight-Through Estimator" confirms: `ste(cₜ) = cₜ + stopgradient(1 − cₜ)`. The claim that H-Net uses this and the project does not is an intentional deviation — the component correctly identifies it as such. The STE is present in the H-Net source; the "not used" claim is about the project's own code and is not a source-verifiable claim (it is self-referential to `phase2/model.py`).

**Source pointer:** `hnet_2507.07955.md` lines 135–142.

---

### 13. H-Net: no stop-gradient through boundary probability

**Claim:** "H-Net...has no stop-gradient through the boundary probability computation — gradient isolation is our own addition."

**Verification:** VERIFIED. `hnet_2507.07955.md` lines 211–214 explicitly state: "The only explicit gradient-blocking mechanism is the STE in Equation (7)...There is no mention of gradient isolation or detaching gradients through the boundary probability computation itself." The claim that `learned_isolated` is an original project contribution with no antecedent in either H-Net or DLCM is consistent with both sources.

**Source pointer:** `hnet_2507.07955.md` lines 211–214; `dlcm.md` lines 136–143.

---

### 14. Note on index convention in code vs. equation (H-Net Eq. 4)

**Claim:** The component states at line 114: "q is from tokens `[0..L-2]`, k is from tokens `[1..L-1]`. After padding, `boundary_prob[t]` measures dissimilarity between token `t−1` and token `t`."

**Verification:** VERIFIED with a note. The paper's Eq. (4) states `qₜ = W_q x̂ₜ` and `pₜ = (1/2)(1 − qₜᵀkₜ₋₁ / ...)`, i.e., q from current token t and k from previous token t−1. The code computes `q` on `hidden_states[:, :-1]` and `k` on `hidden_states[:, 1:]`, which is q from position `t` and k from position `t+1`, then assigns the result to `cos_sim[t]`, yielding `boundary_prob[t+1]` after padding. The final semantics are the same: `boundary_prob[t]` measures dissimilarity between token `t−1` and token `t`. The component correctly describes the net effect. The `hnet_2507.07955.md` source note at line 97 confirms this: "q from token `t-1` and k from token `t`" — a consistent description of the same index shift. No contradiction.

**Source pointer:** `hnet_boundary.py` lines 110–112 (inline comments confirm the semantics); `hnet_2507.07955.md` lines 85–97.

---

## Unverified / Self-Referential Claims (not failures — correctly labeled as project-internal)

The following claims in the component refer to the project's own implementation (`phase2/model.py`, `phase2/train.py`) and are not verifiable against the listed source files. They are correctly labeled as project-internal design decisions or intentional deviations, not as claims sourced from H-Net or DLCM:

1. **Top-M hard selection** (`phase2/model.py:242-243`) — The component correctly identifies this as an intentional deviation from H-Net's variable-count argmax threshold. Not a source claim; no verification required.

2. **Simplified quadratic compression loss** (`phase2/model.py:480`, `phase2/train.py:592`) — The component explicitly states this is a deviation from H-Net Eq. 10 and DLCM Eq. 10. Not a source claim.

3. **`learned_isolated` gradient isolation** — Explicitly stated to be an original project addition with no antecedent in either source. Consistent with source files confirming neither paper uses stop-gradient through the boundary predictor.

4. **`boundary_entropy` diagnostic metric** (`phase2/train.py:84`) — A monitoring utility defined within the project. Not claimed to derive from any source paper. Correctly labeled as project-internal.

5. **Verification checklist items 1–10** — These reference `phase2/model.py` and `phase2/train.py` line numbers. They are test specifications for the project's own code, not claims about source papers or reference code.

---

## Internal Contradictions

**None found.**

The following potential ambiguities were examined and resolved:

- **Index convention discrepancy (H-Net Eq. 4 vs. code):** As discussed in item 14 above, the paper equation and the code use different but equivalent index conventions. The component correctly notes this and correctly describes the net semantics. Not a contradiction.

- **DLCM Eq. 10 minimum value:** H-Net's ℒ_ratio has minimum 1.0; DLCM's ℒ_aux has minimum 0.0 (due to the −1 offset). The component correctly states both: "Minimum is 1.0 when F = G = 1/N" for H-Net, and "DLCM form (structurally identical, shifted by −1 so minimum is 0)" for DLCM. No contradiction.

- **`cosine_rule` description:** The component says the cosine rule mode is "W_q = W_k = I at all times, never updated." This is correctly distinguished from `learned_e2e` which starts at identity but learns. No contradiction with sources.

---

## Summary Table

| Item | Source | Status |
|---|---|---|
| H-Net Eq. 4 (boundary probability) | `hnet_2507.07955.md` lines 59–70 | VERIFIED (verbatim) |
| p₁ = 1.0 (H-Net) | `hnet_2507.07955.md` line 70 | VERIFIED (verbatim) |
| W_q/W_k identity init | `hnet_boundary.py` lines 79–83 | VERIFIED (verbatim) |
| `_no_reinit = True` | `hnet_boundary.py` lines 82–83 | VERIFIED (verbatim) |
| Parallel forward code (lines 113–123) | `hnet_boundary.py` lines 113–123 | VERIFIED (verbatim) |
| Boundary mask code (lines 130–134) | `hnet_boundary.py` lines 130–134 | VERIFIED (verbatim) |
| Step mode code (lines 169–192) | `hnet_boundary.py` lines 168–192 | VERIFIED (verbatim) |
| DLCM Eq. 6 (boundary probability) | `dlcm.md` lines 68–74 | VERIFIED (verbatim) |
| p₁ = 1 (DLCM) | `dlcm.md` lines 83, 245 | VERIFIED (verbatim) |
| Index convention difference (H-Net vs. DLCM) | `dlcm.md` lines 254–267 | VERIFIED |
| H-Net Eq. 10 (ratio loss) | `hnet_2507.07955.md` lines 173–190 | VERIFIED (verbatim) |
| α = 0.03 (H-Net) | `hnet_2507.07955.md` line 190 | VERIFIED (verbatim) |
| DLCM Eq. 10 (auxiliary loss) | `dlcm.md` lines 114–122 | VERIFIED (verbatim) |
| DLCM Eq. 10 minimum is 0 (−1 offset) | `dlcm.md` lines 120–122 | VERIFIED |
| DLCM Eq. 24 (gradient conflict) | `dlcm.md` lines 136–143 | VERIFIED (verbatim) |
| No stop-gradient in H-Net | `hnet_2507.07955.md` lines 211–215 | VERIFIED |
| No stop-gradient in DLCM | `dlcm.md` lines 136–143 | VERIFIED |
| H-Net Eq. 7 (STE) | `hnet_2507.07955.md` lines 135–142 | VERIFIED (verbatim) |
| Top-M deviation from H-Net | Project-internal | N/A (correctly labeled) |
| Quadratic loss deviation | Project-internal | N/A (correctly labeled) |
| `learned_isolated` as original addition | Project-internal, confirmed by sources | N/A (correctly labeled) |
