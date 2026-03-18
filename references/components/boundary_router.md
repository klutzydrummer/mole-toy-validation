# Component: BoundaryRouter

## Component

**Name:** BoundaryRouter
**Description:** Produces soft boundary probabilities from encoder outputs and selects M = L // R concept-token positions via one of three routing strategies (learned end-to-end, cosine rule, or fixed stride).

---

## Sources

**Papers:**
- `sources/papers/hnet_2507.07955.md` — H-Net (arXiv:2507.07955), Hwang et al. 2025. Primary source for boundary probability formula, identity init, p₁=1.0, ratio loss.
- `sources/papers/dlcm.md` — DLCM (arXiv:2512.24617), Qu et al. 2025. Independent confirmation of the cosine-dissimilarity formula and gradient conflict analysis.

**Code:**
- `sources/code/hnet_boundary.py` — H-Net official `RoutingModule` and `DeChunkLayer` from `goombalab/hnet/hnet/modules/dc.py` (MIT license, verbatim).

---

## Authoritative equations

### H-Net Eq. (4) — Boundary probability (learned routing)

Source: H-Net (arXiv:2507.07955), Equation 4.

```
qₜ = W_q x̂ₜ
kₜ = W_k x̂ₜ
pₜ = (1/2)(1 − (qₜᵀ kₜ₋₁) / (‖qₜ‖ ‖kₜ₋₁‖)) ∈ [0, 1]
bₜ = 𝟙{pₜ ≥ 0.5}
```

`p₁ = 1.0` by definition (sequence always begins with a boundary).
W_q and W_k are initialized to the identity matrix so that at init the router computes raw cosine similarity with no learned transformation applied.

### DLCM Eq. (6) — Boundary probability (independent formulation)

Source: DLCM (arXiv:2512.24617), Equation 6.

```
pₜ = (1 − cos(qₜ₋₁, kₜ)) / 2 = (1/2)(1 − (qₜ₋₁ᵀ kₜ) / (‖qₜ₋₁‖₂ ‖kₜ‖₂))
```

Index convention differs from H-Net by one position (q from t−1, k from t vs. H-Net's q from t, k from t−1), but both measure the same adjacent-token cosine dissimilarity. DLCM also enforces `p₁ = 1`.

### H-Net Eq. (10) / DLCM Eq. (10) — Ratio / auxiliary loss

Source: H-Net (arXiv:2507.07955), Equation 10; DLCM (arXiv:2512.24617), Equation 10.

H-Net form:
```
ℒ_ratio = (N / (N−1)) · ((N−1) · F · G + (1−F) · (1−G))

F = (1/L) ∑ₜ bₜ       (fraction of tokens selected as boundaries)
G = (1/L) ∑ₜ pₜ       (mean boundary probability)
```
Minimum is 1.0 when F = G = 1/N. α = 0.03 in all H-Net experiments.

DLCM form (structurally identical, shifted by −1 so minimum is 0):
```
ℒ_aux = (R / (R−1)) · [(R−1) · F_global · G_global + (1−F_global) · (1−G_global)] − 1
```

**Our implementation uses a simpler quadratic proxy** (see "Our implementation" section), not the full H-Net/DLCM ratio loss.

### DLCM — Gradient conflict

Source: DLCM (arXiv:2512.24617), Equation 24.

```
∇_θ ℒ_total = ∇_θ ℒ_CE        +    λ · ∇_θ ℒ_aux
               (anti-compression)     (pro-compression)
```

The LM loss and the compression regularizer pull the boundary predictor in opposite directions. DLCM names this "Gradient Conflict in Learned Boundary Prediction" but does not apply a stop-gradient to resolve it. H-Net likewise has no stop-gradient through the boundary probability computation — gradient isolation is our own addition (see `learned_isolated` mode below).

---

## Reference implementation

Source: `sources/code/hnet_boundary.py`, verbatim from `goombalab/hnet` (MIT license).

**Identity initialization of W_q and W_k:**

```python
# hnet_boundary.py lines 79-83
with torch.no_grad():
    self.q_proj_layer.weight.copy_(torch.eye(d_model))
    self.k_proj_layer.weight.copy_(torch.eye(d_model))
self.q_proj_layer.weight._no_reinit = True
self.k_proj_layer.weight._no_reinit = True
```

`_no_reinit = True` prevents any downstream initialization sweep from overwriting the identity.

**Boundary probability computation (training, parallel mode):**

```python
# hnet_boundary.py lines 113-123
cos_sim = torch.einsum(
    "b l d, b l d -> b l",
    F.normalize(self.q_proj_layer(hidden_states[:, :-1]), dim=-1),
    F.normalize(self.k_proj_layer(hidden_states[:, 1:]), dim=-1),
)
# this clamp should no-op as long as no precision issues are encountered
boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)

# Force boundary probability of the first element to 1.0
PAD_PROB = 1.0
boundary_prob = F.pad(boundary_prob, (1, 0), "constant", PAD_PROB)
```

Note: q is from tokens `[0..L-2]`, k is from tokens `[1..L-1]`. After padding, `boundary_prob[t]` measures dissimilarity between token `t−1` and token `t`.

**Boundary mask (threshold at 0.5):**

```python
# hnet_boundary.py lines 130-134
boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)
selected_idx = torch.argmax(boundary_prob, dim=-1)
boundary_mask = selected_idx == 1
```

**Step (autoregressive inference):**

```python
# hnet_boundary.py lines 169-192
cos_sim = torch.einsum(
    "b d, b d -> b",
    F.normalize(self.q_proj_layer(inference_params.last_hidden_state), dim=-1),
    F.normalize(self.k_proj_layer(hidden_states), dim=-1),
)
boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)
inference_params.last_hidden_state.copy_(hidden_states)
boundary_prob = torch.where(
    inference_params.has_seen_tokens,
    boundary_prob,
    torch.ones_like(boundary_prob),
)
```

The step mode compares the current token's k-projection against the stored last hidden state's q-projection, enforcing `p=1.0` for the very first token seen.

---

## Our implementation

**File pointer:** `/home/brandon/Projects/toy-validation/phase2/model.py:156`
Class `BoundaryRouter` at line 156; `ZoneE` (which owns the router) at line 258.

**Routing modes:**

| Mode name | Config key(s) | Description |
|---|---|---|
| `cosine_rule` | `hdc_rulebased` | No learned params. Raw cosine dissimilarity of encoder outputs. |
| `learned_e2e` | `hdc_gate`, `hdc_r2`, `hdc_r8`, upcycle variants | W_q/W_k learned; LM gradients flow through boundary_probs into the router. |
| `fixed_stride` | `hdc_stride` | Selects every R-th token. No signal. Lower-bound baseline. |
| `learned_isolated` | `hdc_e2e_isolated` (A5 ablation only) | Same as `learned_e2e` but boundary_probs is detached before Zone D. Router only receives gradient from `loss_comp`. |

**Intentional deviations from H-Net reference:**

1. **Top-M hard selection instead of soft argmax threshold.** H-Net uses `argmax(boundary_prob)` to select a variable number of boundary tokens. Our `BoundaryRouter` always selects exactly `M = L // R` tokens via `boundary_probs.topk(M)`. This enforces a fixed compression ratio per batch rather than a stochastic one, trading dynamic compression for training stability.

2. **Simplified compression loss.** H-Net uses the full ratio loss (Eq. 10 above). Our training loop uses a simpler quadratic proxy (`phase2/model.py:480`, `phase2/train.py:592`):
   ```
   loss_comp = (boundary_probs.mean() - 1/R)²
   ```
   This penalizes deviation of the mean boundary probability from the target rate. It lacks the F·G product structure of the H-Net/DLCM loss but is easier to tune. `lambda_comp` is a hyperparameter passed at train time.

3. **No STE (Straight-Through Estimator).** H-Net wraps confidence scores with `ste(c) = c + stopgradient(1 − c)` (H-Net Eq. 7) to round forward while preserving continuous gradients. We do not apply STE because top-M selection is not used as a multiplicative weight on the token representations; the boundary_probs themselves flow into Zone D's EMA without discretization.

4. **Cosine rule uses raw encoder outputs without W_q/W_k.** The `cosine_rule` mode (`hnet_boundary.py` equivalent would be W_q = W_k = I at all times, never updated). This is a strict subset of learned routing with frozen identity projections.

5. **`learned_isolated` gradient isolation (our addition).** Neither H-Net nor DLCM propose stop-gradient through the boundary probability. This mode is an original ablation (A5) to test whether end-to-end LM gradients are necessary for learning useful boundaries, or whether the compression loss alone is sufficient. Expected result: near-random boundaries, confirming that LM signal is required.

**`boundary_entropy` diagnostic metric** (`phase2/train.py:84`):

```python
def boundary_entropy(boundary_probs: torch.Tensor) -> float:
    p = boundary_probs.detach().float().clamp(1e-6, 1.0 - 1e-6)
    ent = -(p * p.log() + (1 - p) * (1 - p).log()).mean().item()
    return ent
```

This is the mean binary cross-entropy of the boundary probability distribution, treated as a Bernoulli variable at each position. It is a monitoring metric only — not used in the training loss. High entropy means the router is uncertain and produces near-uniform probabilities (0.5 everywhere). Low entropy means the router is confident, with probabilities near 0 or 1. Decreasing `boundary_entropy` during training is a success criterion for A1 (see `CLAUDE.md:199`). The clamp uses `1e-6` rather than `1e-8` to avoid float32 ULP aliasing at `p=1.0` (see code comment at `train.py:86`).

---

## Verification checklist

1. **Identity init preserved.** Confirm `BoundaryRouter.W_q.weight` and `W_k.weight` are `torch.eye(d)` immediately after construction for `learned_e2e` and `learned_isolated` modes. (`phase2/model.py:195-196`)

2. **p₁ = 1.0 always set.** In all three non-stride modes, position 0 is padded with `value=1.0`. Confirm `boundary_probs[:, 0].all() == 1.0` for a freshly constructed model's first forward pass. (`phase2/model.py:230, 238`)

3. **Top-M selects exactly M tokens.** Check `boundary_idx.shape[1] == seq_len // R` for all routing modes. Confirm position 0 is always in `boundary_idx` (since `boundary_probs[:, 0] = 1.0` always wins topk). (`phase2/model.py:242-243`)

4. **`fixed_stride` has no parameters.** Confirm `len(list(model.zone_e.router.parameters())) == 0` when `routing="fixed_stride"`. (`phase2/model.py:211-217`)

5. **`learned_isolated` detaches before Zone D.** Confirm that `boundary_probs_for_zd.requires_grad == False` when routing is `learned_isolated`, while `boundary_probs.requires_grad == True` (pre-detach). (`phase2/model.py:247`)

6. **`cosine_rule` and `learned_e2e` produce identical output at init.** Immediately after construction (before any gradient steps), both modes should produce the same `boundary_probs` for the same input, since `learned_e2e` starts with identity projections. Verify numerically.

7. **`loss_comp` shape and value.** `loss_comp = (boundary_probs.mean() - 1/R)**2`. At R=4, target mean is 0.25. Confirm this is a scalar and its value is near 0 when the router correctly selects 1/4 of positions. (`phase2/train.py:592`)

8. **`boundary_entropy` NaN guard.** Verify no NaN in `boundary_entropy()` when `boundary_probs` contains values exactly 0.0 or 1.0 (as produced by `fixed_stride`). The clamp at `1e-6` should handle this. (`phase2/train.py:89`)

9. **Gradient flow in `learned_e2e`.** Confirm that `W_q.weight.grad` and `W_k.weight.grad` are non-None after a backward pass with `routing="learned_e2e"`. Confirm they are None with `routing="learned_isolated"` (only `loss_comp` contributes, but `loss_comp` uses `boundary_probs` pre-detach — verify `loss_comp` does use the pre-detach `boundary_probs`, not `boundary_probs_for_zd`). (`phase2/model.py:589-593`)

10. **Ratio loss not used.** Confirm the training loop does not use H-Net Eq. 10 or DLCM Eq. 10. Only `loss_comp = (bp.mean() - 1/R)**2` is added to `loss_ntp`. This is an intentional deviation from both reference papers. (`phase2/train.py:592-593`)
