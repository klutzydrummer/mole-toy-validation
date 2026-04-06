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
# hnet_boundary.py lines 168-192
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

**File pointer:** `phase2/components/boundary_router.py:15`
Class `BoundaryRouter` at line 15. The router is owned by `OuterModel` (not ZoneE) at `phase2/model.py:142`.

**Routing modes:**

| Mode name | Config key(s) | Description |
|---|---|---|
| `cosine_rule` | `outer_crl`, `outer_crl_full`, `outer_crl_r2` | No learned params. Raw cosine dissimilarity of encoder outputs. |
| `learned_e2e` | `outer_crl_learned`, `outer_crl_full_learned`, `outer_transformer`, `outer_diff_attn`, `outer_mla`, `outer_crl_learned_noste` | W_q/W_k learned; LM gradients flow through boundary_probs into the router. |
| `fixed_stride` | `outer_strided` | Selects evenly spaced positions. No signal. Lower-bound baseline. |

**Intentional deviations from H-Net reference:**

1. **Threshold selection (`p >= 0.5`), variable M per sequence instead of global topk.** H-Net uses `argmax(boundary_prob) == 1` (equivalent to `p >= 0.5`) to produce a variable number of boundary tokens per sequence. Our `BoundaryRouter` does the same: `boundary_mask = (boundary_probs >= 0.5)` (`phase2/components/boundary_router.py:111`). M varies per sequence; M_max is the maximum across the batch. Position 0 is always selected since `boundary_probs[:, 0] = 1.0 >= 0.5`. This matches the H-Net formulation exactly.

2. **Full H-Net ratio loss used (no deviation).** Our training loop uses the full H-Net Eq. 10 ratio loss via `ratio_loss()` (`phase2/train.py:95–101`, called at `phase2/train.py:333`):
   ```
   ℒ_ratio = (N / (N-1)) * ((N-1) * F * G + (1-F) * (1-G))
   F = (boundary_probs >= 0.5).float().mean()   # hard fraction selected
   G = boundary_probs.mean()                     # average soft probability
   N = 1 / target_rate
   ```
   This is the full F·G product structure from H-Net Eq. 10. There is no deviation from H-Net here.

3. **STE IS applied in SimpleDecoder (`use_ste=True` default).** H-Net wraps confidence scores with `ste(c) = c + stopgradient(1 − c)` (H-Net Eq. 7). Our `SimpleDecoder` (`phase2/components/zone_d.py:105–106`) applies the same STE by default. The ablation config `outer_crl_learned_noste` sets `use_ste=False` to test the effect of removing STE. The config key `use_ste` flows through `phase2/model.py` into `SimpleDecoder.__init__`.

4. **q/k index convention transposed relative to H-Net and DLCM.** H-Net computes `dot(q_{t-1}, k_t)` (q from the earlier token, k from the later token). DLCM uses the same convention. Our `learned_e2e` mode computes `dot(q_t, k_{t-1})` — q from the current (later) token, k from the prior (earlier) token (`phase2/components/boundary_router.py:102`):
   ```python
   sim = (q[:, 1:] * k[:, :-1]).sum(dim=-1)  # q_t · k_{t-1}
   ```
   At init (W_q=W_k=I) both conventions produce identical output because dot product is commutative for normalized vectors. After training, W_q specializes on the current token and W_k on the prior token — the reverse of both H-Net and DLCM. The cosine dissimilarity semantics are preserved; only the learned specialization of the two projections differs. **This deviation has not been ablated.**

5. **Cosine rule uses raw encoder outputs without W_q/W_k.** The `cosine_rule` mode normalizes encoder outputs directly without learned projections (`phase2/components/boundary_router.py:83–89`). This is equivalent to the H-Net RoutingModule with W_q = W_k = I frozen permanently. A strict subset of learned routing.

6. **`_no_reinit` omitted; replaced with post-init re-application.** The H-Net reference sets `_no_reinit = True` on W_q/W_k to prevent any downstream `_init_weights` sweep from overwriting the identity initialization. Our code instead initializes W_q/W_k to identity directly in `BoundaryRouter.__init__` (`phase2/components/boundary_router.py:51–52`) and then re-applies `nn.init.eye_` in `OuterModel.__init__` after the global `self.apply(_init_weights)` sweep (`phase2/model.py:157–159`). This achieves the same outcome — identity weights survive — via a different mechanism.

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

1. **Identity init preserved.** Confirm `BoundaryRouter.W_q.weight` and `W_k.weight` are `torch.eye(d)` immediately after construction for `learned_e2e` mode. Init at `phase2/components/boundary_router.py:51–52`; re-applied after `self.apply(_init_weights)` at `phase2/model.py:157–159`.

2. **p₀ = 1.0 always set.** In cosine_rule and learned_e2e modes, position 0 is padded with `value=1.0`. Confirm `boundary_probs[:, 0].all() == 1.0` for a freshly constructed model's first forward pass. (`phase2/components/boundary_router.py:89` for cosine_rule, `:104` for learned_e2e)

3. **Threshold selection (`p >= 0.5`), variable M per sequence.** `boundary_mask = (boundary_probs >= 0.5)` (`phase2/components/boundary_router.py:111`). M_max is the max boundary count across the batch. Position 0 is always selected since `boundary_probs[:, 0] = 1.0 >= 0.5`. Uses `>=` not `>` to match H-Net `argmax == 1` convention.

4. **`fixed_stride` has no parameters.** Confirm `len(list(model.router.parameters())) == 0` when `routing="fixed_stride"`. Router is at `model.router` (`phase2/model.py:142`).

5. **`learned_isolated` mode removed.** Only `cosine_rule`, `learned_e2e`, and `fixed_stride` are supported. (`phase2/components/boundary_router.py:41–53`)

6. **`cosine_rule` and `learned_e2e` produce identical output at init.** Immediately after construction (before any gradient steps), both modes produce the same `boundary_probs` for the same input. Both compute `normalize(enc_t) · normalize(enc_{t-1})` at init (W_q=W_k=I, dot product commutative). Verify numerically.

7. **`ratio_loss` shape and value.** `ratio_loss(bp, target_rate)` returns a scalar. At F=G=1/N the loss equals 1.0 (its minimum). (`phase2/train.py:95–101`, called at `:333`)

8. **`boundary_entropy` NaN guard.** Verify no NaN in `boundary_entropy()` when `boundary_probs` contains values exactly 0.0 or 1.0 (as produced by `fixed_stride`). The clamp at `1e-6` handles this. (`phase2/train.py:85–92`)

9. **Gradient flow in `learned_e2e`.** Confirm that `W_q.weight.grad` and `W_k.weight.grad` are non-None after a backward pass with `routing="learned_e2e"`. No `.detach()` in the learned_e2e branch. (`phase2/components/boundary_router.py:91–104`)

10. **Ratio loss IS used (H-Net Eq. 10).** Confirm the training loop uses `ratio_loss(bp, target_rate)` (H-Net Eq. 10, full F·G product form) added to `loss_ntp`. (`phase2/train.py:333–334`). The simple quadratic proxy `(bp.mean() - 1/R)**2` is NOT used.
