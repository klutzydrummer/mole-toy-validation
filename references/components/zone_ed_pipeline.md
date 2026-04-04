# Zone E / SimpleDecoder Pipeline (Phase 2 Outer Encoder Study)

## Component

**Zone E → BoundaryRouter → SimpleDecoder** — content-aware compression of L language tokens into M concept tokens and reconstruction of L-length representations for next-token prediction. No inner transformer. No Zone D CRL decoder. This design isolates concept token quality as the measurement target.

---

## Sources

**Papers:**
- `sources/papers/hnet_2507.07955.md` — H-Net: Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (Hwang, Wang, Gu; arXiv:2507.07955)
- `sources/papers/sombrero_2601.22805.md` — SOMBRERO: Measuring and Steering Boundary Placement (arXiv:2601.22805)
- `sources/papers/dlcm.md` — DLCM: Dynamic Length Compression Model (arXiv:2512.24617)

**Code:**
- `sources/code/hnet_boundary.py` — verbatim `dc.py` from `github.com/goombalab/hnet`, MIT license, retrieved 2026-03-17

---

## Authoritative equations

All equations below are from **H-Net (arXiv:2507.07955)**.

### Eq. 1 — Main processing flow

```
x̂ˢ = ℰˢ(xˢ),    ẑˢ = 𝓜(xˢ),    ẑˢ = 𝒟ˢ(zˢ)
```

In Phase 2's outer encoder study: `ℰˢ` = pluggable ZoneE encoder, `𝓜` = identity (no inner network — concept tokens ARE Zone E outputs at boundary positions), `𝒟ˢ` = SimpleDecoder.

### Eq. 2 — Chunking operation

```
(xˢ⁺¹, pˢ) = chunk(x̂ˢ)
```

The BoundaryRouter produces M concept tokens and boundary probabilities `p [B, L]`.

### Eq. 3 — Gated residual / dechunking with residual connection

```
zˢ = dechunk(ẑˢ⁺¹, pˢ) + linear(x̂ˢ)
```

"we adopt the first approach – adding a projection (linear) only to the residual connection."

**Critical detail — zero initialization:** "this residual connection is initialized close to 0; earlier versions of H-Net found this to be an important detail."

`linear` here is a plain `nn.Linear(d, d, bias=False)` initialized to zero weight (not sigmoid-gated, not p-modulated).

### Eq. 4 — Boundary probability computation (learned routing)

```
qₜ = W_q x̂ₜ
kₜ = W_k x̂ₜ
pₜ = (1/2)(1 − (qₜᵀ kₜ₋₁) / (‖qₜ‖ ‖kₜ₋₁‖)) ∈ [0, 1]
bₜ = 𝟙{pₜ ≥ 0.5}
```

"p₁ = 1.0 by definition, ensuring the sequence begins with a boundary."

W_q and W_k are initialized to the identity matrix.

### Eq. 5 — EMA smoothing

```
z̄ₜ = Pₜ ẑₜ + (1 − Pₜ) z̄ₜ₋₁
```

"The smoothing module applies an exponential moving average (EMA)." Runs over M concept token positions (not L). Pₜ is boundary probability at concept position i, clamped to [0.1, 1.0].

### Eq. 6 — Confidence scoring

```
cₜ = pₜ^(bₜ) (1 − pₜ)^(1−bₜ) = { pₜ      if bₜ = 1
                                   { 1 − pₜ  if bₜ = 0
```

"The coefficient c quantifies the routing module's confidence in its boundary decisions."

### Eq. 7 — Straight-Through Estimator (STE)

```
ste(cₜ) = cₜ + stopgradient(1 − cₜ)
```

"rounds confidence scores to 1.0 in the forward pass while maintaining continuous gradients during backpropagation." In forward pass: `ste(c_t) = 1.0` (all outputs are full-magnitude). In backward: gradient flows through `c_t` — a router that makes uncertain decisions (p≈0.5) receives a gradient signal pushing toward decisive choices.

### Eq. 8 — Causal plug-back

```
z̃ₜ = z̄_{∑ₖ₌₁ᵗ bₖ}
```

**Implementation (from `hnet_boundary.py`):**
```python
plug_back_idx = torch.cumsum(boundary_mask, dim=1) - 1  # (B, L)
out = torch.gather(out, dim=1,
                   index=plug_back_idx.unsqueeze(-1).expand(-1, -1, d_model))
```

### Eq. 9 — Upsampler output

```
upsampler(z̄, c)ₜ = ste(cₜ) · z̃ₜ
```

"Multiplying upsampled vectors by their confidence scores incentivizes the routing module to make confident, accurate decisions."

### Eq. 10 — Ratio loss (compression target)

```
ℒ_ratio = (N / (N−1)) · ((N−1) · F · G + (1−F) · (1−G))

F = (1/L) ∑ₜ₌₁ᴸ bₜ       (hard fraction selected as boundaries)
G = (1/L) ∑ₜ₌₁ᴸ pₜ       (average boundary probability)
```

Minimum = 1.0 when F = G = 1/N (both hard selection and soft probability match target). Combined objective: `ℒ = ℒ_AR + α ℒ_ratio`, α = 0.03.

---

## Reference implementation

All code below is **verbatim from `sources/code/hnet_boundary.py`** (goombalab/hnet `dc.py`, MIT license).

### RoutingModule — W_q / W_k identity initialization

```python
self.q_proj_layer = nn.Linear(d_model, d_model, bias=False)
self.k_proj_layer = nn.Linear(d_model, d_model, bias=False)
with torch.no_grad():
    self.q_proj_layer.weight.copy_(torch.eye(d_model))
    self.k_proj_layer.weight.copy_(torch.eye(d_model))
self.q_proj_layer.weight._no_reinit = True
self.k_proj_layer.weight._no_reinit = True
```

### RoutingModule — boundary probability computation

```python
cos_sim = torch.einsum(
    "b l d, b l d -> b l",
    F.normalize(self.q_proj_layer(hidden_states[:, :-1]), dim=-1),
    F.normalize(self.k_proj_layer(hidden_states[:, 1:]), dim=-1),
)
boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)
boundary_prob = F.pad(boundary_prob, (1, 0), "constant", 1.0)
```

### DeChunkLayer — plug-back (Eq. 8)

```python
plug_back_idx = torch.cumsum(boundary_mask, dim=1) - 1  # (B, L)
out = torch.gather(
    out,
    dim=1,
    index=plug_back_idx.unsqueeze(-1).expand(-1, -1, self.d_model),
)
```

---

## Our implementation

**File:** `phase2/model.py`

### Zone E — pluggable encoders

`phase2/model.py` — `CRLEncoder`, `TransformerEncoder`, `DiffAttnEncoder`, `MLAEncoder`, `IdentityEncoder`

All implement `forward(x: [B, L, d]) -> encoder_out [B, L, d]`. `CRLEncoder` uses down_proj → 3×CRL(d//4, log_a_init=3.0) → up_proj. Transformer variants use 4 TransformerBlock layers with `n_layers_outer=4`.

**Deviation — recurrence instead of Mamba:** H-Net uses Mamba-2 in encoder/decoder. Our CRLEncoder uses Griffin/Hawk RG-LRU. Same role (contextual encoding before boundary selection), different operator.

**Deviation — no width expansion:** H-Net requires monotonically non-decreasing widths across stages. Phase 2 outer uses a single width `d` throughout.

### BoundaryRouter

`phase2/model.py` — `BoundaryRouter.forward`

Three routing modes: `cosine_rule`, `learned_e2e`, `fixed_stride`. `cosine_rule` and `learned_e2e` force `p_0 = 1.0` via `F.pad(p, (1, 0), value=1.0)`. Variable M per sequence, padded to M_max with `concept_mask [B, M_max]` tracking valid entries.

**Deviation — threshold selection instead of hard topk:** H-Net uses `b_t = 𝟙{p_t ≥ 0.5}` with variable M. This implementation matches H-Net exactly (no topk).

**Deviation — W_q/W_k no_reinit via re-application:** H-Net sets `_no_reinit = True` on W_q/W_k weights to prevent downstream re-initialization. Our implementation re-applies `nn.init.eye_` after `self.apply(_init_weights)` in `OuterModel.__init__`. Same outcome, different mechanism.

### SimpleDecoder

`phase2/model.py` — `SimpleDecoder.forward`

Four steps:

1. **EMA smoothing (Eq. 5)** over M_max concept positions using `_parallel_scan`. `p_at_bounds.clamp(min=0.1)` prevents EMA collapse (see below).

2. **Plug-back (Eq. 8)** via `torch.cumsum(boundary_mask.long(), dim=1) - 1`. Maps each of L positions to the M index of its most recent boundary. Directly matches reference implementation.

3. **Confidence scoring + STE (Eq. 6–7–9):**
   ```python
   b_float = (boundary_probs >= 0.5).float()
   c_t     = b_float * boundary_probs + (1 - b_float) * (1 - boundary_probs)  # Eq. 6
   ste_c   = c_t + (1.0 - c_t).detach()                                       # Eq. 7
   upsampled = ste_c.unsqueeze(-1) * plugback                                  # Eq. 9
   ```

4. **Residual (Eq. 3):** `out = upsampled + self.residual_proj(encoder_out)` where `residual_proj = nn.Linear(d, d, bias=False)` initialized with `weight=0`.

### p_at_bounds.clamp(min=0.1) — EMA collapse prevention

`phase2/model.py` — `SimpleDecoder.forward`

If boundary_probs at selected positions are near zero early in training, the EMA degenerates: every `h_i ≈ h_0`. The `min=0.1` clamp ensures each selected boundary does at least a 10% blend toward its concept token.

H-Net avoids this by clamping `p` to `[1e-4, 1-1e-4]` before the scan. Our `min=0.1` is a stronger guard.

### Ratio loss (Eq. 10) — `ratio_loss()` in train.py

`phase2/train.py` — `ratio_loss()`

Implements H-Net Eq. 10 exactly. α = 0.03 (H-Net default). `outer_crl` and `outer_strided` use α = 0.

### Full forward pass (OuterModel)

`phase2/model.py` — `OuterModel.forward`

```
embed(x) [B, L, d]
  → encoder → encoder_out [B, L, d]
  → BoundaryRouter → concept_tokens [B, M_max, d], boundary_probs [B, L],
                     boundary_idx [B, M_max], concept_mask [B, M_max]
  → SimpleDecoder(concept_tokens, encoder_out, boundary_probs,
                  boundary_idx, concept_mask) → token_repr [B, L, d]
  → lm_head → logits [B, L, vocab_size]

Returns: (logits [B, L, V], boundary_probs [B, L], compression_ratio scalar)

Training:
  loss_ntp   = cross_entropy(logits, targets)
  loss_ratio = ratio_loss(boundary_probs, target_rate)  # H-Net Eq. 10
  loss       = loss_ntp + alpha * loss_ratio             # alpha=0.03
```

---

## Verification checklist

1. **Position-0 boundary for cosine_rule / learned_e2e.** `boundary_probs[:, 0] == 1.0` for all inputs in these modes. `cumsum` at position 0 = 1, so `plug_back_idx[:, 0] = 0` — maps to first (valid) concept token. Not applicable to `fixed_stride`.

2. **Variable M per sequence.** Assert that different sequences in the same batch can have different values of `concept_mask.sum(dim=1)`. Fixed-stride is the only mode with constant M.

3. **EMA initialized at h0 = concept_tokens[:, 0].** Verify that `smoothed[:, 0] == concept_tokens[:, 0]` (first concept always has p=1.0, so h̄_0 = concept_0 exactly).

4. **p_at_bounds.clamp(min=0.1) takes effect.** On a uniform-noise input (all positions similar, low boundary probability), confirm `p_at_bounds.min() >= 0.1` and `smoothed` is not collapsed to a single repeated vector.

5. **Plug-back correctness via cumsum.** For boundary_mask = [1,0,0,1,0,1,0,0,0] (B=1, L=9), cumsum = [1,1,1,2,2,3,3,3,3], plug_back_idx = [0,0,0,1,1,2,2,2,2]. Verify positions 0–2 gather smoothed[:,0], positions 3–4 gather smoothed[:,1], positions 5–8 gather smoothed[:,2].

6. **STE is identity in forward pass.** Assert `ste_c.allclose(torch.ones_like(ste_c))` (all 1.0 in forward, regardless of boundary_probs values).

7. **residual_proj weight=0 at init.** After `OuterModel.__init__`, assert `model.decoder.residual_proj.weight.abs().max() == 0.0`.

8. **W_q / W_k identity initialization survives _init_weights.** After `OuterModel.__init__`, assert `model.router.W_q.weight.allclose(torch.eye(d))` and same for W_k.

9. **Ratio loss minimum at target.** With `boundary_probs = torch.full([B, L], 1/N)` and `boundary_mask` also at rate 1/N, verify `ratio_loss ≈ 1.0` (the minimum value). With F=G=1/N: `(N/(N-1)) * ((N-1)*(1/N)*(1/N) + (1-1/N)*(1-1/N))` = 1.0.

10. **EMA collapse detection — MANDATORY PRE-FLIGHT CHECK.**
    `utils/smoke_test.py` must pass before any Phase 2 training run.
    Checks (updated for OuterModel API):
    - `encoder_out.var(dim=1).mean() > 1e-4` — Zone E positions are diverse.
    - `concept_tokens.var(dim=1).mean() > 1e-4` — concept tokens are diverse.
    - loss at steps 800-1000 < 0.90 × loss at steps 0-200 — model is learning.
    - No NaN/inf anywhere.

11. **Ratio loss drives compression toward target_rate.** Over training, `compression_ratio` (logged as `hdc/compression_ratio`) should converge toward `target_rate` (0.25). Persistent deviation indicates `alpha` or router gradient signal needs tuning.

12. **boundary_entropy decreases over training.** `hdc/boundary_entropy` should fall from ~0.693 (maximum, p≈0.5 everywhere) toward lower values as the router becomes more decisive. Per SOMBRERO (arXiv:2601.22805), low entropy is necessary but not sufficient — boundaries must also align with high-surprisal positions.
