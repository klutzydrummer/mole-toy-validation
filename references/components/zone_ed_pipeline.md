# Zone E / Zone D Pipeline

## Component

**Zone E → Inner → Zone D pipeline** — end-to-end hierarchical compression and reconstruction that maps L input tokens to M = L // R concept tokens, processes them through an inner network, then reconstructs L output representations via EMA smoothing, plug-back, and a gated residual skip from Zone E.

---

## Sources

**Papers:**
- `sources/papers/hnet_2507.07955.md` — H-Net: Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (Hwang, Wang, Gu; arXiv:2507.07955)
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

Encoder processing, main network computation, and decoder reconstruction across hierarchical stages. In our two-zone mapping: `ℰˢ` = Zone E, `𝓜` = InnerTransformer, `𝒟ˢ` = Zone D.

### Eq. 2 — Chunking operation

```
(xˢ⁺¹, pˢ) = chunk(x̂ˢ)
```

The chunking layer produces the downsampled concept tokens and boundary probabilities used by the dechunking stage.

### Eq. 3 — Gated residual / dechunking with residual connection

```
zˢ = dechunk(ẑˢ⁺¹, pˢ) + linear(x̂ˢ)
```

"we adopt the first approach – adding a projection (linear) only to the residual connection."

**Critical detail — zero initialization:** "this residual connection is initialized close to 0; earlier versions of H-Net found this to be an important detail, but it may be less important when combined with additional techniques such as LR modulation."

### Eq. 4 — Boundary probability computation (learned routing)

```
qₜ = W_q x̂ₜ
kₜ = W_k x̂ₜ
pₜ = (1/2)(1 − (qₜᵀ kₜ₋₁) / (‖qₜ‖ ‖kₜ₋₁‖)) ∈ [0, 1]
bₜ = 𝟙{pₜ ≥ 0.5}
```

"p₁ = 1.0 by definition, ensuring the sequence begins with a boundary."

W_q and W_k are initialized to the identity matrix; the `_no_reinit = True` flag prevents downstream re-initialization from overwriting this.

### Eq. 5 — EMA smoothing

```
z̄ₜ = Pₜ ẑₜ + (1 − Pₜ) z̄ₜ₋₁
```

"The smoothing module applies an exponential moving average (EMA) with the following definition." Pₜ is the boundary probability at position t; a high value (boundary) blends strongly toward the new concept token.

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

"The Straight-Through Estimator (STE)...rounds confidence scores to 1.0 in the forward pass while maintaining continuous gradients during backpropagation."

### Eq. 8 — Causal plug-back

```
z̃ₜ = z̄_{∑ₖ₌₁ᵗ bₖ}
```

"The upsampling operation repeats each compressed vector until the next boundary position, ensuring that each reconstructed position receives information from its most recent chunk."

### Eq. 9 — Upsampler output

```
upsampler(z̄, c)ₜ = ste(cₜ) · z̃ₜ
```

"Multiplying upsampled vectors by their confidence scores incentivizes the routing module to make confident, accurate decisions."

### Eq. 10 — Ratio loss (compression target)

```
ℒ_ratio = (N / (N−1)) · ((N−1) · F · G + (1−F) · (1−G))

F = (1/L) ∑ₜ₌₁ᴸ bₜ       (fraction selected as boundaries)
G = (1/L) ∑ₜ₌₁ᴸ pₜ       (average boundary probability)
```

Minimum value 1.0 when F = G = 1/N, where N is the target compression ratio. Combined objective: `ℒ = ℒ_AR + α ∑ₛ ℒ_ratio^s`, α = 0.03.

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

### RoutingModule — boundary probability computation (forward pass)

```python
# q from tokens [0..L-2], k from tokens [1..L-1]
cos_sim = torch.einsum(
    "b l d, b l d -> b l",
    F.normalize(self.q_proj_layer(hidden_states[:, :-1]), dim=-1),
    F.normalize(self.k_proj_layer(hidden_states[:, 1:]), dim=-1),
)
boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)

PAD_PROB = 1.0
boundary_prob = F.pad(boundary_prob, (1, 0), "constant", PAD_PROB)
```

### DeChunkLayer — EMA via Mamba-2 scan kernel (training)

```python
p = torch.clamp(boundary_prob[..., -1].float(), min=1e-4, max=1 - (1e-4))
# ...
dt = torch.log(1 / (1 - p)).to(self.dtype)
x = (hidden_states / dt[..., None]).to(self.dtype)
A = -torch.ones((self.nheads,), device=hidden_states.device, dtype=torch.float32)
b = p.to(self.dtype)
c = torch.ones_like(b)
out = mamba_chunk_scan_combined(...)
```

### DeChunkLayer — EMA step mode (inference)

```python
result = p * current_hidden_states + (1 - p) * inference_params.last_value
inference_params.last_value.copy_(result)
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

### Zone E (ZoneE class)

`phase2/model.py:258–313` — `ZoneE.forward`

Pipeline: `embed(x)` is done in `HDCModel.forward` before `ZoneE` is called. Inside `ZoneE`: `down_proj` (d → d/4) → 3× `CausalRecurrenceLayer(d/4)` → `up_proj` (d/4 → d) producing `encoder_out [B, L, d]` (the Zone D skip connection) → `BoundaryRouter` → `gather` to produce `concept_tokens [B, M, d]`.

**Deviation — recurrence instead of Mamba:** H-Net uses Mamba-2 layers in its encoder/decoder. Our Zone E and Zone D use `CausalRecurrenceLayer` (Griffin/Hawk RG-LRU). This is an intentional substitution to avoid the `mamba_ssm` dependency in the outer zones; the role (contextual encoding before boundary selection) is the same.

**Deviation — no width expansion:** H-Net requires monotonically non-decreasing widths D⁰ ≤ D¹ ≤ ... and appends a shared trainable vector to expand dimensions. Our outer/inner widths are both d=256; no dimension change occurs at zone boundaries.

**Deviation — dense re-indexing for inner positions:** Concept tokens use positions 0, 1, ..., M-1 for RoPE inside InnerTransformer. Sparse original-position indexing (using the actual boundary positions) is left as ablation A-new.

**Deviation — 3 recurrence layers instead of H-Net's 4:** Zone E and Zone D each use 3× `CausalRecurrenceLayer`. H-Net uses 4 layers in its encoder/decoder stages. This is an intentional reduction for parameter budget reasons at the 5M toy scale.

### BoundaryRouter (boundary selection)

`phase2/model.py:156–251` — `BoundaryRouter.forward`

Three routing modes: `cosine_rule` (no learned params), `learned_e2e` (H-Net style), `fixed_stride` (lower bound). In all modes position 0 is forced to p=1.0 (no predecessor, always a boundary). Selection uses `topk(M)` rather than H-Net's thresholding at 0.5, ensuring exactly M = L // R concept tokens are selected regardless of the learned distribution.

**Deviation — top-M instead of threshold:** H-Net selects boundaries by `argmax(boundary_prob) == 1` (soft threshold at 0.5), which yields a variable number of tokens per sequence. We use `topk(M)` to guarantee exactly M = L // R tokens, because our inner transformer requires a fixed sequence length.

### Zone D (ZoneD class)

`phase2/model.py:370–464` — `ZoneD.forward`

Four steps:

1. **EMA smoothing** (`phase2/model.py:426–436`): Implements Eq. 5. Uses `_parallel_scan` rather than the Mamba-2 scan kernel, for the same reason as Zone E (no mamba_ssm dependency).

2. **Plug-back** (`phase2/model.py:444–451`): Implements Eq. 8 via `torch.searchsorted` on `boundary_idx` rather than `cumsum(boundary_mask)`. Semantically equivalent: both map each position j to the index of the most recent boundary.

3. **Gated residual** (`phase2/model.py:453–456`): Implements Eq. 3 with a refinement: the residual weight is `(1 - p_j) * sigmoid(W_gate * encoder_out_j)` rather than a plain `linear`. Non-boundary positions (low p) lean more heavily on Zone E's encoder_out, preserving fine-grained local information not seen by the inner network. This addresses the gradient conflict described in DLCM (arXiv:2512.24617, Eq. 24): at non-boundary positions, the model has already received CE signal to compress; the gated residual provides a low-resistance path for these positions without requiring additional auxiliary loss terms.

4. **Decoder recurrence** (`phase2/model.py:458–463`): d → d/4 → 3× `CausalRecurrenceLayer` → d/4 → d → RMSNorm.

### p_at_bounds.clamp(min=0.1) — EMA collapse prevention

`phase2/model.py:426`

```python
p_at_bounds = boundary_probs.gather(1, boundary_idx).clamp(min=0.1)  # [B, M]
```

This is an intentional deviation from H-Net. If `boundary_probs` at selected positions are near zero (possible early in training with the cosine router on low-contrast encoder outputs), the EMA degenerates: every `h_i ≈ h_0`, collapsing all M concept positions to the first token and making the inner transformer invisible. The `min=0.1` clamp ensures each selected boundary does at least a 10% blend toward its concept token, preventing EMA collapse.

H-Net avoids this by construction: its `p` values fed to the EMA are clamped to `[1e-4, 1 - 1e-4]` (see `DeChunkLayer.forward`, line `p = torch.clamp(boundary_prob[..., -1].float(), min=1e-4, max=1-(1e-4))`), but that only prevents exactly-zero values. Our `min=0.1` is a stronger guard applied specifically at the M selected positions.

### Full forward pass (HDCModel)

`phase2/model.py:584–611` — `HDCModel.forward`

```
embed(x) [B,L,d]
  → ZoneE → concept_tokens [B,M,d], encoder_out [B,L,d],
             boundary_probs [B,L], boundary_idx [B,M]
  → InnerTransformer → concept_out [B,M,d]
  → ZoneD(concept_out, encoder_out, boundary_probs, boundary_idx) → token_repr [B,L,d]
  → lm_head → logits [B,L,vocab_size]
```

Training loss: `loss = loss_ntp + lambda_comp * (boundary_probs.mean() - 1/R)^2`

The `loss_comp` term is our simplified form of H-Net's Eq. 10 ratio loss. H-Net's exact form involves both F (fraction selected) and G (average probability); our form uses only G (average probability), which is sufficient for the top-M selection strategy.

---

## Verification checklist

1. **Position-0 boundary is always 1.0.** In `BoundaryRouter.forward` (`cosine_rule` and `learned_e2e` modes), verify `F.pad(p, (1, 0), value=1.0)` produces `boundary_probs[:, 0] == 1.0` for all inputs. Confirm that position 0 always appears in `boundary_idx` (it always wins `topk` because 1.0 is the maximum possible value).

2. **Top-M selection yields exactly M = L // R tokens.** Assert `boundary_idx.shape[1] == L // R` after `BoundaryRouter.forward` for sequences of length L. Test at L=512, R=4 (M=128) and at L=512, R=8 (M=64).

3. **EMA recurrence is correct at position 0.** Verify that `ZoneD.forward` initializes `h0 = concept_out[:, 0]` (not zeros), reflecting that the first concept token is always a boundary (p=1.0, so `h̄_0 = 1.0 * concept_0`). If zeros were used, every sequence would begin with a zero hidden state passed to plug-back.

4. **p_at_bounds.clamp(min=0.1) takes effect.** Run `hdc_rulebased` on a uniform-noise input sequence where cosine similarity is high (low-contrast). Confirm `p_at_bounds.min() >= 0.1` and that `smoothed` is not collapsed to a single repeated vector.

5. **Plug-back correctness.** For a known `boundary_idx = [0, 4, 9]` (B=1, M=3, L=12), verify that positions 0–3 map to bucket 0, positions 4–8 map to bucket 1, and positions 9–11 map to bucket 2, using `torch.searchsorted` with `right=True`.

6. **Gated residual weighting.** At a known boundary position j where `boundary_probs[:,j] ≈ 1.0`, verify that `(1 - p_expanded) * gate * encoder_out` ≈ 0, so the output is dominated by `plugback`. At a non-boundary position where `boundary_probs[:,j] ≈ 0.0`, verify the output is dominated by the gated encoder_out residual.

7. **W_q / W_k identity initialization survives _init_weights.** In `HDCModel.__init__`, `_init_weights` runs `nn.init.normal_` on all `nn.Linear` layers, which would overwrite the identity init. Verify that the explicit `nn.init.eye_` re-application at lines 568–571 restores the correct identity weights after `self.apply(self._init_weights)`.

8. **Compression ratio R controls M correctly.** For configs `hdc_r2` (R=2), `hdc_r4` (R=4), `hdc_r8` (R=8) at sequence length L=512, verify InnerTransformer receives inputs of shape `[B, 256, d]`, `[B, 128, d]`, `[B, 64, d]` respectively.

9. **boundary_probs_for_zd is detached only for isolated routing.** In `hdc_e2e_isolated`, confirm `boundary_probs_for_zd` is the detached tensor and that `torch.autograd.grad` from `loss_ntp` does not reach `BoundaryRouter.W_q` or `BoundaryRouter.W_k` through Zone D. In all other configs confirm gradients do flow.

10. **EMA collapse detection.** After training for 1000 steps with `hdc_rulebased`, compute the variance of `smoothed` vectors across the M dimension. If variance < 1e-4 (all concept positions have collapsed to nearly the same vector), the `clamp(min=0.1)` is not effective and the encoder outputs may be too uniform. This indicates Zone E is not producing meaningful boundary signal.

11. **Loss_comp drives boundary_probs toward 1/R.** Over training, log `boundary_probs.mean()`. It should converge toward `1/R` (0.25 for R=4). Large persistent deviation indicates `lambda_comp` needs tuning or the router is not receiving sufficient gradient signal.
