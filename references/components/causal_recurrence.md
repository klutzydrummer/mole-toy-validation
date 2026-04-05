# Component: causal_recurrence

**One-line description:** Real-Gated Linear Recurrent Unit (RG-LRU) — an input-dependent
gated linear recurrence with norm-preserving sqrt(1 − a_t²) scaling, used in Zones E and D
as lightweight sequence encoders/decoders operating at d/4 width.

---

## Sources

| Role | File |
|------|------|
| Paper | `sources/papers/griffin_2402.19427.md` — De et al. 2024, "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models", arXiv:2402.19427 |
| Reference code | `sources/code/griffin_rglru.py` — verbatim `RGLRU` and `rnn_scan` from `google-deepmind/recurrentgemma`, Apache 2.0 |

---

## Authoritative equations

All equations are from Griffin (arXiv:2402.19427), Section 2.4, unless noted.

**Eq. (1) — Recurrence gate:**
```
r_t = σ(W_a x_t + b_a)
```

**Eq. (2) — Input gate:**
```
i_t = σ(W_x x_t + b_x)
```

**Eq. (3) — Recurrent coefficient:**
```
a_t = a^(c · r_t)
```
where `a = σ(Λ)`, `Λ` is a learnable per-channel vector, and `c = 8` (scalar constant).

**Eq. (4) — Hidden state update:**
```
h_t = a_t ⊙ h_{t-1} + √(1 − a_t²) ⊙ (i_t ⊙ x_t)
```

**Eq. (6) — Log-space computation of a_t** (Appendix A, for numerical stability):
```
log a_t = log σ(Λ)^(c · r_t)
        = −c · softplus(Λ) ⊙ r_t
```
In code: `log_a = -8.0 * gate_a * softplus(a_param)`, then `a = exp(log_a)`.

**Norm-preservation identity** (Griffin paper, sqrt(1 − a_t²) section):
```
Var[h_t] = a_t² · Var[h_{t-1}] + (1 − a_t²) · Var[i_t ⊙ x_t] ≈ 1
```
The coefficient `√(1 − a_t²)` is chosen so that h_t has approximately unit variance when
both h_{t-1} and the gated input have unit variance.

---

## Reference implementation

Source: `sources/code/griffin_rglru.py`, Google DeepMind RecurrentGemma, Apache 2.0.
Copyright 2024 DeepMind Technologies Limited.

**Gate and coefficient computation (RGLRU.forward, lines 304–319 of source):**
```python
# Gates for x and a.
gate_x = torch.sigmoid(self.input_gate(x))
gate_a = torch.sigmoid(self.a_gate(x))

# Compute the parameter `A` of the recurrence.
log_a = -8.0 * gate_a * nn.functional.softplus(self.a_param)
a = torch.exp(log_a)
a_square = torch.exp(2 * log_a)

# Gate the input.
gated_x = x * gate_x

# Apply gamma normalization to the input. We need to clip the derivatives of
# `sqrt` in order to prevent NaNs during training in bfloat16.
multiplier = SqrtBoundDerivative.apply(1 - a_square)
multiplier = reset[..., None] + ~reset[..., None] * multiplier
normalized_x = gated_x * multiplier.type(x.dtype)
```

**Sequential scan kernel (rnn_scan, lines 153–157 of source):**
```python
for t in range(x.shape[1]):
    h_t = a[:, t].type(acc_dtype) * h_t + x[:, t].type(acc_dtype)
    y[:, t] = h_t.type(x.dtype)
```
The reference accumulates in `acc_dtype=torch.float32` regardless of input dtype.

**Learnable parameter initialization (rnn_param_init, lines 169–177 of source):**
```python
# Proportional to area in a ring.
tensor.uniform_(min_rad ** 2 + eps, max_rad ** 2 + eps)
tensor.log_().mul_(0.5)
# Inverse softplus transform.
return tensor.neg_().exp_().sub_(1.0).log_()
```
Called with `min_rad=0.9, max_rad=0.999`, making `a^c` uniform in [0.9, 0.999] at init.

**Gradient clipping on sqrt (SqrtBoundDerivative, lines 182–196 of source):**
```python
_MAX_SQRT_GRADIENT = 1000.0

class SqrtBoundDerivative(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sqrt(x)
    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        clipped_x_times_4 = torch.clip(4.0 * x, min=1 / (_MAX_SQRT_GRADIENT**2))
        return grad_output / torch.sqrt(clipped_x_times_4)
```

---

## Our implementation

**Primary class:** `phase2/components/causal_recurrence.py:51` — `CausalRecurrenceLayer`

**Parallel scan helper:** `phase2/components/causal_recurrence.py:16` — `_parallel_scan`

**Usage in Zone E:** `phase2/components/zone_e.py:31` — `nn.ModuleList([CausalRecurrenceLayer(d_inner, log_a_init=3.0) for _ in range(3)])` (CRLEncoder); `phase2/components/zone_e.py:58` (CRLEncoderFull)

**Usage in Zone D:** `phase2/components/zone_d.py:12` — imports `_parallel_scan` directly; no `CausalRecurrenceLayer` in Zone D

### Intentional deviations from reference

1. **Parallel scan instead of sequential loop.** The reference `rnn_scan` uses a Python
   `for t in range(L)` loop (O(L) kernel launches). Our `_parallel_scan` (`phase2/components/causal_recurrence.py:16`)
   uses the closed-form log-space prefix scan:
   ```
   log_A_t = cumsum(log(a_t), dim=1)
   h_t     = A_t * h0 + A_t * cumsum(b_t / A_t, dim=1)
   ```
   This is mathematically equivalent for h0=0 and reduces to ~6 tensor ops per instance,
   making it compilable by `torch.compile` into a single fused kernel graph. The derivation
   is in the docstring at `phase2/components/causal_recurrence.py:19–48`.

2. **log_a parameterization instead of softplus(a_param), with role-specific init.**
   The reference stores `a_param` and computes decay as `exp(-8 * gate_a * softplus(a_param))`,
   initialized via `rnn_param_init(min_rad=0.9, max_rad=0.999)` so that `a^c` is uniform
   in [0.9, 0.999] across channels. Our implementation stores `log_a` and computes
   `a_base = sigmoid(log_a)`, then `a_t = a_base^(8 * r_t)`.

   **Critical: log_a_init differs by role.** Using 7.5 everywhere (as originally implemented)
   set all channels to a_base≈0.9994, half-life≈289 steps, and gradient
   (1-sigmoid(7.5))≈0.0006 — effectively frozen. All encoder_out positions converge to the
   same running-average vector; the inner transformer is blind; the model collapses to
   unigram prediction (~9.7 BPC). This failure burned three 50k-step Phase 2 runs.

   Current init values:
   - Zone E: `log_a_init=3.0` → a_base≈0.953, half-life≈4 steps, gradient≈0.047 (`phase2/components/zone_e.py:32`)
   - Zone D: `log_a_init=0.0` → a_base=0.500, half-life<1 step,  gradient=0.250 (not used via CausalRecurrenceLayer; Zone D uses `_parallel_scan` directly)

   The softplus parameterization (gradient = sigmoid(a_param) ≈ 0.5–1.0 regardless of
   the decay value) is strictly better conditioned. This is a known remaining deviation
   to address after A1 validation passes.

3. **float32 promotion for sigmoid(log_a) only.** The reference promotes all accumulation
   to float32 inside `rnn_scan`. We promote only the `sigmoid(log_a)` computation to float32
   before casting back to working dtype (`phase2/components/causal_recurrence.py:111`). The parallel scan itself
   runs fully in float32 inside `_parallel_scan` (`phase2/components/causal_recurrence.py:41–48`). The net effect
   is the same: no float16 saturation in the decay path.

4. **sqrt clamped to 1e-6 instead of custom autograd function.** The reference uses
   `SqrtBoundDerivative` to clip the sqrt gradient at 1000.0. We instead clamp the
   argument before sqrt: `(1.0 - a_t * a_t).clamp(min=1e-6)` (`phase2/model.py:145`).
   This prevents the zero-argument NaN without a custom backward, which is simpler under
   `torch.compile`. The tradeoff is that we do not replicate the exact gradient magnitude
   cap of 1000.0; under normal operation with `a_t < 1`, the argument is bounded away from
   zero by the initialization, so this difference is not expected to matter in practice.

5. **Causal depthwise conv (kernel=4) prepended.** The reference RG-LRU does not include
   a convolution; Griffin's recurrent block wraps the RG-LRU in a temporal conv separately.
   Our `CausalRecurrenceLayer` folds the width-4 causal depthwise conv into the layer
   (`phase2/model.py:102–127`). Gates are computed from the post-conv activations, not raw
   x. This matches Griffin's recurrent block structure but makes CausalRecurrenceLayer
   self-contained.

6. **No segment boundary reset.** The reference multiplies `a` by `~reset` to zero out
   the recurrent state at document boundaries (`rnn_scan` line 134). Our implementation
   does not track segment positions; there is no reset mechanism. This is acceptable for
   the toy validation setting (single-document enwik8 slices).

7. **Output projection + RMSNorm.** The reference outputs `y = h_t` directly. Our layer
   appends `out_proj` (Linear d→d, no bias) and `RMSNorm` (`phase2/model.py:115–116,
   149`). This is an additional stabilization step not present in the Griffin RG-LRU.

---

## Verification checklist

1. **Gate computation:** Confirm `r_t = sigmoid(W_r(x_conv))` and `i_t = sigmoid(W_i(x_conv))`
   both produce values in (0, 1). Check `phase2/model.py:130–131`.

2. **Recurrent coefficient:** Confirm `a_t = sigmoid(log_a)^(8 * r_t)` produces values
   strictly in (0, 1). Zone E init: `sigmoid(3.0)^8 ≈ 0.68`. Zone D init: `sigmoid(0.0)^8
   = 0.004`. Both are in (0, 1). Confirm `log_a_init=3.0` in ZoneE.recurrence and
   `log_a_init=0.0` in ZoneD.recurrence (phase2/model.py lines ~281, ~398).

3. **Float32 promotion for sigmoid(log_a):** The float16 max less than 1.0 is 0.99951
   (exponent=-1, mantissa=1023: value = (1 + 1023/1024) * 0.5 = 2047/2048). sigmoid(7.5)
   = 0.99944 rounds to 0.99951 in float16 — NOT to 1.0. However, the code correctly
   computes in float32 regardless, which is the right practice. Confirm
   `torch.sigmoid(self.log_a.float()).to(x.dtype)` at `phase2/model.py:139`.

4. **Norm-preserving input term:** Confirm `b_t = sqrt((1 − a_t²).clamp(min=1e-6)) * (i_t * x_conv)`
   at `phase2/model.py:145`. The clamp prevents NaN; the sqrt term is Eq. (4)'s
   `√(1 − a_t²)` coefficient.

5. **Parallel scan correctness:** Verify `_parallel_scan(a_t, b_t)` with a known sequence.
   For L=2, h0=0: `h_1 = b_1`, `h_2 = a_2 * b_1 + b_2`. Confirm `phase2/model.py:71–76`.

6. **Float32 inside parallel scan:** Confirm `a = a_t.float()` and `b = b_t.float()` at
   `phase2/model.py:69–70`. Result cast back with `.to(a_t.dtype)` at line 76.

7. **log_A clamping:** Confirm `cumsum(log(a)).clamp(min=-80.0)` at `phase2/model.py:71`.
   Without the clamp, float32 can underflow to zero for long sequences with small a_t.

8. **Zone E 3-layer stack:** Confirm `ZoneE.recurrence` is `ModuleList` of 3
   `CausalRecurrenceLayer(d_outer)` at `phase2/model.py:281`. Each layer operates at
   `d_outer = d // 4` (default `d=256`, `d_outer=64`).

9. **Zone D 3-layer stack:** Same as Zone E — confirm `ZoneD.recurrence` at
   `phase2/model.py:398`.

10. **h_state dtype:** Confirm that hidden state accumulation inside `_parallel_scan` is
    in float32 and that the return is cast back to input dtype. Check that no float16 values
    saturate during a real training run by logging `a_t.min()` for the first 100 steps.
