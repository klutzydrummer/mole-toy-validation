# Component: nGPT (Normalized Transformer — Hypersphere Representations)

## Component

**Name:** nGPT
**Classes:** `NGPTCausalAttention`, `NGPTMLACausalAttention`, `NGPTDiffCausalAttention`,
`l2_norm`, `normalize_ngpt_weights` in `phase1/components/ngpt.py`

**Description:** Hyperspherical variant of the transformer where all hidden states and
weight matrices are constrained to unit L2-norm. Hidden states travel on S^{d-1} through
learnable LERP interpolation steps ("eigen learning rates" α). Q and K are normalized per
head after RoPE; attention scale is inverted (s_qk replaces 1/√d_h). All weight matrices
are row-normalized after each optimizer step. Three attention variants: standard, MLA, and
DiffAttn — each mirroring the corresponding non-nGPT class with sphere adaptations.

The update rule and α parameters live in `TransformerBlock._forward_ngpt` (wiring, not here).

---

## Sources

**Primary paper:**
- `sources/papers/ngpt_2410.01131.md`
  — "nGPT: Normalized Transformer with Representation Learning on the Hypersphere"
  — Loshchilov et al., NVIDIA, arXiv:2410.01131, October 2024 (v2: April 2025)
  — Equations 10–11 define the spherical update rule used in `_forward_ngpt`
  — Section 2.2 defines Q/K normalization and inverted attention scale
  — Section 2.3 defines weight normalization
  — Section 2.5 defines logit scale s_z

**Reference implementation:**
- `sources/code/ngpt_reference.py`
  — Extracted from https://github.com/NVIDIA/ngpt (official NVIDIA implementation)
  — `justnorm` function, alpha-based update pattern, weight normalization loop

**Related (novel compositions — no published direct precedent):**
- `sources/papers/shc_2603.20896.md` — spectral-sphere-constrained mHC (closest to ngpt_mhc_a)
- `sources/papers/product_manifold_2412.07033.md` — product manifold attention in tangent space
- `sources/papers/jpmhc_2602.18308.md` — orthogonal hyper-connections via Stiefel manifold

---

## Authoritative equations

All equations verbatim from **arXiv:2410.01131** unless otherwise noted.

### Spherical update rule (Eq. 10–11)

```
h_sub = L2Norm(F(h))                     # sublayer output normalized
h     = L2Norm(h + α ⊙ (h_sub − h))     # LERP then renorm (Eq. 11)
```

- `α ∈ ℝ^d` — per-dim learnable, stored with init 0.05
- Applied with scale 1/√d: `α_eff = α * (1/√d)` — effective init ≈ 0.0022 at d=512
- `L2Norm(x) = x / ||x||₂` along last dim

### Q/K normalization (Section 2.2)

```
q = L2Norm(q)    # per head, after RoPE, shape [B, nh, L, d_h]
k = L2Norm(k)    # per head, after RoPE
```

Attention logits become cosine similarities in [−1, 1]. To restore softmax temperature:
```
attention = softmax(q * s_qk @ k.T) @ v
```
`s_qk` is learnable (init √d_h), applied as `q * s_qk` with `scale=1.0` in SDPA so
gradient flows through `s_qk`.

### Weight normalization (Section 2.3)

After each optimizer step, normalize all weight rows:
```python
w = F.normalize(w, dim=-1)    # row normalization (dim=-1 = last dim of 2D weight)
```
Covers all `nn.Linear` and `nn.Embedding` weight matrices. Tied parameters (embed/lm_head)
normalized once via seen-id set.

### Logit scale (Section 2.5)

```
logits = lm_head(h) * s_z
```
`s_z` learnable scalar, init √d ≈ 22.6. Needed because embed.weight is unit-norm →
raw logits are cosine similarities (range ≈ [−1, 1]); s_z restores softmax temperature.

---

## Our implementation

### l2_norm (ngpt.py:31–33)
```python
def l2_norm(x):
    return F.normalize(x, dim=-1)
```

### NGPTCausalAttention (ngpt.py:36–81)
Mirrors `CausalSelfAttention`. After RoPE:
```python
q = F.normalize(q, dim=-1)    # line ~76
k = F.normalize(k, dim=-1)
out = F.scaled_dot_product_attention(q * self.s_qk, k, v, is_causal=True, scale=1.0)
```
`self.s_qk = nn.Parameter(torch.tensor(math.sqrt(d // n_heads)))` — init √d_h.

### NGPTMLACausalAttention (ngpt.py:84–170)
Mirrors `MLACausalAttention` with two key differences:
1. `kv_norm` and `q_norm` removed — input is already unit-norm in nGPT forward path
2. Full concatenated `q = [q_c; q_rope]` and `k = [k_c; k_rope]` normalized per head:
   ```python
   q = torch.cat([q_c, q_rope], dim=-1)   # [B, nh, L, d_h + d_h_R]
   k = torch.cat([k_c, k_rope], dim=-1)
   q = F.normalize(q, dim=-1)
   k = F.normalize(k, dim=-1)
   ```
   `self.s_qk = nn.Parameter(torch.tensor(math.sqrt(d_head + d_h_R)))` — init √(d_h + d_h_R).

### NGPTDiffCausalAttention (ngpt.py:173–240)
Mirrors `DifferentialCausalAttention`. Q has shape [B, 2nh, L, d_h] — each of 2nh
half-heads normalized independently (F.normalize along dim=-1 normalizes each half-head):
```python
q = F.normalize(q, dim=-1)    # [B, 2nh, L, d_h] — each half-head on sphere
k = F.normalize(k, dim=-1)    # [B, nh,  L, d_h]
attn = F.scaled_dot_product_attention(q * self.s_qk, k_rep, v_rep, is_causal=True, scale=1.0)
```
λ computation unchanged — it is a scalar blend, not a projection needing sphere constraint.

### normalize_ngpt_weights (ngpt.py:243–263)
```python
@torch.no_grad()
def normalize_ngpt_weights(model):
    seen = set()
    for m in model.modules():
        if isinstance(m, nn.Linear | nn.Embedding):
            w = m.weight
            if id(w) not in seen:
                seen.add(id(w))
                w.data.copy_(F.normalize(w.data, dim=-1))
```

---

## Intentional deviations from paper

| Deviation | Paper | Our impl | Reason |
|-----------|-------|----------|--------|
| norm1/norm2 kept | Paper removes LayerNorm entirely | RMSNorm kept in TransformerBlock, not called in `_forward_ngpt` | Avoids conditional `__init__`; cost is 2×[d] params per block |
| norm_out kept | Paper removes final norm | Kept in ToyTransformer, functionally identity on unit-norm input | Compatibility — no measurable effect |
| Warmup kept | Paper removes warmup | 1k-step warmup retained | Controlled comparison; warmup shouldn't hurt |
| `_ngpt_normalize` flag approach NOT used | N/A | Uses module-type check (`nn.Linear | nn.Embedding`) | Simpler; no risk of missing a module |
| NGPTMLACausalAttention: no kv_norm/q_norm | N/A (our MLA addition) | Removed: input already unit-norm → RMSNorm becomes scalar | Documented in class docstring |

---

## Novel compositions (no published precedent)

### ngpt_mhc_a (Option A: multi-sphere)

Each of n=4 mHC streams lives on S^{d-1}. After H_res mixing, each stream is renormed:
```
mixed_i = L2Norm(Σ_j H_res[i,j] * x_j)   # Fréchet mean approximation
h = L2Norm(branch_fn(H_pre · x))           # branch output on sphere
x_i = L2Norm(mixed_i + α_i·post_i·(h − mixed_i))  # per-stream α interp
```
Per-stream α: `[n_streams, d]` instead of `[d]`. See `_forward_ngpt_mhc_a` in
`transformer_block.py`.

Closest published precedents:
- sHC (arXiv:2603.20896): sphere constraint on H_res itself (not hidden states)
- PM-Transformer (arXiv:2412.07033): product manifold attention in tangent space

### ngpt_mhc_c (Option C: nGPT-around-sublayer)

Full mHC block treated as one sublayer from nGPT's perspective:
```
x_new = hc(x, lambda inp: l2_norm(branch_fn(inp)))  # branch output L2Normed
h = L2Norm(x_new)                                    # full block output renormed
x = L2Norm(x + α * (h − x))                         # per-stream α interp
```
Simpler composition; sphere enforced once after the mHC block rather than per-stream.
See `_forward_ngpt_mhc_c` in `transformer_block.py`.

---

## Verification checklist

- [ ] `l2_norm` uses `F.normalize(x, dim=-1)` — normalizes along last dim
- [ ] `NGPTCausalAttention`: Q and K normalized per head after RoPE, before SDPA
- [ ] `NGPTCausalAttention`: SDPA called with `scale=1.0`, scale applied as `q * s_qk`
- [ ] `NGPTCausalAttention`: `s_qk` init = `sqrt(d // n_heads)`
- [ ] `NGPTMLACausalAttention`: no kv_norm or q_norm (input already unit-norm)
- [ ] `NGPTMLACausalAttention`: full `[q_c; q_rope]` normalized per head (not parts separately)
- [ ] `NGPTMLACausalAttention`: `s_qk` init = `sqrt(d_head + d_h_R)`
- [ ] `NGPTDiffCausalAttention`: each of 2nh Q half-heads normalized independently
- [ ] `NGPTDiffCausalAttention`: K (shape [B, nh, L, d_h]) normalized per head
- [ ] `NGPTDiffCausalAttention`: λ computation (sigmoid) unchanged
- [ ] `normalize_ngpt_weights`: uses `dim=-1` (row normalization, not column)
- [ ] `normalize_ngpt_weights`: uses seen-id set to skip tied parameters exactly once
- [ ] `ToyTransformer`: embed output L2Normed before first block when `use_ngpt`
- [ ] `ToyTransformer`: `ngpt_s_z` init = `sqrt(d)`, multiplies logits
- [ ] `TransformerBlock._forward_ngpt`: alpha stored as [d], applied with `1/sqrt(d)` scale
- [ ] `TransformerBlock._forward_ngpt`: sublayer called without norm1/norm2 pre-norm
- [ ] `ngpt_mhc_a`: alpha stored as [n_streams, d] when use_mhc + use_ngpt
- [ ] `ngpt_mhc_c`: alpha stored as [n_streams, d] when use_mhc + use_ngpt
