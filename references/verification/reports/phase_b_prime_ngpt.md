# Phase B' Verification Report: nGPT

**Component:** `ngpt`
**Implementation files:** `phase1/components/ngpt.py`, `phase1/components/_shared.py`
**Spec:** `references/components/ngpt.md`
**Primary source:** `references/sources/papers/ngpt_2410.01131.md` (arXiv:2410.01131)
**Reference code:** `references/sources/code/ngpt_reference.py` (NVIDIA/ngpt, MIT)
**Verified:** 2026-04-15
**Verdict:** PASS with documented deviations

---

## Source cross-check

### Spherical update rule (Eq. 10–11, arXiv:2410.01131)

**Paper:** `h = L2Norm(h + α ⊙ (L2Norm(F(h)) − h))`

**NVIDIA ref (ngpt_reference.py):**
```python
A_norm = self.justnorm(h)
B_norm = self.justnorm(h_att)
res = A_norm + lr * (B_norm - A_norm)
h = self.justnorm(res)
```

**Our impl (`transformer_block.py`, `_forward_ngpt`):**
```python
h_a = l2_norm(self.attn(x))
x   = l2_norm(x + alpha_a * (h_a - x))
```

**Status: VERIFIED.** Functionally identical. NVIDIA normalizes `h` before the LERP step
(`A_norm = justnorm(h)`); our impl maintains `x` on the sphere across blocks so `l2_norm(x) = x`.
The `l2_norm(sublayer(x))` matches `B_norm = justnorm(h_att)`.

---

### l2_norm (ngpt.py:31–33)

**Paper:** L2 normalization along the embedding dimension.

**Our impl:**
```python
def l2_norm(x): return F.normalize(x, dim=-1)
```

**NVIDIA ref:**
```python
def justnorm(self, x): return x / x.norm(p=2, dim=-1, keepdim=True)
```

**Status: VERIFIED.** Equivalent. `F.normalize` handles zero-norm gracefully.

---

### Q/K normalization and s_qk (Section 2.2)

**Paper:** Q and K normalized to unit norm per head; attention scale inverted to s_qk (init √d_h).

**NGPTCausalAttention (ngpt.py:75–79):**
```python
q = F.normalize(q, dim=-1)
k = F.normalize(k, dim=-1)
out = F.scaled_dot_product_attention(q * self.s_qk, k, v, is_causal=True, scale=1.0)
```
`self.s_qk = nn.Parameter(torch.tensor(math.sqrt(d // n_heads)))` ✓

**Status: VERIFIED** with documented deviation from NVIDIA ref.

**Deviation (documented in ngpt_reference.py):** NVIDIA uses per-dim per-head sqk `[n_head, d_head]`
applied to both Q and K; our impl uses a scalar applied to Q only with `scale=1.0`. Both achieve
inverted-scale behavior. Our scalar matches the paper's Section 2.2 description more closely.

---

### NGPTMLACausalAttention: no kv_norm/q_norm (ngpt.py:114, 121)

**Claim:** kv_norm and q_norm removed for nGPT path — input already unit-norm.

**Our impl:** `W_DKV`, `W_UK`, `W_UV`, `W_DQ`, `W_UQ` used directly with no RMSNorm applied
to latents. `MLACausalAttention` has `self.kv_norm = RMSNorm(d_c)` and `self.q_norm = RMSNorm(d_c_q)`;
`NGPTMLACausalAttention` omits both.

**Status: VERIFIED.** Intentional deviation from `MLACausalAttention` (no paper precedent —
novel nGPT+MLA composition).

---

### NGPTMLACausalAttention: full head normalization (ngpt.py:165–166)

**Claim (spec):** Full `[q_c; q_rope]` and `[k_c; k_rope]` concatenated and then normalized
per head — not parts separately.

**Our impl:**
```python
q = torch.cat([q_c, q_rope], dim=-1)   # [B, nh, L, d_h + d_h_R]
k = torch.cat([k_c, k_rope], dim=-1)
q = F.normalize(q, dim=-1)
k = F.normalize(k, dim=-1)
```

**Status: VERIFIED.** Content and positional components concatenated first, then single
F.normalize call normalizes the full head vector.

---

### NGPTMLACausalAttention: s_qk init (ngpt.py:131)

**Claim:** `s_qk = sqrt(d_head + d_h_R)` — scaled for full concatenated head dim.

**Our impl:** `self.s_qk = nn.Parameter(torch.tensor(math.sqrt(full_head_dim)))` where
`full_head_dim = self.d_head + self.d_h_R`. At d=512, n_heads=8: d_head=64, d_h_R=32 → full=96,
s_qk init ≈ 9.8.

**Status: VERIFIED.**

---

### NGPTDiffCausalAttention: per-half-head normalization (ngpt.py:222–223)

**Claim:** Each of 2nh Q half-heads normalized independently.

**Our impl:**
```python
q = F.normalize(q, dim=-1)   # [B, 2nh, L, d_h] — F.normalize on last dim normalizes each [d_h] vector
k = F.normalize(k, dim=-1)   # [B, nh,  L, d_h]
```

**Status: VERIFIED.** `F.normalize(x, dim=-1)` on shape `[B, 2nh, L, d_h]` normalizes each
`d_h`-vector independently, i.e., each of the 2nh half-heads is individually unit-normed.

---

### NGPTDiffCausalAttention: λ unchanged (ngpt.py:237)

**Claim:** Lambda computation (sigmoid) unchanged — it is a scalar blend, not a sphere projection.

**Our impl:**
```python
lam = torch.sigmoid(self.W_lambda(x)).transpose(1, 2).unsqueeze(-1)
```

**Status: VERIFIED.** Identical to `DifferentialCausalAttention`.

---

### normalize_ngpt_weights (ngpt.py:243–263)

**Claim:** Row normalization (`dim=-1`), covers all `nn.Linear | nn.Embedding`, uses seen-id set.

**Paper (Section 2.3):** "normalize each weight matrix W such that each row vector has unit L2 norm."

**NVIDIA ref:** Uses `dim=0` (column normalization). CONFLICT WITH PAPER.

**Our impl:**
```python
w.data.copy_(F.normalize(w.data, dim=-1))   # row normalization
```

**Status: VERIFIED** (follows paper, not NVIDIA ref). Deviation from NVIDIA ref documented
in `ngpt_reference.py`.

**Seen-id set:** Present at ngpt.py:257 (`seen = set()`, checked at line 261). Correctly
handles tied embed/lm_head weights.

---

### ToyTransformer: embed L2Norm (model.py)

**Claim:** Embed output L2Normed before first block when `use_ngpt`.

**Our impl (model.py `forward`):**
```python
h = self.embed(x)
if self.use_ngpt:
    h = l2_norm(h)
```

**Status: VERIFIED.**

---

### ToyTransformer: ngpt_s_z init (model.py)

**Claim:** `ngpt_s_z` init = `sqrt(d)`.

**Our impl:** `self.ngpt_s_z = nn.Parameter(torch.tensor(math.sqrt(d)))`

**Status: VERIFIED.**

**Deviation from NVIDIA ref documented:** NVIDIA uses per-vocab vector `sz` init=1.0; our
scalar init=sqrt(d) follows the paper's Section 2.5 more closely.

---

### TransformerBlock._forward_ngpt: no pre-norm

**Claim:** sublayer called without norm1/norm2 when `use_ngpt` (single-stream).

**Our impl (`_forward_ngpt`):**
```python
h_a = l2_norm(self.attn(x))      # no self.norm1 call
x   = l2_norm(x + alpha_a * (h_a - x))
h_m = l2_norm(self.ffn(x))       # no self.norm2 call
```

**Status: VERIFIED.**

---

### ngpt_mhc_a and ngpt_mhc_c: per-stream alpha

**Claim:** alpha shape `[n_streams, d]` when `use_ngpt and use_mhc`.

**Our impl (`transformer_block.py`, `__init__`):**
```python
if use_ngpt:
    if use_mhc:
        self.alpha_a = nn.Parameter(torch.full((n_streams, d), 0.05))
        self.alpha_m = nn.Parameter(torch.full((n_streams, d), 0.05))
    else:
        self.alpha_a = nn.Parameter(torch.full((d,), 0.05))
        self.alpha_m = nn.Parameter(torch.full((d,), 0.05))
```

**Status: VERIFIED.** Novel compositions — no published precedent. Per-stream alpha
`[n_streams, d]` is correct for broadcasting against `[B, L, n_streams, d]`.

---

### ngpt_mhc_a: norm1/norm2 kept in branch

**Claim:** norm1/norm2 applied inside branch_fn for mHC variants (H_pre·x is not on sphere).

**Our impl (`_ngpt_mhc_a_step`):**
```python
branch_in = einsum("n,blnd->bld", pre_w, x)   # weighted sum of unit vectors — NOT on sphere
h = l2_norm(branch_fn(branch_in))             # branch_fn applies norm1/norm2 inside
```
`branch_fn = lambda inp: self.attn(self.norm1(inp))` — norm1 IS called.

**Status: VERIFIED.**

---

## Summary of intentional deviations

| Item | Paper | Our impl | NVIDIA ref | Notes |
|------|-------|----------|------------|-------|
| s_qk | scalar per layer | scalar `sqrt(d_h)` | per-dim per-head `[n_head, d_head]` | Follow paper |
| sz (logit scale) | scalar | scalar `sqrt(d)` | per-vocab `[vocab_size]` | Follow paper |
| Weight norm dim | rows (dim=-1) | dim=-1 | dim=0 (columns) | Follow paper — NVIDIA ref contradicts paper |
| norm1/norm2 | removed | kept but unused in standard nGPT | removed | Avoided conditional __init__ |
| Warmup | removed | 1k-step warmup kept | n/a | Controlled comparison |
| NGPTMLACausalAttention kv_norm/q_norm | n/a (novel) | removed | n/a | Input already unit-norm |
| ngpt_mhc_a/c | no precedent | implemented | n/a | Novel compositions |

**Overall verdict: PASS.** All deviations are documented, intentional, and either follow
the paper over the NVIDIA ref or are necessary for novel compositions.
