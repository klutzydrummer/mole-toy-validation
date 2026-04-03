# Component: attention_rope_norms

Attention building blocks — RoPE positional encoding, RMSNorm, SwiGLU FFN, and MLA latent
attention — as used in the MoLE toy-validation transformer.

---

## Sources

### Papers
- `sources/papers/rope_2104.09864.md` — Su et al. (2021), "RoFormer: Enhanced Transformer with
  Rotary Position Embedding," arXiv:2104.09864
- `sources/papers/rmsnorm_1910.07467.md` — Zhang & Sennrich (2019), "Root Mean Square Layer
  Normalization," arXiv:1910.07467
- `sources/papers/swiglu_2002.05202.md` — Shazeer (2020), "GLU Variants Improve Transformer,"
  arXiv:2002.05202
- `sources/papers/mla_deepseek_v2_2405.04434.md` — DeepSeek-AI (2024), "DeepSeek-V2: A Strong,
  Economical, and Efficient Mixture-of-Experts Language Model," arXiv:2405.04434

### Code
- `sources/code/rope.py` — GPT-NeoX rotate_half form and LLaMA complex-number form
- `sources/code/rmsnorm.py` — original authors' version (bzhangGo/rmsnorm) and LLaMA-style rsqrt
  form
- `sources/code/swiglu.py` — LLaMA-style SwiGLUFeedForward module
- `sources/code/mla_attention.py` — DeepSeek-V3 MLA class, adapted from
  `deepseek-ai/DeepSeek-V3/inference/model.py`

---

## Authoritative equations

### 1. RoPE

Source: Su et al. (2021), arXiv:2104.09864 (`sources/papers/rope_2104.09864.md`).

**Inverse-frequency formula** (theta schedule):

```
θ_j = 10000^(-2j/d),  j = 0, 1, ..., d/2 - 1
```

Equivalently:

```
inv_freq[j] = 1 / (10000 ^ (2j / d))
```

This produces a geometric sequence of per-dimension-pair frequencies; lower-indexed dimensions
rotate faster, higher-indexed dimensions more slowly.

**Position-dependent transformation** (rotation matrix form):

```
f_q(q, m) = R_{θ,m} · q
f_k(k, n) = R_{θ,n} · k
```

The rotation matrix R_{θ,m} is block-diagonal with 2×2 rotation blocks. Applied element-wise
to consecutive pairs (x_{2j}, x_{2j+1}):

```
[x_{2j}']   = [cos(m·θ_j)  -sin(m·θ_j)] [x_{2j}  ]
[x_{2j+1}']   [sin(m·θ_j)   cos(m·θ_j)] [x_{2j+1}]
```

**rotate_half form** (equivalent, used in practice):

```
rotate_half(x):  splits x into [x1 | x2], returns [-x2 | x1]

rotated(x, m) = x * cos(m·Θ) + rotate_half(x) * sin(m·Θ)
```

where cos(m·Θ) and sin(m·Θ) are vectors of cos/sin values for each dimension pair, broadcast
over the full d-dimensional vector.

**Relative position property** (key correctness criterion):

```
⟨f_q(q, m), f_k(k, n)⟩ = g(q, k, m - n)
```

This holds because R_{θ,m}^T · R_{θ,n} = R_{θ,m-n}: the attention score depends only on the
relative offset (m - n), not on absolute positions.

---

### 2. RMSNorm

Source: Zhang & Sennrich (2019), arXiv:1910.07467 (`sources/papers/rmsnorm_1910.07467.md`).

**Core formula:**

```
RMSNorm(x) = g · x / RMS(x)

where:
  RMS(x) = sqrt( (1/d) Σ x_i² + ε )
          = sqrt( x.pow(2).mean(-1, keepdim=True) + ε )

  g = learnable scale parameter (shape: d), initialized to ones
  ε = small constant (typically 1e-6 or 1e-8)
```

Equivalently using `rsqrt`:

```
RMSNorm(x) = g · x · rsqrt( mean(x², dim=-1, keepdim=True) + ε )
```

**Key differences from LayerNorm:**
- No mean subtraction (re-centering invariance is dropped entirely).
- No learnable bias term (typically omitted).
- ε is placed *inside* the square root, not added to the RMS after the sqrt.

---

### 3. SwiGLU

Source: Shazeer (2020), arXiv:2002.05202 (`sources/papers/swiglu_2002.05202.md`).

**SwiGLU definition:**

```
SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊙ (xV + c)
```

where Swish(x) = x · σ(x) = x · sigmoid(x) (also called SiLU in PyTorch).

**Full FFN sublayer** (no bias, three weight matrices, as in LLaMA):

```
FFN_SwiGLU(x, W1, W2, W3) = (Swish(x @ W1.T) * (x @ W3.T)) @ W2.T
```

- W1: gate projection  (d_model → d_ffn) — passed through SiLU
- W3: value/up projection (d_model → d_ffn) — linear branch
- W2: output/down projection (d_ffn → d_model)

**Hidden dimension scaling:**

To keep total parameter count equal to a standard two-matrix FFN with hidden_dim = 4 × d_model,
SwiGLU (which uses three matrices) reduces the inner dimension:

```
hidden_dim = (2/3) × (4 × d_model) = 8/3 × d_model ≈ 2.667 × d_model
```

Parameter balance: 3 × d × h' = 2 × d × h  →  h' = 2h/3  →  with h = 4d: h' = 8d/3.

---

### 4. MLA (Multi-Head Latent Attention)

Source: DeepSeek-AI (2024), arXiv:2405.04434 (`sources/papers/mla_deepseek_v2_2405.04434.md`).
Equation numbers reference that paper.

**Low-rank KV joint compression** (Eq. 9–11):

```
c_t^{KV} = W^{DKV} h_t                                        (9)
k_t^C    = W^{UK}  c_t^{KV}                                   (10)
v_t^C    = W^{UV}  c_t^{KV}                                   (11)
```

Only c_t^{KV} ∈ ℝ^{d_c} is cached at inference (not the expanded K and V). This reduces KV
cache from 2·n_h·d_h elements per token to d_c elements.

**Query compression** (Eq. 12–13):

```
c_t^Q = W^{DQ} h_t                                            (12)
q_t^C = W^{UQ} c_t^Q                                          (13)
```

An RMSNorm is applied after c_t^Q and c_t^{KV} for training stability.

**Decoupled RoPE** (Eq. 14–19):

Standard RoPE cannot be applied to content keys that are reconstructed from a position-
independent latent (the W^{UK} absorption trick requires W^{UK} to be position-independent).
The solution is to maintain separate positional components:

```
q_t^R = RoPE(W^{QR} c_t^Q)                                    (14)
k_t^R = RoPE(W^{KR} h_t)                                       (15)

q_{t,i} = [q_{t,i}^C ; q_{t,i}^R]                             (16)
k_{t,i} = [k_{t,i}^C ; k_t^R]                                  (17)

o_{t,i} = Σ_j Softmax_j( q_{t,i}^T k_{j,i} / sqrt(d_h + d_h^R) ) v_{j,i}^C   (18)

u_t = W^O [o_{t,1}; ...; o_{t,n_h}]                           (19)
```

k_t^R is a single shared key vector across all n_h heads; it is position-dependent and cached
separately from c_t^{KV}. The attention scale denominator is sqrt(d_h + d_h^R).

**KV absorption trick** (inference efficiency):

```
q_nope_absorbed = q_nope @ W^{UK}.T
scores = einsum(q_nope_absorbed, c_cache) + einsum(q_pe, pe_cache)
```

W^{UK} and W^{UV} are folded into Q and output projections; only (c_t^{KV}, k_t^R) are stored.

**Cache footprint comparison** (DeepSeek-V2 numbers: n_h=128, d_h=128, d_c=512, d_h^R=64):

| Mechanism | Cache per token |
|-----------|----------------|
| MHA | 2 × n_h × d_h = 32768 elements |
| MLA (naive) | n_h × (d_h + d_h^R) = 24576 elements |
| MLA (absorption) | d_c + d_h^R = 576 elements |

From the paper: "MLA requires only a small amount of KV cache, equal to GQA with only 2.25
groups, but can achieve stronger performance than MHA."

---

## Reference implementation

### RoPE — rotate_half form (GPT-NeoX / HuggingFace style)

Source: `sources/code/rope.py`, lines 24–103. Attribution: GPT-NeoX positional_embeddings.py,
adapted in reference file from Su et al. arXiv:2104.09864.

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)           # (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset=0):
    seq_len = q.shape[-2]
    cos = cos[offset : offset + seq_len]
    sin = sin[offset : offset + seq_len]
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot
```

### RMSNorm — LLaMA rsqrt form

Source: `sources/code/rmsnorm.py`, lines 79–102. Attribution: Meta LLaMA model.py, original
paper bzhangGo/rmsnorm.

```python
class RMSNormLlama(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
```

### SwiGLU — LLaMA-style module

Source: `sources/code/swiglu.py`, lines 29–77. Attribution: Meta LLaMA model.py, Shazeer (2020)
arXiv:2002.05202.

```python
class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim=None, multiple_of=256, bias=False):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * d_model) / 3)        # = 8/3 * d_model
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=bias)   # W1
        self.up_proj   = nn.Linear(d_model, hidden_dim, bias=bias)   # W3
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=bias)   # W2

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

### MLA — DeepSeek-V3 absorption mode (verbatim source)

Source: `sources/code/mla_attention.py`, lines 186–268. Attribution: DeepSeek-AI,
`deepseek-ai/DeepSeek-V3/inference/model.py`. The verbatim block (commented out) in that file
preserves the original ColumnParallelLinear / RowParallelLinear version; the active code replaces
those with `nn.Linear` for standalone use.

Key excerpt (absorption-mode forward, from the verbatim block at lines 251–266):

```python
# else:  (absorption mode)
#     wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
#     q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
#     self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
#     self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
#     scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
#               torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
```

---

## Our implementation

### RMSNorm

`phase1/model.py:30–38`

```python
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight
```

Matches the LLaMA rsqrt form exactly. No deviations: float32 upcast, epsilon inside the sqrt
(via `add(eps).rsqrt()`), no mean subtraction, no bias. `phase2/model.py` imports this class
directly from phase1 (`from phase1.model import RMSNorm`); no separate implementation exists.

### RoPE

`phase1/model.py:41–54` — precomputation and application functions:

```python
def precompute_rope(d_head, max_len=4096, theta=10000.0):
    pos = torch.arange(max_len, dtype=torch.float32)
    freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
    angles = pos.unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    """x: [B, n_heads, L, d_head]"""
    d_half = x.shape[-1] // 2
    cos = cos[:x.shape[2], :d_half].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[2], :d_half].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., :d_half], x[..., d_half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
```

**Intentional deviation from reference rotate_half form:** The reference code uses
`rotate_half(x) = [-x2 | x1]` then computes `x * cos + rotate_half(x) * sin`, which expands
to `[x1*cos - x2*sin | x2*cos + x1*sin]`. Our `apply_rope` computes the same result directly
in a single `torch.cat` without the intermediate `rotate_half` helper. The math is identical:

```
x * cos + rotate_half(x) * sin
  = [x1 | x2] * [cos | cos] + [-x2 | x1] * [sin | sin]
  = [x1*cos - x2*sin | x2*cos + x1*sin]
```

**Deviation in table layout:** The reference builds a `(seq_len, dim)` cache via
`torch.cat([freqs, freqs], dim=-1)`. Our code stores a `(max_len, d_head/2)` table (half-width)
and slices `cos[:, :d_half]` at apply time. The values accessed are identical.

No other deviations. `CausalSelfAttention` at `phase1/model.py:57–78` applies RoPE to both q
and k before calling `F.scaled_dot_product_attention`.

### SwiGLU

`phase1/model.py:313–325`

```python
class SwiGLU(nn.Module):
    def __init__(self, d, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = int(d * 8 / 3)
            d_ff = ((d_ff + 63) // 64) * 64
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up   = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

Matches the reference exactly. The only difference from the reference `SwiGLUFeedForward` is
the alignment multiple: reference uses `multiple_of=256`; our code rounds to the nearest 64
(`((d_ff + 63) // 64) * 64`). This is a hardware-efficiency choice and has no effect on
correctness or the 8/3 parameter ratio.

### MLA

A simplified `MLACausalAttention` class exists at `phase1/model.py:81–148` — it implements the
low-rank KV/Q compression (DeepSeek-V2 Eq. 9–13) but intentionally omits decoupled RoPE (Eq. 14–19)
and the KV-cache absorption trick (inference optimization). Standard RoPE is applied to full K and Q
heads. See `references/components/mla_attention.md` for the full spec and intentional deviations.

The full MLA as specified in Eq. 14–19 (decoupled RoPE, absorption) is not implemented here — those
equations are retained for reference only.

---

## Verification checklist

### RoPE

1. **Frequency formula:** `inv_freq = 1.0 / (10000 ** (torch.arange(0, d_head, 2) / d_head))`.
   Confirm shape is `(d_head/2,)` and values decrease monotonically.

2. **Position grid:** `angles = outer(positions, inv_freq)`, shape `(seq_len, d_head/2)`.
   Confirm `angles[m, j] == m / (10000 ** (2j / d_head))`.

3. **Rotation correctness:** For `apply_rope`, confirm the output equals
   `[x1*cos - x2*sin | x2*cos + x1*sin]` — equivalent to the rotate_half form.

4. **Applied to both q and k:** Inspect `CausalSelfAttention.forward` — both `q` and `k` pass
   through `apply_rope` before attention (`phase1/model.py:74–75`).

5. **Relative position property:** `dot(rotate(q, m), rotate(k, n))` must equal
   `dot(rotate(q, 0), rotate(k, n-m))`. Use `verify_relative_position_property()` in
   `sources/code/rope.py` as a reference test.

6. **Float dtype:** `precompute_rope` uses `dtype=torch.float32` explicitly; `apply_rope`
   operates on the input dtype (no upcast). Confirm no precision issues arise in bfloat16
   training runs.

### RMSNorm

7. **No mean subtraction:** Confirm the forward pass contains no `x - x.mean(...)` term.

8. **Epsilon placement:** Confirm `eps` is added *before* the square root — specifically as
   `.add(self.eps).rsqrt()` — not after.

9. **Float32 upcast:** Confirm `x.float()` is called before the power and mean, then
   `.type_as(x)` casts back before multiplying by `self.weight`.

10. **Scale parameter:** `self.weight` shape must be `(d,)`, initialized to ones.
    Confirm no bias parameter exists in `RMSNorm.__init__`.

11. **Normalization dimension:** The `.mean(-1, keepdim=True)` operates on the feature
    dimension (last dim), not batch or sequence dimensions.

### SwiGLU

12. **Three matrices:** Confirm `gate`, `up`, and `down` projections all exist, each a
    `nn.Linear` with `bias=False`.

13. **SiLU on gate branch only:** In `forward`, `F.silu` is applied to `self.gate(x)` only.
    `self.up(x)` is linear. Confirm no activation on the up branch.

14. **Element-wise product:** The `*` operator (Hadamard product) is used to combine gate and
    up outputs, not addition or matrix multiplication.

15. **Hidden dimension ratio:** `d_ff / d` should be approximately 8/3 ≈ 2.667. For `d=256`
    (toy model default), `int(256 * 8/3) = 682`, rounded to nearest 64 gives `d_ff = 704`,
    ratio = 2.75. Confirm rounding does not yield a 4× standard FFN dimension.

16. **No bias:** All three `nn.Linear` layers use `bias=False`.

### MLA (for future implementation)

17. **KV down-projection:** `wkv_a` output must be split into `[kv_lora_rank | qk_rope_head_dim]`
    — the first part is the content latent c_t^{KV}, the second is the shared RoPE key k_t^R.

18. **RMSNorm on latents:** `kv_norm` is applied to c_t^{KV} before it is used (either passed
    to `wkv_b` in naive mode, or stored in the KV cache in absorption mode).

19. **Decoupled RoPE:** RoPE is applied to `q_pe` and `k_pe` only. Content components `q_nope`
    and `k_nope^C` must receive no positional rotation.

20. **Attention scale:** Denominator is `sqrt(qk_nope_head_dim + qk_rope_head_dim)` — the full
    concatenated head dim — not just `sqrt(qk_nope_head_dim)`.

21. **Absorption correctness:** In absorption mode, `q_nope = einsum("bshd,hdc->bshc", q_nope,
    wkv_b[:, :qk_nope_head_dim])` folds W^{UK} into the query. The KV cache stores only
    `kv_norm(kv)` (shape: kv_lora_rank) and `k_pe` (shape: qk_rope_head_dim).

22. **KV cache size:** Absorption-mode cache per token per layer must be
    `(kv_lora_rank + qk_rope_head_dim)` elements — substantially smaller than MHA's
    `2 * n_heads * head_dim`.
