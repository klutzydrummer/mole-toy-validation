# Component: Differential Attention V2

## Component

**Name:** Differential Attention V2 (and Diff+MLA composition)
**Classes:**
- `DifferentialCausalAttention` in `phase1/components/diff_attention.py` — pure Diff V2
- `DiffMLAAttention` in `phase1/components/diff_attention.py` — Diff V2 + MLA KV compression (novel composition)

**Description:** Differential attention cancels noise by subtracting two softmax attention
outputs scaled by a learned per-token, per-head coefficient λ. V2 (January 2026) doubles the
query heads while keeping standard KV, enabling standard FlashAttention/SDPA and removing V1's
per-head RMSNorm and static λ initialization. `DiffMLAAttention` applies the same differential
mechanism to an MLA-compressed KV path.

---

## Sources

**Papers (V1):**
- `sources/papers/diff_attn_v1_2410.05258.md`
  — Differential Transformer (arXiv:2410.05258, ICLR 2025)
  — V1 design; documents the baseline against which V2 changes are measured.

**Blog post (V2):**
- `sources/papers/diff_attn_v2_2026_01.md`
  — Differential Transformer V2 (Microsoft blog, January 20, 2026)
  — Authoritative V2 equations, pseudocode, and ablations. No arXiv preprint at time of writing.

**MLA (for DiffMLAAttention):**
- `sources/papers/mla_deepseek_v2_2405.04434.md`
  — DeepSeek-V2 (arXiv:2405.04434)
  — Equations 9–13 for the KV and Q compression paths used in DiffMLAAttention.
- `sources/code/mla_attention.py`
  — DeepSeek-V3 reference implementation for KV/Q compression naive forward.

---

## Authoritative equations

### V2 head layout

From `sources/papers/diff_attn_v2_2026_01.md`:

```
q ∈ ℝ^{N × 2h × d}      — doubled Q heads
k ∈ ℝ^{N × h × d}       — standard KV heads (unchanged from baseline)
v ∈ ℝ^{N × h × d}       — standard width (not double-wide as in V1)
lam ∈ ℝ^{N × h × 1}     — token-specific, head-specific λ via sigmoid projection
```

### V2 differential operation (verbatim from blog pseudocode)

```python
def DiffAttnV2(q, k, v, lam):
    """
    q:   (N, 2h, d)
    k:   (N, h_kv, d)
    v:   (N, h_kv, d)
    lam: (N, h, 1)
    """
    attn = flash_attn_func(q, k, v)
    attn1, attn2 = (attn[:, 0::2],    # even heads (q1 group)
                    attn[:, 1::2])     # odd heads  (q2 group)

    lam_val = sigmoid(lam)
    attn = attn1 - lam_val * attn2
    return attn
```

**Critical note (verbatim from blog):**
> "DIFF V2 subtracts two heads that are in the same GQA group (share the same key and value).
> This is crucial to performance."

Interleaved split `[0::2]` / `[1::2]` is the correct GQA pairing.
Split `[:nh//2]` / `[nh//2:]` is **wrong** and causes training instability (blog ablation 1).

### V2 λ parameterization

λ is projected from the input per-token, per-head:

```
lam_i = sigmoid(W_λ · x_i)    W_λ ∈ ℝ^{n_heads × d}
```

- `bias=True` so `sigmoid(0) = 0.5` at initialization — a reasonable starting λ.
- Token- and head-specific; no shared λ_init constant.
- Context RMS of output: `(0, √2)` — lower bound 0 enables elimination of attention sinks.

### V2 changes from V1 (summary)

| Aspect | V1 | V2 |
|--------|----|----|
| Q heads | h/2 pairs (separate q1, q2) | 2h heads (contiguous pairs) |
| K/V width | h/2 heads, V double-wide (2·d_h) | h heads, V standard (d_h) |
| λ | static exp reparameterization + λ_init | token-specific sigmoid projection |
| Per-head RMSNorm | yes | **removed** |
| FlashAttention | requires custom kernel | standard (no modification needed) |
| Context RMS range | [1/√n, 1) after RMSNorm | **(0, √2)** |

---

## Reference implementation

**Source:** `sources/papers/diff_attn_v2_2026_01.md` (blog pseudocode)
**Note:** No separate code file exists for V2; the authoritative pseudocode is in the blog.

### DiffAttnV2 — complete reference

```python
def DiffAttnV2(q, k, v, lam):
    """
    q:   (N, 2h, d)
    k:   (N, h_kv, d)
    v:   (N, h_kv, d)
    lam: (N, h, 1)
    """
    attn = flash_attn_func(q, k, v)
    attn1, attn2 = (attn[:, 0::2],
                    attn[:, 1::2])
    lam_val = sigmoid(lam)
    attn = attn1 - lam_val * attn2
    return attn
```

**Our adaptation:** `flash_attn_func` → `F.scaled_dot_product_attention(..., is_causal=True)`;
K and V repeated via `repeat_interleave(2, dim=1)` to match 2h query heads for SDPA.

---

## Our implementation

**File:** `phase1/components/diff_attention.py`

| Class | Lines | Role |
|-------|-------|------|
| `DifferentialCausalAttention` | 18–92 | Pure Diff V2: doubled Q, standard KV, sigmoid λ |
| `DifferentialCausalAttention.__init__` | 43–62 | Projections, W_λ, RoPE buffers |
| `DifferentialCausalAttention.forward` | 64–92 | Forward: project → rope → repeat_kv → SDPA → diff → out |
| `DiffMLAAttention` | 95–177 | Diff V2 + MLA KV compression (novel composition) |
| `DiffMLAAttention.__init__` | 116–147 | MLA KV/Q projections, W_λ, RoPE buffers |
| `DiffMLAAttention.forward` | 149–177 | Forward: MLA compress → rope → repeat_kv → SDPA → diff → out |

### DifferentialCausalAttention — projections (phase1/components/diff_attention.py:50–58)

```python
self.W_Q = nn.Linear(d, 2 * d, bias=False)   # d → 2·n_heads·d_head (doubled Q)
self.W_K = nn.Linear(d, d, bias=False)        # d → n_heads·d_head
self.W_V = nn.Linear(d, d, bias=False)        # d → n_heads·d_head
self.W_lambda = nn.Linear(d, n_heads, bias=True)  # token-specific λ per head
self.out = nn.Linear(d, d, bias=False)
```

### DifferentialCausalAttention — forward (phase1/components/diff_attention.py:64–92)

```python
q = self.W_Q(x).reshape(B, L, 2 * nh, dh).transpose(1, 2)  # [B, 2h, L, dh]
k = self.W_K(x).reshape(B, L, nh, dh).transpose(1, 2)       # [B, h, L, dh]
v = self.W_V(x).reshape(B, L, nh, dh).transpose(1, 2)

q = apply_rope(q, self.rope_cos, self.rope_sin)
k = apply_rope(k, self.rope_cos, self.rope_sin)

k_rep = k.repeat_interleave(2, dim=1)   # [B, 2h, L, dh] — GQA pairing
v_rep = v.repeat_interleave(2, dim=1)

attn = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)  # [B, 2h, L, dh]

attn1 = attn[:, 0::2]   # [B, h, L, dh]
attn2 = attn[:, 1::2]

lam = torch.sigmoid(self.W_lambda(x)).transpose(1, 2).unsqueeze(-1)  # [B, h, L, 1]

out = (attn1 - lam * attn2).transpose(1, 2).reshape(B, L, D)
return self.out(out)
```

### DiffMLAAttention — novel composition (phase1/components/diff_attention.py:116–147)

KV and Q paths follow MLA (Eq. 9–13 from arXiv:2405.04434), then Diff V2 differential:

```python
# MLA KV compression
self.W_DKV = nn.Linear(d, d_c, bias=False)        # d_c = d//4 = 128 at d=512
self.W_UK  = nn.Linear(d_c, nh * dh, bias=False)  # single K (not doubled)
self.W_UV  = nn.Linear(d_c, nh * dh, bias=False)  # standard V (V2: not double-wide)

# MLA Q: doubled for Diff V2 GQA pairing
self.W_DQ  = nn.Linear(d, d_c_q, bias=False)      # d_c_q = d//2 = 256 at d=512
self.W_UQ  = nn.Linear(d_c_q, 2 * nh * dh, bias=False)  # 2h heads

# Token-specific λ (Diff V2)
self.W_lambda = nn.Linear(d, nh, bias=True)
```

Forward (phase1/components/diff_attention.py:149–177):
```python
c_kv = self.W_DKV(x)
k = self.W_UK(c_kv).reshape(B, L, nh, dh).transpose(1, 2)
v = self.W_UV(c_kv).reshape(B, L, nh, dh).transpose(1, 2)

c_q = self.W_DQ(x)
q   = self.W_UQ(c_q).reshape(B, L, 2 * nh, dh).transpose(1, 2)

q = apply_rope(q, self.rope_cos, self.rope_sin)
k = apply_rope(k, self.rope_cos, self.rope_sin)

k_rep = k.repeat_interleave(2, dim=1)
v_rep = v.repeat_interleave(2, dim=1)

attn = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
attn1 = attn[:, 0::2]
attn2 = attn[:, 1::2]

lam = torch.sigmoid(self.W_lambda(x)).transpose(1, 2).unsqueeze(-1)
out = (attn1 - lam * attn2).transpose(1, 2).reshape(B, L, D)
return self.out(out)
```

### Intentional deviations from V2 blog reference

| Class | Deviation | Reason |
|-------|-----------|--------|
| Both | `F.scaled_dot_product_attention` + `repeat_interleave(2)` instead of `flash_attn_func` | FlashAttention not available in CPU dev environment; SDPA is equivalent with explicit KV repetition |
| Both | λ projected from `x` (input), not from intermediate attention state | Blog shows `lam: (N, h, 1)` as input to `DiffAttnV2`; we project it from `x` at the same point in the forward pass — semantically equivalent |
| Both | `layer_idx` parameter kept in signature but unused | API compatibility; V1 used layer_idx for λ_init; V2 eliminates λ_init, so layer_idx is irrelevant in V2 |
| `DiffMLAAttention` | V is standard width d_h (not double-wide) | Correct for V2; V1 used 2·d_h |
| `DiffMLAAttention` | KV compressed via MLA latent; Q compressed via separate MLA latent | Novel composition; no published precedent. K and V still share a KV latent (MLA Eq. 9–11) |
| `DiffMLAAttention` | No RMSNorm on latents | Same rationale as `MLACausalAttention` (see mla_attention.md) |

---

## Verification checklist

### DifferentialCausalAttention

1. **Doubled Q, standard KV**: Confirm `self.W_Q = nn.Linear(d, 2 * d, bias=False)` at line 183
   and `self.W_K = nn.Linear(d, d, bias=False)` at line 184. Q output is `2 * n_heads * d_head`,
   not `n_heads * d_head`.

2. **GQA pairing via repeat_interleave**: Confirm `k.repeat_interleave(2, dim=1)` at line 211 and
   `v.repeat_interleave(2, dim=1)` at line 212. This produces `[h0,h0,h1,h1,...]` (interleaved),
   matching `Q[0::2]` to K[0] and `Q[1::2]` to K[0] — same GQA group shares K,V.

3. **Correct head split (interleaved, not blocked)**: Confirm `attn[:, 0::2]` at line 218 and
   `attn[:, 1::2]` at line 219. NOT `attn[:, :nh//2]` / `attn[:, nh//2:]` (the wrong ablation).

4. **Sigmoid λ, not exp reparameterization**: Confirm `torch.sigmoid(self.W_lambda(x))` at line 222.
   No `torch.exp()` involved in λ computation.

5. **W_lambda bias=True**: Confirm `nn.Linear(d, n_heads, bias=True)` at line 189. `bias=True`
   ensures sigmoid(0) = 0.5 at initialization instead of a potentially degenerate starting point.

6. **No per-head RMSNorm**: Confirm there is no RMSNorm, LayerNorm, or norm applied to `attn`,
   `attn1`, or `attn2` before or after the differential operation. V1's RMSNorm is not present.

7. **λ shape**: Confirm `self.W_lambda(x)` produces `[B, L, n_heads]`, then `.transpose(1, 2)`
   → `[B, n_heads, L]`, then `.unsqueeze(-1)` → `[B, n_heads, L, 1]` at line 222. This broadcasts
   correctly against `attn1` of shape `[B, n_heads, L, d_head]`.

8. **Output shape**: Confirm `(attn1 - lam * attn2)` at line 224 has shape `[B, n_heads, L, d_head]`,
   and `.transpose(1, 2).reshape(B, L, D)` collapses to `[B, L, D]` where `D = n_heads * d_head`.

### DiffMLAAttention

9. **KV shared latent**: Confirm `self.W_DKV` at line 265, `self.W_UK` at line 266, and
   `self.W_UV` at line 267 exist; and that `k = self.W_UK(c_kv)` and `v = self.W_UV(c_kv)` at
   lines 288–289 both operate on the same `c_kv = self.W_DKV(x)` at line 287.

10. **Doubled Q via MLA**: Confirm `self.W_UQ = nn.Linear(d_c_q, 2 * nh * dh, bias=False)` at
    line 271 — the up-projection produces 2h heads, not h heads.

11. **K and V standard width in DiffMLAAttention**: Confirm `self.W_UK = nn.Linear(d_c, nh * dh)` at
    line 266 (output dim = n_heads * d_head, not 2 * n_heads * d_head). V is not double-wide.

12. **Same GQA pairing correctness in DiffMLAAttention**: Confirm `k.repeat_interleave(2, dim=1)`
    at line 299 and `attn[:, 0::2]` at line 304 — same correct interleaved split as
    `DifferentialCausalAttention`.

13. **W_lambda same spec in both classes**: Confirm `nn.Linear(d, n_heads, bias=True)` at line 274.
    Same specification as `DifferentialCausalAttention` line 189 — token-specific, head-specific,
    sigmoid-bounded.
