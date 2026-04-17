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

### DiffMLAAttention — novel composition (updated April 2026)

Updated to match the April 2026 MLA spec (see mla_attention.md): d_c=d//2, decoupled RoPE
(Eq. 14-19), RMSNorm at both latent bottlenecks.

KV and Q paths follow updated MLA spec, then Diff V2 differential:

```python
# MLA KV compression with RMSNorm (Eq. 9-11)
self.W_DKV  = nn.Linear(d, d_c, bias=False)         # d_c = d//2 = 256 at d=512
self.kv_norm = RMSNorm(d_c)
self.W_UK   = nn.Linear(d_c, nh * dh, bias=False)   # single K
self.W_UV   = nn.Linear(d_c, nh * dh, bias=False)   # standard V (V2: not double-wide)

# MLA Q: doubled for Diff V2 GQA pairing, with RMSNorm (Eq. 12-13)
self.W_DQ   = nn.Linear(d, d_c_q, bias=False)       # d_c_q = d//2 = 256 at d=512
self.q_norm  = RMSNorm(d_c_q)
self.W_UQ   = nn.Linear(d_c_q, 2 * nh * dh, bias=False)   # 2h content heads

# Decoupled RoPE (Eq. 14-15)
self.W_QR   = nn.Linear(d_c_q, 2 * nh * d_h_R, bias=False)  # 2×nh — matches doubled Q
self.W_KR   = nn.Linear(d, d_h_R, bias=False)                 # single shared positional key

# Token-specific λ (Diff V2)
self.W_lambda = nn.Linear(d, nh, bias=True)
```

Where `d_h_R = d_head // 2` (= 32 at d=512). RoPE buffers use `precompute_rope(d_h_R, max_len)`.

Forward key structure:
```python
# KV: compress → norm → expand
c_kv  = self.kv_norm(self.W_DKV(x))
k_c   = self.W_UK(c_kv).reshape(B, L, nh, dh).transpose(1, 2)         # [B, nh, L, dh]
v     = self.W_UV(c_kv).reshape(B, L, nh, dh).transpose(1, 2)

# Q: compress → norm → doubled expand
c_q   = self.q_norm(self.W_DQ(x))
q_c   = self.W_UQ(c_q).reshape(B, L, 2*nh, dh).transpose(1, 2)       # [B, 2nh, L, dh]

# Decoupled RoPE
q_rope = apply_rope(self.W_QR(c_q).reshape(B, L, 2*nh, d_h_R).transpose(1,2), ...)
k_rope = apply_rope(self.W_KR(x).reshape(B, L, 1, d_h_R).transpose(1,2), ...).expand(B,nh,L,d_h_R)

# Concatenate content + positional (Eq. 16-17)
q      = torch.cat([q_c, q_rope], dim=-1)       # [B, 2nh, L, dh + d_h_R]
k_full = torch.cat([k_c, k_rope], dim=-1)       # [B, nh,  L, dh + d_h_R]

# GQA pairing; V not rotated (V is indexed by head dim dh)
k_rep = k_full.repeat_interleave(2, dim=1)      # [B, 2nh, L, dh + d_h_R]
v_rep = v.repeat_interleave(2, dim=1)           # [B, 2nh, L, dh]

attn = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
# SDPA returns V's head dim: [B, 2nh, L, dh]
attn1 = attn[:, 0::2]    # [B, nh, L, dh]
attn2 = attn[:, 1::2]
lam   = torch.sigmoid(self.W_lambda(x)).transpose(1, 2).unsqueeze(-1)
out   = (attn1 - lam * attn2).transpose(1, 2).reshape(B, L, D)
return self.out(out)
```

**Note on W_QR:** In standalone MLA, W_QR maps `d_c_q → nh × d_h_R` (one positional vector per head).
In DiffMLAAttention, Q is doubled to 2h heads, so W_QR maps `d_c_q → 2×nh × d_h_R`. This ensures
each of the 2nh query half-heads gets its own positional component.

### Intentional deviations from V2 blog reference

| Class | Deviation | Reason |
|-------|-----------|--------|
| Both | `F.scaled_dot_product_attention` + `repeat_interleave(2)` instead of `flash_attn_func` | FlashAttention not available in CPU dev environment; SDPA is equivalent with explicit KV repetition |
| Both | λ projected from `x` (input), not from intermediate attention state | Blog shows `lam: (N, h, 1)` as input to `DiffAttnV2`; we project it from `x` at the same point in the forward pass — semantically equivalent |
| Both | `layer_idx` parameter kept in signature but unused | API compatibility; V1 used layer_idx for λ_init; V2 eliminates λ_init, so layer_idx is irrelevant in V2 |
| `DiffMLAAttention` | V is standard width d_h (not double-wide) | Correct for V2; V1 used 2·d_h |
| `DiffMLAAttention` | KV compressed via MLA latent; Q compressed via separate MLA latent | Novel composition; no published precedent. K and V still share a KV latent (MLA Eq. 9–11) |
| `DiffMLAAttention` | d_c = d//2 (not d//4) | April 2026: d//4 causes ~4.4% degradation per arXiv:2506.09342. d//2 is near-parity (+0.3%). Matches updated standalone MLA. |
| `DiffMLAAttention` | Decoupled RoPE (Eq. 14-19) included | April 2026: without decoupled RoPE, MLA degrades ~3% vs MHA. W_QR produces 2×nh positional queries to match doubled Q. |
| `DiffMLAAttention` | RMSNorm at both latents | April 2026: paper recommendation for training stability. Matches updated standalone MLA. |

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

### DiffMLAAttention (updated April 2026)

9. **KV shared latent with RMSNorm**: Confirm `self.kv_norm = RMSNorm(d_c)` applied to
   `self.W_DKV(x)` before W_UK/W_UV. `d_c = d//2` (not d//4).

10. **Q latent with RMSNorm**: Confirm `self.q_norm = RMSNorm(d_c_q)` applied to `self.W_DQ(x)`.
    `self.W_UQ` maps `d_c_q → 2 * nh * dh` (doubled Q heads).

11. **K and V standard width**: Confirm `self.W_UK = nn.Linear(d_c, nh * dh)` — V is not double-wide.

12. **Decoupled RoPE projections**: Confirm `self.W_QR = nn.Linear(d_c_q, 2 * nh * d_h_R)` (maps to
    2×nh positional vectors, one per doubled Q half-head) and `self.W_KR = nn.Linear(d, d_h_R)`.
    RoPE buffers use `precompute_rope(d_h_R, max_len)` — sized for d_h_R, not d_head.

13. **Decoupled RoPE forward**: Confirm `q = torch.cat([q_c, q_rope], dim=-1)` and
    `k_full = torch.cat([k_c, k_rope], dim=-1)` — head dim becomes dh + d_h_R for Q and K.
    V is NOT rotated. SDPA output has head dim dh (V's dim), so `reshape(B, L, D)` is valid.

14. **GQA pairing correctness**: Confirm `k_full.repeat_interleave(2, dim=1)` (not `k`) is used for
    k_rep, so the full concatenated K (content+positional) is repeated, not just content.
    Confirm `attn[:, 0::2]` and `attn[:, 1::2]` for the interleaved split.

15. **W_lambda same spec as DifferentialCausalAttention**: Confirm `nn.Linear(d, n_heads, bias=True)` —
    token-specific, head-specific, sigmoid-bounded.
