# Component: MLA Attention (Multi-Head Latent Attention)

## Component

**Name:** MLA — Multi-Head Latent Attention
**Class:** `MLACausalAttention` in `phase1/components/mla_attention.py`

**Description:** Causal self-attention with low-rank KV compression, decoupled RoPE, and
RMSNorm at both latent bottlenecks. K and V are computed from a shared compressed latent
`c_KV = RMSNorm(W_DKV · x) ∈ ℝ^{d_c}`. Q uses a separate compressed latent with its own
RMSNorm. Decoupled RoPE (Eq. 14–19): content path (no RoPE) and positional path (RoPE on
`d_h_R`-dim projections) are concatenated per head. Updated April 2026 from the original
no-RMSNorm, no-decoupled-RoPE version.

---

## Sources

**Papers:**
- `sources/papers/mla_deepseek_v2_2405.04434.md`
  — DeepSeek-V2 (arXiv:2405.04434, DeepSeek-AI, May 2024)
  — Equations 9–13 define KV and Q compression used verbatim.

**Code:**
- `sources/code/mla_attention.py`
  — Extracted from DeepSeek-V3 inference/model.py; tensor-parallel boilerplate removed.
  — `MLA.forward` in naive mode (lines 44–183) is the direct reference for the training-time
    forward pass used here.

---

## Authoritative equations

All equations verbatim from: **DeepSeek-V2** (arXiv:2405.04434), Section 2.1.

### Equations 9–11 — Low-rank KV joint compression

```
c_t^{KV} = W^{DKV} h_t                                        (9)
k_t^C    = W^{UK}  c_t^{KV}                                   (10)
v_t^C    = W^{UV}  c_t^{KV}                                   (11)
```

Variables:
- `h_t` — input hidden state at position t, shape ∈ ℝ^d
- `c_t^{KV}` — KV compression latent, shape ∈ ℝ^{d_c}  (only this is cached at inference)
- `W^{DKV}` ∈ ℝ^{d_c × d}          — down-projection: input → KV latent
- `W^{UK}`  ∈ ℝ^{n_h·d_h × d_c}    — up-projection: KV latent → keys
- `W^{UV}`  ∈ ℝ^{n_h·d_h × d_c}    — up-projection: KV latent → values
- `d_c` = KV compression dimension (≪ n_h·d_h)

At toy scale (d=512, n_heads=8, d_head=64): `d_c = d//2 = 256` vs standard `n_h·d_h = 512`.
KV parameter reduction: 2 × (d × d_c) vs 2 × (d × d) — a 2× reduction (changed from 4×
in April 2026; d//4 caused degradation even with decoupled RoPE per arXiv:2506.09342).

### Equations 12–13 — Query compression

```
c_t^Q = W^{DQ} h_t                                            (12)
q_t^C = W^{UQ} c_t^Q                                          (13)
```

Variables:
- `W^{DQ}` ∈ ℝ^{d_c' × d}          — down-projection: input → Q latent
- `W^{UQ}` ∈ ℝ^{n_h·d_h × d_c'}    — up-projection: Q latent → queries
- `d_c'` = Q compression dimension (larger than d_c; reduces activation memory not KV cache)

At toy scale: `d_c' = d//2 = 256`.

Paper quote (verbatim):
> "An RMSNorm is applied after c_t^Q (and c_t^{KV}) for training stability."

**Our implementation now includes these norms** (updated April 2026, see deviations).

### Equations 14–19 — Decoupled RoPE

```
q_t^R = RoPE(W^{QR} c_t^Q)    per-head positional query from Q latent     (14)
k_t^R = RoPE(W^{KR} h_t)      single shared positional key from input      (15)

q_{t,i} = [q_{t,i}^C ; q_{t,i}^R]   content + positional concatenated      (16)
k_{t,i} = [k_{t,i}^C ; k_t^R]       content + shared positional key         (17)

o_{t,i} = Σ_j Softmax_j( q_{t,i}^T k_{j,i} / sqrt(d_h + d_h^R) ) v_{j,i}  (18)
u_t = W^O [o_{t,1}; ...; o_{t,n_h}]                                         (19)
```

Where `d_h^R = d_h / 2` (= 32 at d=512). `W^{QR}` projects the Q latent to per-head
RoPE vectors; `W^{KR}` projects input directly to a **single shared** positional key
(broadcast to all heads at attention time). **Our implementation now includes decoupled
RoPE** (updated April 2026, see deviations).

---

## Reference implementation

**Source:** `sources/code/mla_attention.py`, lines 44–183
**Attribution:** DeepSeek-AI, extracted from DeepSeek-V3 inference/model.py
**Mode:** Naive (training-compatible; no absorption trick)

### Naive forward — KV compression and Q compression

Verbatim from `sources/code/mla_attention.py`, lines 127–146 (naive mode, simplified):

```python
# KV compression (Eq. 9–11) — sources/code/mla_attention.py:144
kv = self.wkv_b(self.kv_norm(kv))              # down-proj → norm → up-proj
kv = kv.view(bsz, seqlen, self.n_local_heads,
             self.qk_nope_head_dim + self.v_head_dim)
k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

# Q compression (Eq. 12–13) — sources/code/mla_attention.py:127
q = self.wq_b(self.q_norm(self.wq_a(x)))       # down-proj → norm → up-proj
```

Key structural note: in the reference, the down-proj and norm form a single combined path:
`W_DKV (wkv_a) + RMSNorm (kv_norm)` before `W_UK/W_UV (wkv_b)`. The norm is applied at the
bottleneck. Field names in the source: `wkv_a` = W^{DKV}, `wkv_b` = joint W^{UK}/W^{UV},
`wq_a` = W^{DQ}, `wq_b` = W^{UQ}.

---

## Our implementation

**File:** `phase1/components/mla_attention.py`

| Class | Lines | Role |
|-------|-------|------|
| `MLACausalAttention` | 17–120 | Full MLA layer: KV/Q compression, decoupled RoPE, SDPA, output projection |
| `MLACausalAttention.__init__` | 48–84 | Projection layers, norms, and RoPE buffers |
| `MLACausalAttention.forward` | 86–120 | Forward: compress → norm → rope → cat → SDPA → project |

### KV compression path (phase1/components/mla_attention.py:91–93)

```python
c_kv = self.kv_norm(self.W_DKV(x))                             # Eq. 9 + RMSNorm: [B, L, d_c]
k_c  = self.W_UK(c_kv).reshape(B, L, nh, dh).transpose(1, 2)  # Eq. 10: content K [B, nh, L, dh]
v    = self.W_UV(c_kv).reshape(B, L, nh, dh).transpose(1, 2)  # Eq. 11: [B, nh, L, dh]
```

### Q compression path (phase1/components/mla_attention.py:96–97)

```python
c_q  = self.q_norm(self.W_DQ(x))                               # Eq. 12 + RMSNorm: [B, L, d_c_q]
q_c  = self.W_UQ(c_q).reshape(B, L, nh, dh).transpose(1, 2)   # Eq. 13: content Q [B, nh, L, dh]
```

### Decoupled RoPE (phase1/components/mla_attention.py:101–115)

```python
q_rope = apply_rope(self.W_QR(c_q).reshape(B,L,nh,d_h_R).transpose(1,2), ...)  # Eq. 14
k_rope = apply_rope(self.W_KR(x).reshape(B,L,1,d_h_R).transpose(1,2), ...).expand(B,nh,L,d_h_R)  # Eq. 15
q = torch.cat([q_c, q_rope], dim=-1)   # Eq. 16: [B, nh, L, dh + d_h_R]
k = torch.cat([k_c, k_rope], dim=-1)   # Eq. 17: [B, nh, L, dh + d_h_R]
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # Eq. 18
```

### Intentional deviations from DeepSeek-V2

| Deviation | Reason |
|-----------|--------|
| RMSNorm **included** on both c_t^{KV} and c_t^Q | Added April 2026. Empirical evidence (arXiv:2506.09342) showed MLA underperforms MHA without this. Now matches paper recommendation. |
| Decoupled RoPE (Eq. 14–19) **included** | Added April 2026. Without decoupled RoPE, MLA degrades ~3% vs MHA even at zero compression (arXiv:2506.09342). k_rope is a single shared positional key broadcast to all heads. |
| d_c = d//2 = 256 (not d//4) | Changed April 2026 from d//4=128. At d//4 with decoupled RoPE, still +4.4% worse than MHA (arXiv:2506.09342 Table 2); d//2 is near-parity (+0.3%). 2× compression vs paper's ~90× at DeepSeek-V2 scale. |
| d_c' = d//2 = 256 (not 1536) | Scaled to toy dimensions |
| d_h_R = d_h//2 = 32 (not 64) | Scaled to toy dimensions; DeepSeek-V2 uses d_h^R = 64 with d_h = 128 |
| Separate `W_KR` instead of joint `wkv_a` output | Reference code uses a single `wkv_a: d → d_c + d_h_R` that outputs the KV latent and the RoPE key together, split afterward. Our implementation uses separate `W_DKV: d → d_c` and `W_KR: d → d_h_R`. Functionally equivalent; separate projections are clearer and avoid a non-standard split. |
| No KV-cache absorption trick | Inference optimization only; not relevant during training |
| All projections are `nn.Linear` with `bias=False` | Matches reference code convention; no bias in attention projections |

---

## Verification checklist

1. **KV shared latent with RMSNorm**: Confirm `self.kv_norm` is an `RMSNorm(d_c)` applied to
   `self.W_DKV(x)` before W_UK/W_UV. Both K_C and V read from the same normed latent.

2. **Q separate latent with RMSNorm**: Confirm `self.q_norm` is an `RMSNorm(d_c_q)` applied to
   `self.W_DQ(x)`. Q latent is independent of KV latent.

3. **d_c defaults**: Confirm `d_c = d // 2` (= 256 at d=512) and `d_c_q = d // 2` (= 256 at d=512).

4. **Decoupled RoPE projections**: Confirm `self.W_QR` maps `d_c_q → n_heads * d_h_R` and
   `self.W_KR` maps `d → d_h_R` where `d_h_R = d_head // 2` (= 32 at d=512).

5. **Shared positional key broadcast**: Confirm `k_rope` is computed from `self.W_KR(x)` with
   shape `[B, 1, L, d_h_R]` before `.expand(B, nh, L, d_h_R)` — a single shared key for all heads.

6. **Content/positional concatenation**: Confirm `q = torch.cat([q_c, q_rope], dim=-1)` and
   `k = torch.cat([k_c, k_rope], dim=-1)` produce head dim `dh + d_h_R = 96` at d=512.

7. **V not rotated**: Confirm `apply_rope` is NOT called on `v` — only on `q_rope` and `k_rope`.
   Content keys K_C also have no RoPE.

8. **Output shape contract**: SDPA outputs `[B, nh, L, dh]` (V's head dim), so
   `out.transpose(1, 2).reshape(B, L, D)` produces `[B, L, D]` correctly since `D = nh * dh`.

9. **RoPE buffer size**: Confirm `precompute_rope(self.d_h_R, max_len)` — buffers are sized for
   `d_h_R` (32), not `d_head` (64). `apply_rope` splits by `d_half = shape[-1] // 2` internally.

10. **No shared parameters between K and V paths**: Confirm `W_UK` and `W_UV` are independent
    `nn.Linear` instances reading from the same `c_kv`.

11. **Parameter count (April 2026)**: With d=512, n_heads=8, d_head=64, d_c=256, d_c_q=256, d_h_R=32:
    - W_DKV(512×256) + kv_norm(256) + W_UK(256×512) + W_UV(256×512) = 131,072 + 256 + 131,072 + 131,072
    - W_DQ(512×256) + q_norm(256) + W_UQ(256×512) = 131,072 + 256 + 131,072
    - W_QR(256×256) + W_KR(512×32) = 65,536 + 16,384
    - out(512×512) = 262,144
    - Total attention params ≈ 1,000,192 (vs standard MHA 3×512²+512² = 1,048,576 — ~95% of MHA)
    Verify by counting `MLACausalAttention` parameters in a shape check run.
