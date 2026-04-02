# Component: MLA Attention (Multi-Head Latent Attention)

## Component

**Name:** MLA — Multi-Head Latent Attention
**Class:** `MLACausalAttention` in `phase1/model.py`

**Description:** Causal self-attention with low-rank KV compression. Instead of projecting
queries, keys, and values independently at full dimension, both K and V are computed from a
shared compressed latent `c_KV = W_DKV · x ∈ ℝ^{d_c}` where d_c ≪ d. Q uses a separate
latent bottleneck. RoPE applied at full head dimension. Standard causal softmax attention.

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

At toy scale (d=512, n_heads=8, d_head=64): `d_c = d//4 = 128` vs standard `n_h·d_h = 512`.
KV parameter reduction: 2 × (d × d_c) vs 2 × (d × d) — a 4× reduction.

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

**File:** `phase1/model.py`

| Class | Lines | Role |
|-------|-------|------|
| `MLACausalAttention` | 81–148 | Full MLA layer: KV/Q compression, RoPE, SDPA, output projection |
| `MLACausalAttention.__init__` | 102–128 | Projection layers and RoPE buffers |
| `MLACausalAttention.forward` | 130–148 | Forward: compress → rope → SDPA → project |

### KV compression path (phase1/model.py:134–137)

```python
c_kv = self.W_DKV(x)                                    # Eq. 9: [B, L, d_c]
k = self.W_UK(c_kv).reshape(B, L, nh, dh).transpose(1, 2)  # Eq. 10: [B, nh, L, dh]
v = self.W_UV(c_kv).reshape(B, L, nh, dh).transpose(1, 2)  # Eq. 11: [B, nh, L, dh]
```

### Q compression path (phase1/model.py:139–141)

```python
c_q = self.W_DQ(x)                                      # Eq. 12: [B, L, d_c_q]
q = self.W_UQ(c_q).reshape(B, L, nh, dh).transpose(1, 2)   # Eq. 13: [B, nh, L, dh]
```

### RoPE and SDPA (phase1/model.py:143–148)

```python
q = apply_rope(q, self.rope_cos, self.rope_sin)
k = apply_rope(k, self.rope_cos, self.rope_sin)
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
out = out.transpose(1, 2).reshape(B, L, D)
return self.out(out)
```

### Intentional deviations from DeepSeek-V2

| Deviation | Reason |
|-----------|--------|
| No RMSNorm on c_t^{KV} or c_t^Q | Paper says "for training stability"; toy scale (d=512) does not exhibit the same bottleneck instability; adds parameters without benefit at this scale |
| No decoupled RoPE (Eq. 14–19) | Decoupled RoPE enables the KV-cache absorption trick at inference — not the objective here. Standard RoPE applied to full K and Q heads. |
| No KV-cache absorption trick | Inference optimization only; not relevant during training |
| d_c = d//4 = 128 (not 4·d_h) | Scaled to toy dimensions; DeepSeek-V2 uses d_c = 4·d_h = 512 in its much larger model |
| d_c' = d//2 = 256 (not 1536) | Same; scaled to toy dimensions |
| All projections are `nn.Linear` with `bias=False` | Matches reference code convention; no bias in attention projections |

---

## Verification checklist

1. **KV shared latent**: Confirm `self.W_DKV` is a single linear `d → d_c` at line 116, and that
   both `self.W_UK` (`d_c → n_heads*d_head`) at line 117 and `self.W_UV` at line 118 read from
   the same `c_kv = self.W_DKV(x)` output at line 135. K and V do not have independent inputs.

2. **Q separate latent**: Confirm `self.W_DQ` is independent of `W_DKV` — a separate linear at
   line 121. Q and KV latents are computed from the same input `x` but via different projections.

3. **d_c defaults**: Confirm `d_c = d // 4` at line 110 (= 128 at d=512) and `d_c_q = d // 2`
   at line 112 (= 256 at d=512).

4. **No shared parameters between K and V paths**: Confirm `W_UK` and `W_UV` are independent
   `nn.Linear` instances at lines 117–118. They read from the same `c_kv` but apply different
   weight matrices. K and V must not share weights.

5. **RoPE on both Q and K**: Confirm `apply_rope` is called on both `q` and `k` at lines 143–144.
   V is not rotated (standard attention convention).

6. **Output shape contract**: Confirm `out.transpose(1, 2).reshape(B, L, D)` at line 147 produces
   shape `[B, L, D]` before the final linear. With `D = d = n_heads * d_head`, this is correct.

7. **No RMSNorm on latents**: Confirm there is no LayerNorm or RMSNorm applied to `c_kv` or
   `c_q` in the forward pass. The reference applies norms at this bottleneck, but we intentionally
   omit them (see deviations).

8. **Parameter count reduction**: With d=512, n_heads=8, d_head=64:
   - Standard MHA: W_Q + W_K + W_V = 3 × 512² = 786,432 params
   - MLA (our): W_DKV(512×128) + W_UK(128×512) + W_UV(128×512) + W_DQ(512×256) + W_UQ(256×512)
     = 65,536 + 65,536 + 65,536 + 131,072 + 131,072 = 458,752 params  (~58% of MHA)
   Verify by counting `MLACausalAttention` parameters in a shape check run.
