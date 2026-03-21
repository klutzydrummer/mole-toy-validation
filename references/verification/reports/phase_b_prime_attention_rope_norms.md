# Phase B' Validation Report: attention_rope_norms

**Component file:** `references/components/attention_rope_norms.md`
**Validator:** Phase B' validation agent
**Date:** 2026-03-21 (re-verified; original 2026-03-17)
**Overall verdict: PASS**

**Re-verification note (2026-03-21):** Full re-check performed after `phase2/model.py` log_a_init fix. Attention, RoPE, RMSNorm, SwiGLU are unaffected. All previously verified claims remain PASS. Implementation cross-checks now confirmed against actual line numbers: RMSNorm at `phase1/model.py:30-38`, `precompute_rope` at 41-54 (float32 dtype, half-width cache), `apply_rope` at line 74-75, SwiGLU at 81-93 (`int(d*8/3)` rounded to nearest 64). `phase2/model.py` imports RMSNorm directly from phase1 (line 37). MLA confirmed absent from both model files. Pre-existing note: MLA naive cache row (24576) in the comparison table is arithmetically correct but derived, not explicitly in the cited paper's table — minor traceability gap, not an error.

All equations and code snippets in the component file are traceable to their cited sources.
One minor point of note (the MLA source file attribution) and two technical observations are
flagged below, but none rise to the level of a FAIL.

---

## 1. Verified Claims

### 1.1 RoPE — Inverse-frequency formula

**Component claim:**
```
θ_j = 10000^(-2j/d),  j = 0, 1, ..., d/2 - 1
inv_freq[j] = 1 / (10000 ^ (2j / d))
```
**Source:** `sources/papers/rope_2104.09864.md`, "Inverse Frequency (theta) Formula" section.

Verdict: **VERIFIED** — verbatim match in the paper source file.

---

### 1.2 RoPE — Position-dependent transformation (rotation matrix form)

**Component claim:**
```
f_q(q, m) = R_{θ,m} · q
f_k(k, n) = R_{θ,n} · k
[x_{2j}']   = [cos(m·θ_j)  -sin(m·θ_j)] [x_{2j}  ]
[x_{2j+1}']   [sin(m·θ_j)   cos(m·θ_j)] [x_{2j+1}]
```
**Source:** `sources/papers/rope_2104.09864.md`, "Position-Dependent Transformation" section.

Verdict: **VERIFIED** — block-diagonal rotation matrix and per-pair application are verbatim
in the source.

---

### 1.3 RoPE — rotate_half form

**Component claim:**
```
rotate_half(x):  splits x into [x1 | x2], returns [-x2 | x1]
rotated(x, m) = x * cos(m·Θ) + rotate_half(x) * sin(m·Θ)
```
**Source:** `sources/papers/rope_2104.09864.md`, "The rotate_half Form" section;
`sources/code/rope.py` lines 55–68 and 101–102.

Verdict: **VERIFIED** — matches both paper source and code source verbatim.

---

### 1.4 RoPE — Relative position property

**Component claim:**
```
⟨f_q(q, m), f_k(k, n)⟩ = g(q, k, m - n)
R_{θ,m}^T · R_{θ,n} = R_{θ,m-n}
```
**Source:** `sources/papers/rope_2104.09864.md`, "Relative Position Property" section.

Verdict: **VERIFIED** — identical formulation in the paper source.

---

### 1.5 RoPE — Reference code snippet (RotaryEmbedding + rotate_half + apply_rotary_pos_emb)

**Component claim:** Source `sources/code/rope.py`, lines 24–103.

The component reproduces:
- `RotaryEmbedding.__init__` and `_build_cache` (lines 24–52 in source)
- `rotate_half` function (lines 55–68 in source)
- `apply_rotary_pos_emb` function (lines 71–103 in source)

Verdict: **VERIFIED** — all three code blocks match the source file verbatim within the
stated line range. Line range 24–103 is accurate.

---

### 1.6 RMSNorm — Core formula

**Component claim:**
```
RMSNorm(x) = g · x / RMS(x)
RMS(x) = sqrt( (1/d) Σ x_i² + ε )
       = sqrt( x.pow(2).mean(-1, keepdim=True) + ε )
RMSNorm(x) = g · x · rsqrt( mean(x², dim=-1, keepdim=True) + ε )
```
**Source:** `sources/papers/rmsnorm_1910.07467.md`, "RMSNorm Formula" section.

Verdict: **VERIFIED** — verbatim match including both the sqrt and rsqrt forms.

---

### 1.7 RMSNorm — Key differences from LayerNorm

**Component claims:**
- No mean subtraction (re-centering dropped)
- No learnable bias term (typically omitted)
- ε is placed inside the square root, not added after the sqrt

**Source:** `sources/papers/rmsnorm_1910.07467.md`, "Key Differences from LayerNorm" table
and "Epsilon Placement" section.

Verdict: **VERIFIED** — all three bullet points are supported by explicit statements and
a comparison table in the paper source.

---

### 1.8 RMSNorm — Reference code snippet (RMSNormLlama)

**Component claim:** Source `sources/code/rmsnorm.py`, lines 79–102.

The component reproduces `RMSNormLlama` (lines 79–102 in source). The code matches exactly.

Verdict: **VERIFIED** — verbatim match. Line range 79–102 is accurate.

---

### 1.9 SwiGLU — Definition

**Component claim:**
```
SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊙ (xV + c)
where Swish(x) = x · σ(x) = x · sigmoid(x)
```
**Source:** `sources/papers/swiglu_2002.05202.md`, "SwiGLU Definition" section.

Verdict: **VERIFIED** — verbatim match including the Swish/SiLU equivalence.

---

### 1.10 SwiGLU — Full FFN sublayer

**Component claim:**
```
FFN_SwiGLU(x, W1, W2, W3) = (Swish(x @ W1.T) * (x @ W3.T)) @ W2.T
```
with W1 as gate, W3 as up, W2 as output projection.

**Source:** `sources/papers/swiglu_2002.05202.md`, "FFN with SwiGLU" section.

Verdict: **VERIFIED** — verbatim match. The role labels (W1 gate, W3 value/up, W2 down)
also match the paper source's explanatory notes.

---

### 1.11 SwiGLU — Hidden dimension scaling

**Component claim:**
```
hidden_dim = (2/3) × (4 × d_model) = 8/3 × d_model ≈ 2.667 × d_model
Parameter balance: 3 × d × h' = 2 × d × h  →  h' = 2h/3  →  with h = 4d: h' = 8d/3.
```
**Source:** `sources/papers/swiglu_2002.05202.md`, "Dimension Expansion Factor" section.

Verdict: **VERIFIED** — numerical values and algebraic derivation match the source exactly.

---

### 1.12 SwiGLU — Reference code snippet (SwiGLUFeedForward)

**Component claim:** Source `sources/code/swiglu.py`, lines 29–77.

The component reproduces `SwiGLUFeedForward.__init__` and `forward` (lines 29–77 in source).
Code matches exactly.

Verdict: **VERIFIED** — verbatim match. Line range 29–77 is accurate.

---

### 1.13 MLA — Low-rank KV joint compression (Eq. 9–11)

**Component claim:**
```
c_t^{KV} = W^{DKV} h_t     (9)
k_t^C    = W^{UK}  c_t^{KV}  (10)
v_t^C    = W^{UV}  c_t^{KV}  (11)
```
**Source:** `sources/papers/mla_deepseek_v2_2405.04434.md`, "Low-Rank Key-Value Joint
Compression (Equations 9–11)" section.

Verdict: **VERIFIED** — verbatim match including equation numbering.

---

### 1.14 MLA — Query compression (Eq. 12–13)

**Component claim:**
```
c_t^Q = W^{DQ} h_t     (12)
q_t^C = W^{UQ} c_t^Q   (13)
```
**Source:** `sources/papers/mla_deepseek_v2_2405.04434.md`, "Query Compression
(Equations 12–13)" section.

Verdict: **VERIFIED** — verbatim match.

---

### 1.15 MLA — RMSNorm applied after latents for training stability

**Component claim:** "An RMSNorm is applied after c_t^Q and c_t^{KV} for training stability."

**Source:** `sources/papers/mla_deepseek_v2_2405.04434.md`, "Query Compression" section:
"An RMSNorm is applied after c_t^Q (and c_t^{KV}) for training stability."

Verdict: **VERIFIED** — supported verbatim.

---

### 1.16 MLA — Decoupled RoPE (Eq. 14–19)

**Component claim:**
```
q_t^R = RoPE(W^{QR} c_t^Q)                                    (14)
k_t^R = RoPE(W^{KR} h_t)                                       (15)
q_{t,i} = [q_{t,i}^C ; q_{t,i}^R]                             (16)
k_{t,i} = [k_{t,i}^C ; k_t^R]                                  (17)
o_{t,i} = Σ_j Softmax_j( q_{t,i}^T k_{j,i} / sqrt(d_h + d_h^R) ) v_{j,i}^C   (18)
u_t = W^O [o_{t,1}; ...; o_{t,n_h}]                           (19)
```
**Source:** `sources/papers/mla_deepseek_v2_2405.04434.md`, "Decoupled Rotary Position
Embedding (Equations 14–19)" section.

Verdict: **VERIFIED** — equations 14–19 match verbatim including the
`sqrt(d_h + d_h^R)` denominator and the shared k_t^R across all n_h heads.

---

### 1.17 MLA — Explanation of why standard RoPE cannot be used with the absorption trick

**Component claim:** "Standard RoPE cannot be applied to content keys that are reconstructed
from a position-independent latent (the W^{UK} absorption trick requires W^{UK} to be
position-independent)."

**Source:** `sources/papers/mla_deepseek_v2_2405.04434.md`, "Decoupled Rotary Position
Embedding" section: "RoPE applies position-dependent rotation to keys. If keys are
reconstructed from a cached latent c_t^{KV} via W^{UK}, the reconstruction matrix W^{UK}
would be position-dependent ... which breaks the absorption trick."

Verdict: **VERIFIED** — supported by the source explanation.

---

### 1.18 MLA — k_t^R is a single shared key vector across all n_h heads

**Component claim:** "k_t^R is a single shared key vector across all n_h heads; it is
position-dependent and cached separately from c_t^{KV}."

**Source:** `sources/papers/mla_deepseek_v2_2405.04434.md`, "Decoupled RoPE" section:
"k_t^R is a single shared key across all n_h heads (not per-head)."

Verdict: **VERIFIED** — explicitly stated in the paper source.

---

### 1.19 MLA — KV absorption trick description

**Component claim:**
```
q_nope_absorbed = q_nope @ W^{UK}.T
scores = einsum(q_nope_absorbed, c_cache) + einsum(q_pe, pe_cache)
```
**Source:** `sources/papers/mla_deepseek_v2_2405.04434.md`, "KV Absorption Trick" section
and `sources/code/mla_attention.py` lines 157–168 (active code) and 253–258 (verbatim block).

Verdict: **VERIFIED** — the pseudocode is consistent with both the paper source and the
code source.

---

### 1.20 MLA — Cache footprint comparison table

**Component claim:**
| Mechanism | Cache per token |
|-----------|----------------|
| MHA | 2 × n_h × d_h = 32768 elements |
| MLA (naive) | n_h × (d_h + d_h^R) = 24576 elements |
| MLA (absorption) | d_c + d_h^R = 576 elements |

**Source:** `sources/papers/mla_deepseek_v2_2405.04434.md`, "KV Cache Size Comparison"
section. Paper gives: MHA = 2 × 128 × 128 = 32768; MLA = 512 + 64 = 576.

Verdict: **PARTIALLY VERIFIED with note.**

- MHA row (32768) and MLA absorption row (576): **VERIFIED** — numbers match the source
  exactly.
- MLA naive row formula "n_h × (d_h + d_h^R) = 24576": This is the cache cost of the
  naive mode as described by the component (caching full K per head at dimension
  d_h + d_h^R = 192, times n_h = 128 heads = 24576). The paper source does not provide
  this intermediate figure explicitly — the paper's table only lists MHA, GQA, MQA, and
  MLA (absorption). However, the value is arithmetically correct and derivable from the
  paper's stated dimensions. This is a computed intermediate, not a verbatim paper claim.
  **Flag: DERIVED, NOT EXPLICITLY SOURCED** (see Section 3).

---

### 1.21 MLA — GQA-equivalence quote

**Component claim:** "MLA requires only a small amount of KV cache, equal to GQA with only
2.25 groups, but can achieve stronger performance than MHA."

**Source:** `sources/papers/mla_deepseek_v2_2405.04434.md`, final paragraph of "KV Cache
Size Comparison" section (verbatim quote).

Verdict: **VERIFIED** — verbatim match.

---

### 1.22 MLA — Reference code snippet (absorption-mode forward, verbatim block)

**Component claim:** Source `sources/code/mla_attention.py`, lines 186–268, verbatim block
at lines 251–266.

The component excerpt:
```python
# else:  (absorption mode)
#     wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
#     q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
#     self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
#     self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
#     scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
#               torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
```

In `mla_attention.py` the verbatim block starts at line 186 (comment header). The `else:`
absorption branch is at lines 251–258. Checking the source:
- Line 253: `wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)` — matches
- Line 254: `q_nope = torch.einsum(...)` — matches
- Line 255: `self.kv_cache[...] = self.kv_norm(kv)` — matches
- Line 256: `self.pe_cache[...] = k_pe.squeeze(2)` — matches
- Lines 257–258: scores einsum — matches

Verdict: **VERIFIED** — verbatim match. The component correctly notes this is from the
commented verbatim block (as distinct from the active simplified code).

---

### 1.23 Our implementation — RMSNorm (phase1/model.py:30–38)

The component asserts the implementation matches the LLaMA rsqrt form and lists specific
properties (float32 upcast, epsilon inside sqrt via `add(eps).rsqrt()`, no mean subtraction,
no bias). These properties are verifiable against `sources/code/rmsnorm.py` (RMSNormLlama,
lines 79–102).

Verdict: **VERIFIED** — the described `phase1/model.py` implementation is consistent with
the reference. (Note: this report validates the component file against the source references,
not against `phase1/model.py` directly; the alignment claim is within scope as an intra-doc
consistency check.)

---

### 1.24 Our implementation — RoPE (phase1/model.py:41–54), deviation acknowledgment

**Component claims:**
- The `apply_rope` function computes `[x1*cos - x2*sin | x2*cos + x1*sin]` directly.
- This is mathematically equivalent to `x * cos + rotate_half(x) * sin`.
- The cache layout differs from the reference (half-width vs. full-width) but produces
  identical values.

The algebra shown in the component is correct:
```
x * cos + rotate_half(x) * sin
  = [x1 | x2] * [cos | cos] + [-x2 | x1] * [sin | sin]
  = [x1*cos - x2*sin | x2*cos + x1*sin]
```
This is traceable to the rotate_half description in both
`sources/papers/rope_2104.09864.md` and `sources/code/rope.py`.

Verdict: **VERIFIED** — the mathematical equivalence claim is correct and traceable.

---

### 1.25 Our implementation — SwiGLU (phase1/model.py:81–93), alignment multiple deviation

**Component claims:**
- The only difference from `SwiGLUFeedForward` is the alignment multiple (64 vs. 256).
- This has no effect on correctness or the 8/3 parameter ratio.

**Source:** `sources/code/swiglu.py` lines 58–61. The reference uses `multiple_of=256`;
the component uses 64. Both round up `int(8/3 * d)` to a hardware multiple. The 8/3
ratio and the three-matrix structure are unchanged.

Verdict: **VERIFIED** — the claimed deviation is accurately described and does not affect
correctness.

---

### 1.26 MLA — statement that it is not yet implemented

**Component claim:** "MLA is not yet implemented in phase1 or phase2. The phase1 transformer
uses standard CausalSelfAttention (phase1/model.py:57–78) — vanilla MHA with RoPE but no
KV compression."

This is a status claim about the project state, not a claim requiring equation-level
traceability. No contradiction with any source file.

Verdict: **NOT APPLICABLE** (project status claim; no source traceability required).

---

## 2. Unverified / Unsupported Claims

### 2.1 MLA naive-mode cache size (24576 elements)

**Location:** Component file, "Cache footprint comparison" table, row "MLA (naive)".

**Claim:** "MLA (naive) | n_h × (d_h + d_h^R) = 24576 elements"

**Status:** The formula and arithmetic are correct (128 × 192 = 24576), but this figure
does not appear in `sources/papers/mla_deepseek_v2_2405.04434.md`. The paper's comparison
table does not include a naive-mode MLA row. This is an inferential/derived value introduced
by the component author.

**Assessment:** The value is mathematically derivable from the paper's hyperparameters and
is not wrong, but it is presented in a table alongside paper-sourced figures without being
flagged as derived. This is a minor traceability gap, not an error.

**Recommendation:** Annotate the row as "(derived)" or "(computed from paper hyperparameters)".

---

## 3. Contradictions

No internal contradictions were found.

Specific checks performed:
- RoPE frequency formula is stated consistently in the equations section and the "Our
  implementation" section (both use base 10000, same index convention).
- RMSNorm epsilon placement is stated consistently ("inside the square root") in the
  equations section, the key differences table, and the Our implementation section.
- SwiGLU hidden dimension is stated consistently as 8/3 × d_model in the equations
  section and the Our implementation section.
- The component correctly notes that the `apply_rope` code deviates from the `rotate_half`
  helper but is mathematically equivalent — this is acknowledged as an intentional deviation,
  not a contradiction.
- The source attribution for `mla_attention.py` states lines 186–268. The verbatim block
  does start at line 186 (comment header) and the last line of the `return x` statement is
  at line 268. This range is accurate.

---

## 4. Attribution Notes

### 4.1 MLA code file cites DeepSeek-V2 paper but code is from DeepSeek-V3 repo

The component file's "Sources" section cites `mla_attention.py` as "adapted from
`deepseek-ai/DeepSeek-V3/inference/model.py`." The code source file header (line 2)
confirms the code is from the DeepSeek-V3 repository (arXiv:2412.19437), while the MLA
mechanism was originally introduced in DeepSeek-V2 (arXiv:2405.04434). The component's
equation references point to the DeepSeek-V2 paper (which is where equations 9–19 appear),
while the code comes from V3. This split sourcing is handled correctly: equations cite the
V2 paper, code cites the V3 repository. No error; this is accurate.

---

## Summary Table

| # | Claim | Verdict |
|---|-------|---------|
| 1.1 | RoPE inverse-frequency formula | VERIFIED |
| 1.2 | RoPE rotation matrix form | VERIFIED |
| 1.3 | RoPE rotate_half form | VERIFIED |
| 1.4 | RoPE relative position property | VERIFIED |
| 1.5 | RoPE reference code (rope.py L24–103) | VERIFIED |
| 1.6 | RMSNorm core formula | VERIFIED |
| 1.7 | RMSNorm differences from LayerNorm | VERIFIED |
| 1.8 | RMSNorm reference code (rmsnorm.py L79–102) | VERIFIED |
| 1.9 | SwiGLU definition | VERIFIED |
| 1.10 | SwiGLU FFN sublayer | VERIFIED |
| 1.11 | SwiGLU hidden dim scaling | VERIFIED |
| 1.12 | SwiGLU reference code (swiglu.py L29–77) | VERIFIED |
| 1.13 | MLA KV compression Eq. 9–11 | VERIFIED |
| 1.14 | MLA query compression Eq. 12–13 | VERIFIED |
| 1.15 | MLA RMSNorm on latents | VERIFIED |
| 1.16 | MLA decoupled RoPE Eq. 14–19 | VERIFIED |
| 1.17 | MLA RoPE incompatibility with absorption | VERIFIED |
| 1.18 | MLA k_t^R shared across heads | VERIFIED |
| 1.19 | MLA absorption trick description | VERIFIED |
| 1.20 | MLA cache table (MHA + absorption rows) | VERIFIED |
| 1.20 | MLA cache table (naive row) | DERIVED — not explicitly in source |
| 1.21 | MLA GQA-equivalence quote | VERIFIED |
| 1.22 | MLA reference code verbatim block L251–266 | VERIFIED |
| 1.23 | Our RMSNorm matches LLaMA form | VERIFIED |
| 1.24 | Our RoPE mathematical equivalence | VERIFIED |
| 1.25 | Our SwiGLU alignment multiple deviation | VERIFIED |
| 1.26 | MLA not implemented status claim | N/A |

**Total: 25 verified, 1 derived (not sourced but not wrong), 0 contradictions.**

**Overall verdict: PASS**
