# DeepSeek-V2: Multi-Head Latent Attention (MLA)

## Citation

**Title:** DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
**Authors:** DeepSeek-AI (156 co-authors)
**ArXiv:** https://arxiv.org/abs/2405.04434
**Submitted:** May 7, 2024; revised June 19, 2024
**BibTeX key:** deepseek-ai2024deepseekv2

```bibtex
@article{deepseekv2,
  title   = {DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model},
  author  = {DeepSeek-AI},
  journal = {arXiv preprint arXiv:2405.04434},
  year    = {2024}
}
```

---

## Abstract (verbatim)

"We present DeepSeek-V2, a strong Mixture-of-Experts (MoE) language model characterized by
economical training and efficient inference. It comprises 236B total parameters, of which 21B are
activated for each token, and supports a context length of 128K tokens. DeepSeek-V2 adopts
innovative architectures including Multi-head Latent Attention (MLA) and DeepSeekMoE.
MLA significantly compresses the Key-Value (KV) cache into latent vectors, ensuring efficient
inference. ... Despite activating only 21B parameters, DeepSeek-V2 achieves top-tier performance
among open-source models."

---

## Section: Multi-Head Latent Attention (MLA)

MLA is introduced as a replacement for standard Multi-Head Attention (MHA) with the goal of
dramatically reducing the KV cache footprint at inference while preserving or improving model
quality.

### Standard MHA Baseline (Equations 1–8)

For a token at position t with input hidden state **h**_t ∈ ℝ^d:

```
q_t = W^Q h_t                                                  (1)
k_t = W^K h_t                                                  (2)
v_t = W^V h_t                                                  (3)

[q_{t,1}; q_{t,2}; ...; q_{t,n_h}] = q_t                     (4)
[k_{t,1}; k_{t,2}; ...; k_{t,n_h}] = k_t                     (5)
[v_{t,1}; v_{t,2}; ...; v_{t,n_h}] = v_t                     (6)

o_{t,i} = sum_{j=1}^{t} Softmax_j( q_{t,i}^T k_{j,i} / sqrt(d_h) ) v_{j,i}   (7)

u_t = W^O [o_{t,1}; o_{t,2}; ...; o_{t,n_h}]                 (8)
```

Where:
- n_h = number of attention heads
- d_h = per-head dimension
- W^Q, W^K, W^V ∈ ℝ^{d × d}  (with d = n_h · d_h)
- W^O ∈ ℝ^{d × d}

Standard MHA requires caching 2 · n_h · d_h elements per token per layer.

---

### Low-Rank Key-Value Joint Compression (Equations 9–11)

The key innovation: instead of caching full K and V, compress them through a low-rank latent
vector **c**_t^{KV} ∈ ℝ^{d_c} where d_c ≪ n_h · d_h.

```
c_t^{KV} = W^{DKV} h_t                                        (9)
k_t^C    = W^{UK}  c_t^{KV}                                   (10)
v_t^C    = W^{UV}  c_t^{KV}                                   (11)
```

Where:
- W^{DKV} ∈ ℝ^{d_c × d}         (down-projection: input → latent)
- W^{UK}  ∈ ℝ^{n_h d_h × d_c}   (up-projection: latent → keys)
- W^{UV}  ∈ ℝ^{n_h d_h × d_c}   (up-projection: latent → values)
- d_c is the KV compression dimension (d_c = 4 d_h for DeepSeek-V2, i.e., 512)

**Only c_t^{KV} needs to be cached** (not the expanded K and V), giving a cache of d_c elements
per token per layer instead of 2 · n_h · d_h.

---

### Query Compression (Equations 12–13)

Queries are also compressed through a low-rank bottleneck (primarily to reduce activation memory
during training):

```
c_t^Q = W^{DQ} h_t                                            (12)
q_t^C = W^{UQ} c_t^Q                                          (13)
```

Where:
- W^{DQ} ∈ ℝ^{d_c' × d}          (down-projection)
- W^{UQ} ∈ ℝ^{n_h d_h × d_c'}    (up-projection)
- d_c' = 1536 for DeepSeek-V2

An RMSNorm is applied after c_t^Q (and c_t^{KV}) for training stability.

---

### Decoupled Rotary Position Embedding (Equations 14–19)

**Problem with standard RoPE + MLA:** RoPE applies position-dependent rotation to keys. If keys
are reconstructed from a cached latent c_t^{KV} via W^{UK}, the reconstruction matrix W^{UK}
would be position-dependent (it would need to commute with RoPE), which breaks the absorption
trick below. Solution: decouple positional and content components.

**Decoupled RoPE equations:**

```
[q_{t,1}^R; q_{t,2}^R; ...; q_{t,n_h}^R] = q_t^R = RoPE(W^{QR} c_t^Q)     (14)
k_t^R = RoPE(W^{KR} h_t)                                                     (15)

q_{t,i} = [q_{t,i}^C ; q_{t,i}^R]                                           (16)
k_{t,i} = [k_{t,i}^C ; k_t^R]                                               (17)

o_{t,i} = sum_{j=1}^{t} Softmax_j( q_{t,i}^T k_{j,i} / sqrt(d_h + d_h^R) ) v_{j,i}^C   (18)

u_t = W^O [o_{t,1}; o_{t,2}; ...; o_{t,n_h}]                               (19)
```

Where:
- W^{QR} ∈ ℝ^{n_h d_h^R × d_c'}    projects query latent to per-head RoPE vectors
- W^{KR} ∈ ℝ^{d_h^R × d}           projects input directly to a SHARED RoPE key vector
- d_h^R = d_h / 2 = 64 for DeepSeek-V2 (the decoupled per-head dimension)
- RoPE(·) applies rotary position embedding
- k_t^R is a **single shared key** across all n_h heads (not per-head)
- The attention score denominator uses sqrt(d_h + d_h^R) for the full concatenated head dim

**Key insight:** k_t^R is computed directly from h_t (not from the latent c_t^{KV}), so it is
position-dependent and cached separately. The content component k_t^C comes from c_t^{KV} and
remains position-independent, enabling the absorption trick.

**What is cached at inference:**
- c_t^{KV} ∈ ℝ^{d_c}    (content latent, position-independent)
- k_t^R   ∈ ℝ^{d_h^R}   (shared RoPE key, position-dependent)

Total cache per token per layer: (d_c + d_h^R) elements.

---

### KV Absorption Trick (Inference Efficiency)

The paper notes that W^{UK} (and W^{UV}) can be **absorbed into W^Q** (and W^O) at inference time:

- Instead of computing k_{t,i}^C = W^{UK} c_t^{KV} and then doing q^T k, compute:
  q_nope_absorbed = q_nope @ W^{UK}.T  →  then score = q_nope_absorbed · c_t^{KV}
- Similarly for values: instead of v = W^{UV} c, compute output in latent space then project once.

This means **W^{UK} and W^{UV} are never applied to c during inference** — they are folded into
the query and output projections. Only c_t^{KV} (shape: d_c) and k_t^R (shape: d_h^R) are stored
in the KV cache.

From the paper: "during inference, since W^{UK} can be absorbed into W^Q, and W^{UV} can be
absorbed into W^O, we even do not need to compute keys and values out for attention."

---

### KV Cache Size Comparison

| Mechanism | KV cache per token (elements) | Notes |
|-----------|-------------------------------|-------|
| MHA       | 2 · n_h · d_h · L             | Full K and V per head |
| GQA (g groups) | 2 · g · d_h · L          | Shared K/V within group |
| MQA       | 2 · d_h · L                   | Single K/V shared |
| **MLA**   | **(d_c + d_h^R) · L**         | Latent + shared RoPE key |

For DeepSeek-V2 with n_h=128, d_h=128, d_c=512, d_h^R=64:
- MHA cache: 2 × 128 × 128 = 32768 elements per token
- MLA cache: 512 + 64 = 576 elements per token
- **Reduction: ~93.3% fewer KV cache elements vs. MHA**
- Equivalent to GQA with only ~2.25 groups, but with stronger performance than MHA

From the paper: "MLA requires only a small amount of KV cache, equal to GQA with only 2.25
groups, but can achieve stronger performance than MHA."

---

## DeepSeek-V2 MLA Hyperparameters

| Symbol | Description | Value |
|--------|-------------|-------|
| L      | Number of layers | 60 |
| d      | Hidden dimension | 5120 |
| n_h    | Number of attention heads | 128 |
| d_h    | Per-head dimension | 128 |
| d_c    | KV compression dimension | 512 (= 4 d_h) |
| d_c'   | Query compression dimension | 1536 |
| d_h^R  | Decoupled RoPE per-head dim | 64 (= d_h / 2) |

---

## Initialization and Training Details

- **Standard deviation:** Parameters initialized with σ = 0.006
- **RMSNorm after latents:** Additional RMSNorm layers are applied after the compressed latent
  vectors c_t^{KV} and c_t^Q to ensure training stability at the bottleneck.
- **Scaling factors:** Applied at width bottlenecks.
- The query LoRA path (W^{DQ} → RMSNorm → W^{UQ}) mirrors the KV compression structure and
  serves primarily to reduce activation memory during training (not inference KV cache).

---

## Summary: What MLA Caches vs. Computes

**Training / naive inference (expand everything):**
1. c_t^{KV} = W^{DKV} h_t                      — compress to latent
2. k_t^C    = W^{UK}  c_t^{KV}                  — expand to content keys
3. v_t^C    = W^{UV}  c_t^{KV}                  — expand to values
4. k_t^R    = RoPE(W^{KR} h_t)                  — shared positional key
5. q_{t,i}  = [W^{UQ} c_t^Q ; RoPE(W^{QR} c_t^Q)]  — combined query
6. scores and output as per Eq. 18–19

**Efficient inference (absorption mode):**
- Cache only: (c_t^{KV}, k_t^R) per position
- At query time: fold W^{UK} into query projection (q_nope ← q_nope @ W^{UK}^T)
- Attention scores: einsum(q_nope_absorbed, c_cache) + einsum(q_pe, pe_cache)
- Output: einsum(scores, c_cache) → fold W^{UV} once per forward pass
