# Differential Transformer V2

## Citation

**Title:** Differential Transformer V2
**Authors:** Tianzhu Ye, Li Dong, Yutao Sun, Furu Wei (Microsoft Research)
**Published:** January 20, 2026
**Blog post:** https://huggingface.co/blog/microsoft/diff-attn-v2
**GitHub:** https://github.com/microsoft/unilm/blob/master/Diff-Transformer/Diff-Transformer-V2
**Notion:** https://spiky-homegrown-4cb.notion.site/Differential-Transformer-V2-2e7baa052def80ecaa93d4d67d125417
**Fetched:** 2026-03-31
**Status:** Blog post (no arXiv preprint at time of writing); experimental results still running

Note: This is a blog post, not a peer-reviewed paper. It presents results from production-scale
LLM pretraining experiments (dense models and a 30A3 MoE, trained on trillions of tokens).

---

## Abstract

DIFF V2 is an improved version of Differential Transformer (DIFF V1, arXiv:2410.05258).
Three focus areas:

1. **Faster Inference & No Custom Attention Kernels** — Q heads doubled, KV heads unchanged;
   matches baseline Transformer decoding speed; direct use of FlashAttention without modification.
2. **Improved Training Stability** — Removes per-head RMSNorm from V1, which caused gradient
   instability in large-scale pretraining at large learning rates.
3. **Simpler Parameterization & Initialization** — Replaces static exp reparameterization with
   token-specific, head-wise projected λ via sigmoid. No λ_init constants.

---

## V2 Architecture

### Head layout change from V1

| Component | V1 | V2 |
|-----------|----|----|
| Query heads | h/2 pairs (q1, q2 separate) | 2h heads (doubled; contiguous pairs) |
| KV heads | h/2 (with double-wide V) | h (standard V, same as baseline) |
| V width | 2·d_h per head | d_h (standard) |
| λ | static exp reparameterization | token-specific sigmoid(W_λ · x) per head |
| Per-head RMSNorm | yes | **no** |
| λ_init per layer | yes (0.8 − 0.6·exp(−0.3·l)) | **no** |

### Critical implementation note (verbatim from blog)

> "DIFF V2 subtracts two heads that are in the same GQA group (share the same key and value).
> This is crucial to performance."

### Parameter count vs baseline

- W_Q: d → 2·n_heads·d_head = 2d  (+d² vs baseline)
- W_K: d → d  (same)
- W_V: d → d  (same; no longer double-wide)
- W_λ: d → n_heads  (tiny projection, ~1% of total)
- W_O: d → d  (same; output projection unchanged because differential reduces 2h→h)
- **Net: ~+d² (25% extra attention params), ~+1% for W_λ**

---

## Authoritative equations

### DIFF V2 pseudocode (verbatim from blog)

```python
def DiffAttnV2(
        q, k, v, lam
):
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

**Notation:** batch omitted; `flash_attn_func` inputs/outputs are `(tokens, heads, head_dim)`.
Heads in the same GQA group arranged contiguously in output.

### DIFF V1 pseudocode for comparison (verbatim from blog)

```python
def DiffAttnV1(
        layer_index, q1, q2, k1, k2, v,
        lam_q1, lam_k1, lam_q2, lam_k2,
):
    """
    q1, q2: (N, h/2, d)
    k1, k2: (N, h_kv/2, d)
    v:      (N, h_kv/2, 2d)
    lam_*: (d,)
    """
    attn1 = flash_attn_func(q1, k1, v)
    attn2 = flash_attn_func(q2, k2, v)

    lam_init = 0.8 - 0.6 * exp(-0.3 * layer_index)
    lam1 = exp(sum(lam_q1 * lam_k1))
    lam2 = exp(sum(lam_q2 * lam_k2))
    lam = lam1 - lam2 + lam_init
    attn = attn1 - lam * attn2

    attn = rmsnorm(attn)
    attn = attn * (1 - lam_init)
    return attn
```

---

## Softmax magnitude analysis (verbatim from blog)

### Standard SDPA context RMS

For Q, K, V ∈ ℝⁿˣᵈ, softmax attention output **c**ᵢ for token i:

```
cᵢ = Σⱼ aᵢⱼ vⱼ,    Σⱼ aᵢⱼ = 1,  aᵢⱼ ≥ 0
RMS(cᵢ) ∈ [1/√n, 1)
```

- Lower bound 1/√n: uniform attention
- Upper bound 1: attention focused on one token

### DIFF V2 context RMS

```
cᵢ = Σⱼ ( Softmax(zᵢⱼ¹) − sigmoid(λᵢ) · Softmax(zᵢⱼ²) ) vⱼ
RMS(cᵢ) ∈ (0, √2)
```

**Lower bound zero**: particularly important — allows attention sinks to be eliminated
(sink tokens can be fully cancelled by the differential). Standard Transformer lower bound
1/√n forces all tokens to attend to something.

**V1 RMSNorm instability**: At uniform attention over n=8192 tokens, RMS ≈ 1/√8192,
so RMSNorm amplifies by √8192 ≈ 90.5×. This causes massive gradient norms and spikes
during large-scale pretraining. Removed in V2.

---

## Design ablations (from blog)

| Ablation | Result |
|----------|--------|
| Wrong GQA pairing (`[:nh//2]` vs `[0::2]`) | Training instability; higher loss at large LR |
| No λ scaling (`attn1 - attn2`) | Excessively small context RMS at init; higher loss |
| Projected λ without sigmoid (unbounded) | Higher loss; less stable than V2 |
| Transformer with 1.5h heads (param-matched) | Higher loss than V2 |

**Critical correctness note**: `attn[:, 0::2]` (interleaved) is the correct GQA pairing —
each differential pair shares K,V. `attn[:, :nh//2]` (split) is wrong and causes instability.

---

## Experimental results (partial; ongoing at time of publication)

- Experiments: Dense models and 30A3 MoE, trained on trillions of tokens
- Learning rates: 6e-4 to 1e-3
- Language modeling loss: DIFF V2 notably lower than Transformer baseline
  (gap of 0.02–0.03 at 1T training tokens)
- Gradient norm: Reduced spikes vs Transformer, especially at large LRs
- Activation outliers: Reduced magnitude

---

## Relationship to related work

| Method | Context RMS range |
|--------|-------------------|
| Standard Softmax | [1/√n, 1) |
| Attention Is Off By One | (0, 1) |
| gpt-oss (learnable scalar s per head) | (0, 1) |
| Gated Attention (element-wise sigmoid) | (0, 1) |
| **DIFF V2** | **(0, √2)** |

DIFF V2's upper bound of √2 is larger than (0,1), but the lower bound of 0 is the key
benefit: standard softmax cannot produce a zero output, forcing attention sinks.
