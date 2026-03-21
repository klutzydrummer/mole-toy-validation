# DLCM: Dynamic Large Concept Models

## Full Citation

**Title:** Dynamic Large Concept Models: Latent Reasoning in an Adaptive Semantic Space
**Authors:** Xingwei Qu, Shaowen Wang, Zihao Huang, Kai Hua, Fan Yin, Rui-Jie Zhu, Jundong Zhou, Qiyang Min, Zihao Wang, Yizhi Li, Tianyu Zhang, He Xing, Zheng Zhang, Yuxuan Song, Tianyu Zheng, Zhiyuan Zeng, Chenghua Lin, Ge Zhang, Wenhao Huang
**Affiliations:** ByteDance, University of Manchester, Mila, Tsinghua University
**arXiv ID:** 2512.24617
**Submitted:** December 31, 2025 (v1); January 5, 2026 (v2)
**URL:** https://arxiv.org/abs/2512.24617
**Official Code:** None linked in paper.

---

## Abstract

DLCM addresses how LLMs apply uniform computation across all tokens despite language's non-uniform information density. The framework "learns semantic boundaries from latent representations and shifts computation from tokens to a compressed concept space." Key contributions include a "compression-aware scaling law" and a parametrization enabling hyperparameter transfer across different compression regimes. At compression ratio R=4, the approach achieves "+2.69% average improvement across 12 zero-shot benchmarks under matched inference FLOPs."

---

## Architecture Overview

DLCM is a four-stage hierarchical pipeline:

### Equation (1) — Encoding

```
H = ℰ(x)
```

A standard causal Transformer encoder processes raw tokens x = [x₁, …, xL] to produce fine-grained representations H of dimension d_token.

### Equation (2) — Segmentation and Pooling

```
C = Φ(H)
```

Variable-length segments are detected and pooled into concept vectors C.

### Equation (3) — Concept Reasoning

```
Z = 𝓜(C)
```

A high-capacity Transformer backbone performs deep reasoning on the compressed concept sequence C.

### Equation (4) — Decoding

```
ŷ = 𝒟(Ψ(H, Z))
```

The decoder reconstructs token-level predictions through causal cross-attention Ψ between token representations H and concept representations Z.

---

## Boundary Detection: Learned Routing

### Equation (5) — Query-Key Projection

```
qₜ = W_q hₜ
kₜ = W_k hₜ
```

### Equation (6) — Boundary Probability via Cosine Dissimilarity

```
pₜ = (1 − cos(qₜ₋₁, kₜ)) / 2 = (1/2)(1 − (qₜ₋₁ᵀ kₜ) / (‖qₜ₋₁‖₂ ‖kₜ‖₂))
```

This measures normalized dissimilarity between consecutive token projections. Note the index convention: q is from the *previous* token (t−1), k is from the *current* token (t). A value near 1.0 indicates high dissimilarity (likely boundary); near 0.0 indicates continuity.

**Threshold rule (inference):**
```
bₜ = [pₜ ≥ 0.5]
```

**Training:** Boundaries use stochastic sampling: `bₜ ~ Bernoulli(pₜ^sharp)` where probabilities are temperature-sharpened to encourage exploration.

**Sequence start / Position 0:** "We enforce p₁ = 1 so that the first token always starts a new concept." This is the only special case for position 0.

---

## Concept Formation: Mean Pooling

### Equation (7) — Concept Formation via Mean Pooling

```
c_k^raw = (1/|Sₖ|) ∑_{t ∈ Sₖ} hₜ
c_k = W_up · c_k^raw
```

Tokens within segment Sₖ are mean-pooled then projected upward from d_token to d_concept dimension. This is the pooling strategy used in DLCM (not EMA-based like H-Net Zone D).

---

## Compression Regularizer: Auxiliary Load-Balancing Loss

### Equation (8) — Global Expected Boundary Rate

```
G_global = (1/|𝒯|) ∑_{(i,t) ∈ 𝒯} pᵢ,ₜ
```

### Equation (9) — Global Actual Boundary Rate

```
F_global = (1/|𝒯|) ∑_{(i,t) ∈ 𝒯} bᵢ,ₜ
```

### Equation (10) — Auxiliary Load-Balancing Loss (Compression Regularizer)

```
ℒ_aux = (R / (R−1)) · [(R−1) · F_global · G_global + (1−F_global) · (1−G_global)] − 1
```

where R is the target compression ratio (R=4 is the primary target in experiments).

**Note:** This loss is structurally identical to H-Net's ℒ_ratio (Eq. 10), with R in place of N. The −1 offset means the loss is 0 at the target operating point F = G = 1/R.

### Equation (15) — Total Training Loss

```
ℒ = ℒ_CE + λ · ℒ_aux
```

No explicit value for λ is stated in the paper.

---

## Gradient Flow Through the Router

- "Gradient Conflict in Learned Boundary Prediction" is explicitly named (Eq. 24):

```
∇_θ ℒ_total = ∇_θ ℒ_CE        +    λ · ∇_θ ℒ_aux
               (anti-compression)     (pro-compression)
```

The paper notes these two gradient signals are in tension: the language modeling loss tends to encourage finer granularity (more tokens), while the auxiliary loss enforces the target compression ratio. There is no stop-gradient or gradient isolation mentioned — gradients from ℒ_CE flow through the boundary predictor.

---

## Decoding: Causal Cross-Attention

### Equation (11) — Concept Smoothing

```
Z̃ = 𝒮(Z)
```

### Equation (12) — Cross-Attention Query Projection

```
Q = H W_Q,    where W_Q ∈ ℝ^{d_token × d_head}
```

### Equation (13) — Cross-Attention Key/Value Projections

```
K = Z̃ W_K,    V = Z̃ W_V,    where W_{K,V} ∈ ℝ^{d_concept × d_head}
```

### Equation (14) — Causal Cross-Attention Output

```
Ψ(H, Z) = Softmax((Q Kᵀ / √d_head) + M) V W_O + H
```

The mask M enforces causal ordering. Note the residual connection: +H is added to the cross-attention output.

### Equation (16) — QK Normalization

```
Q' = RMSNorm(Q),    K' = RMSNorm(K)
```

### Equation (17) — Concept Replication for Efficient Cross-Attention

```
K̃ = repeat_interleave(K, segment_lengths)
Ṽ = repeat_interleave(V, segment_lengths)
```

This converts the L×M attention into standard L×L causal attention compatible with FlashAttention, by repeating each concept's key/value to match the token positions within that segment.

---

## Scaling and Parametrization

### Equation (18) — Width Multipliers for μP

```
s_token = d_token / d_base
s_concept = d_concept / d_base
```

### Equation (19) — Learning Rate Scaling for Encoder/Decoder

```
η_{ℰ,𝒟} = η^base_token · s_token^{-1}
```

### Equation (20) — Learning Rate Scaling for Concept Backbone

```
η_𝓜 = η^base_concept · s_concept^{-1}
```

### Equation (21) — Output Scaling for Logits

```
logits = (1/s_token) · (h_final W_unemb^T)
```

### Equation (22) — Compression-Aware Scaling Law

```
L(N, D, R, P) = E₀ + A_token / (N(1−P) + t_token)^δ₁
                   + A_concept R^γ / (NP + t_concept)^δ₂
                   + A_data / (D + t_data)^α
```

### Equation (23) — Decay-Phase Power Law

```
Δ_decay = k · L_stable^a · R^b · N^c
```

---

## Initialization

"All hidden linear weights W ∈ ℝ^{d_out × d_in} are initialized with variance σ²_base · s⁻¹" and "Embedding weights use fixed σ²_base." No other initialization details are specified (no zero-init of residuals, no identity init of W_q/W_k mentioned — contrast with H-Net which uses identity init for W_q and W_k and zero-init for the residual projection).

---

## Section 7.2.2 — Loss Distribution Analysis (U-Shaped Loss)

Section 7.2.2 ("Loss Distribution Analysis") presents Figure 7, which shows a distinct
**U-shaped improvement pattern** across relative positions within concepts when comparing
DLCM against a token-level baseline:

- **Boundary positions** (start/end of concept): DLCM shows clear improvement — the model
  learns to allocate more capacity at semantic boundaries.
- **Mid-concept positions** (~positions 4–15): performance is mixed; some degradation observed.
  "The presence of red bars in certain mid-concept regions suggests that the compression
  mechanism forces the model to trade off some fine-grained token-level precision."
- **Explanation**: "The concept model sacrifices uniform token-level predictability (resulting
  in minor degradation at specific internal positions) to gain superior performance at semantic
  boundaries and structurally critical tokens."

This U-shaped pattern is the empirical motivation for Zone D's gated residual design in our
implementation: the `(1 - p_j)` weighting gives non-boundary tokens a stronger skip connection
through `encoder_out`, compensating for the mid-chunk precision loss.

**Note on section citation**: This analysis is in Section 7.2 / 7.2.2, NOT Section 4.2.
Prior citations of "DLCM Section 4.2" in this codebase were incorrect and have been fixed.

---

## Key Design Quotes

- "We detect these 'semantic breaks' by measuring local dissimilarity between adjacent tokens."
- "Unlike LCM's fixed sentence boundaries, these boundaries emerge from the model's own latent space through end-to-end optimization."
- "We enforce p₁ = 1 so that the first token always starts a new concept."
- "We use a hard thresholding rule: bₜ = [pₜ ≥ 0.5]" (at inference).
- "Reasoning is inherently hierarchical: humans reason over abstract units such as ideas or concepts before committing to surface realizations."
- At R=4: "reallocates roughly one-third of inference compute into a higher-capacity reasoning backbone."

---

## Relationship to H-Net

DLCM's boundary detection (Eq. 5–6) is nearly identical to H-Net's routing module (H-Net Eq. 4), with minor index convention differences:

| | H-Net | DLCM |
|---|---|---|
| q index | current token t | previous token t−1 |
| k index | previous token t−1 | current token t |
| Cosine formula | `(1 − cos(qₜ, kₜ₋₁)) / 2` | `(1 − cos(qₜ₋₁, kₜ)) / 2` |
| Effectively | same dissimilarity | same dissimilarity |
| W_q/W_k init | identity matrix | unspecified |
| p₁ | forced to 1.0 | forced to 1.0 |
| Threshold | bₜ = [pₜ ≥ 0.5] | bₜ = [pₜ ≥ 0.5] |
| Ratio loss | ℒ_ratio (Eq. 10) | ℒ_aux (Eq. 10), structurally identical |
| Zone D analog | EMA + Mamba-2 scan | mean pooling + cross-attention |
| STE | yes (Eq. 7) | no |
