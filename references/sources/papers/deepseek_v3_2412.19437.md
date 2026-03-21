# DeepSeek-V3 Technical Report — MoE Routing Reference

## Full Citation

**Title:** DeepSeek-V3 Technical Report
**Authors:** DeepSeek-AI (contact: research@deepseek.com)
**arXiv:** 2412.19437
**DOI:** https://doi.org/10.48550/arXiv.2412.19437
**Submitted:** December 27, 2024; last revised February 18, 2025
**URL:** https://arxiv.org/abs/2412.19437
**HTML version:** https://arxiv.org/html/2412.19437v2
**Subject areas:** cs.CL, cs.AI

---

## Abstract (Verbatim)

"We present DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token. To achieve efficient inference and cost-effective training, DeepSeek-V3 adopts Multi-head Latent Attention (MLA) and DeepSeekMoE architectures, which were thoroughly validated in DeepSeek-V2. Furthermore, DeepSeek-V3 pioneers an auxiliary-loss-free strategy for load balancing and sets a multi-token prediction training objective for stronger performance. We pre-train DeepSeek-V3 on 14.8 trillion diverse and high-quality tokens, followed by Supervised Fine-Tuning and Reinforcement Learning stages to fully harness its capabilities. Comprehensive evaluations reveal that DeepSeek-V3 outperforms other open-source models and achieves performance comparable to leading closed-source models. Despite its excellent performance, DeepSeek-V3 requires only 2.788M H800 GPU hours for its full training. In addition, its training process is remarkably stable. Throughout the entire training process, we did not experience any irrecoverable loss spikes or perform any rollbacks. The model checkpoints are available at https://github.com/deepseek-ai/DeepSeek-V3."

---

## Section: DeepSeekMoE Architecture (Section 2.1)

### Model Configuration

DeepSeek-V3 has:
- **671B** total parameters, **37B** activated per token
- Each MoE FFN layer has **N_s** shared experts + **N_r** routed experts
- **K_r** routed experts are activated per token (top-K selection)
- Group-limited routing: experts divided into groups, top-M nodes selected

---

## All MoE Routing Equations (Verbatim)

### Equation 12 — FFN Output with Shared + Routed Experts

```
h'_t = u_t + sum_{i=1}^{N_s} FFN_i^(s)(u_t) + sum_{i=1}^{N_r} g_{i,t} FFN_i^(r)(u_t)
```

**Variables:**
- `u_t` — FFN input token representation at position t
- `h'_t` — FFN output token representation
- `N_s` — number of shared experts (always active)
- `N_r` — number of routed experts (total pool)
- `g_{i,t}` — gating value for routed expert i at token t (defined in Eq. 13)
- `FFN_i^(s)` — shared expert FFN
- `FFN_i^(r)` — routed expert FFN

### Equation 13 — Gating Value Normalization (UNBIASED scores)

```
g_{i,t} = g'_{i,t} / sum_{j=1}^{N_r} g'_{j,t}
```

**Key point:** Normalization is over the UNBIASED affinity scores of selected experts only.
`g'_{i,t}` is nonzero only for the K_r selected experts (from Eq. 14 or 16).

### Equation 14 — Top-K Selection (without load-balance bias)

```
g'_{i,t} = { s_{i,t},  if s_{i,t} in TopK({s_{j,t} | 1 <= j <= N_r}, K_r)
           { 0,         otherwise
```

This is the base selection rule (no bias). The biased version used during training is Eq. 16.

### Equation 15 — Affinity Score via Sigmoid

```
s_{i,t} = Sigmoid(u_t^T e_i)
```

**Variables:**
- `s_{i,t}` — token-to-expert affinity score (scalar in [0,1])
- `u_t` — FFN input for token t (vector)
- `e_i` — centroid/weight vector for routed expert i (the router weight row)

**Why sigmoid, not softmax (verbatim):**
> "Slightly different from DeepSeek-V2, DeepSeek-V3 uses the sigmoid function to compute the affinity scores, and applies a normalization among all selected affinity scores to produce the gating values."

Sigmoid enables independent per-expert scoring — scores are not forced to sum to 1 across all experts. This avoids the zero-sum competition inherent to softmax.

### Equation 16 — Top-K Selection WITH Load-Balance Bias

```
g'_{i,t} = { s_{i,t},  if s_{i,t} + b_i in TopK({s_{j,t} + b_j | 1 <= j <= N_r}, K_r)
           { 0,         otherwise
```

**Variables:**
- `b_i` — per-expert bias term (scalar, maintained per expert)

**CRITICAL DISTINCTION (verbatim):**
> "Note that the bias term is only used for routing. The gating value, which will be multiplied with the FFN output, is still derived from the original affinity score s_{i,t}."

That is:
- **Routing decision** (which experts to pick): uses `s_{i,t} + b_i`  (biased)
- **Weight computation** (how much each expert contributes): uses `s_{i,t}` (unbiased)

This separation prevents load-balancing from degrading model quality.

---

## Load Balancing: Bias Update Rule (Verbatim)

> "During training, we keep monitoring the expert load on the whole batch of each training step. At the end of each step, we will decrease the bias term by γ if its corresponding expert is overloaded, and increase it by γ if its corresponding expert is underloaded, where γ is a hyper-parameter called bias update speed."

**Pseudocode:**
```
for each expert i at end of training step:
    if expert_i is overloaded:
        b_i -= gamma
    elif expert_i is underloaded:
        b_i += gamma
```

**Properties:**
- Updated via heuristic, NOT via gradient
- Decouples load balancing from the model loss
- No auxiliary loss needed for the primary balancing mechanism

---

## Complementary Sequence-Wise Balance Loss (Equations 17–20)

Context (verbatim):
> "Although DeepSeek-V3 mainly relies on the auxiliary-loss-free strategy for load balance, to prevent extreme imbalance within any single sequence, we also employ a complementary sequence-wise balance loss"

### Equation 17 — Balance Loss

```
L_Bal = alpha * sum_{i=1}^{N_r} f_i * P_i
```

where alpha is "assigned an extremely small value" to minimize performance impact.

### Equation 18 — Expert Selection Frequency

```
f_i = (N_r / (K_r * T)) * sum_{t=1}^{T} 1(s_{i,t} in TopK({s_{j,t} | 1<=j<=N_r}, K_r))
```

- `T` — number of tokens in the sequence
- Normalizes so that uniform routing gives f_i = 1 for all i

### Equation 19 — Normalized Affinity Score (for loss only)

```
s'_{i,t} = s_{i,t} / sum_{j=1}^{N_r} s_{j,t}
```

Used only inside the balance loss, not for routing or weight computation.

### Equation 20 — Mean Normalized Affinity

```
P_i = (1/T) * sum_{t=1}^{T} s'_{i,t}
```

---

## Node-Limited Routing (Verbatim)

> "Like the device-limited routing used by DeepSeek-V2, DeepSeek-V3 also uses a restricted routing mechanism to limit communication costs during training. In short, we ensure that each token will be sent to at most M nodes, which are selected according to the sum of the highest K_r/M affinity scores of the experts distributed on each node."

---

## Summary: Routing Algorithm (Step by Step)

1. Compute logits: `logits_i = u_t^T e_i` for all N_r experts
2. Compute unbiased affinity scores: `s_{i,t} = sigmoid(logits_i)`  [Eq. 15]
3. Compute biased scores for selection: `s_{i,t} + b_i`  [Eq. 16]
4. (Optional) Apply group-limited / node-limited routing mask
5. Select top-K_r experts by biased score: `indices = topk(s + b, K_r)`
6. Gather **unbiased** scores for selected experts: `weights_raw = s[indices]`
7. Normalize: `weights = weights_raw / sum(weights_raw)`  [Eq. 13]
8. Scale: `weights *= route_scale`
9. Combine: `output = sum_i weights_i * FFN_i(u_t)`  [Eq. 12]
10. At end of batch step: update biases by ±gamma based on load  [Eq. 16 note]

---

## Relevance to MoLE MoL FFN Routing

The MoL FFN in MoLE follows DeepSeek-V3 routing semantics:
- Use **sigmoid** (not softmax) for per-expert scores
- Keep a per-expert **auxiliary bias** for load balancing, updated by sign rule (not gradient)
- Use **unbiased scores** for weight composition (bias only affects selection)
- Router weights: standard linear layer (`nn.Linear(d_model, n_experts)`, no special init mentioned in paper)

See `references/sources/code/deepseek_v3_moe.py` for a cleaned reference implementation.

---

## Fetched: 2026-03-17
