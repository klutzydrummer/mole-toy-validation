# Component: MoL FFN (Mixture-of-LoRAs Feed-Forward Network)

## Component

**Name:** MoL FFN — Mixture-of-LoRAs Feed-Forward Network

**Description:** A sparse-expert FFN where each "expert" is a rank-4 LoRA adapter over
shared base weights, routed using DeepSeek-V3 sigmoid-based gating with unbiased weight
composition and auxiliary-loss-free load balancing.

---

## Sources

**Papers:**
- `sources/papers/deepseek_v3_2412.19437.md`
  — DeepSeek-V3 Technical Report, arXiv:2412.19437 (DeepSeek-AI, Dec 2024)
  — Equations 12–16 define the routing algorithm used verbatim.

**Code:**
- `sources/code/deepseek_v3_moe.py`
  — Extracted from `https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py`
  — `Gate` class is the direct reference for routing logic.

---

## Authoritative equations

All equations verbatim from: **DeepSeek-V3 Technical Report** (arXiv:2412.19437), Section 2.1.

### Equation 12 — FFN output combining shared and routed experts

```
h'_t = u_t + sum_{i=1}^{N_s} FFN_i^(s)(u_t) + sum_{i=1}^{N_r} g_{i,t} FFN_i^(r)(u_t)
```

Variables:
- `u_t` — FFN input at position t
- `h'_t` — FFN output
- `N_s` — number of shared experts (always active)
- `N_r` — number of routed experts (pool)
- `g_{i,t}` — gating value for routed expert i at token t (Eq. 13)

### Equation 13 — Gating value: normalize over UNBIASED scores of selected experts only

```
g_{i,t} = g'_{i,t} / sum_{j=1}^{N_r} g'_{j,t}
```

`g'_{i,t}` is nonzero only for the K_r selected experts; normalization is over
selected experts only (not the full pool).

### Equation 14 — Top-K selection without load-balance bias (base rule)

```
g'_{i,t} = { s_{i,t},  if s_{i,t} in TopK({s_{j,t} | 1 <= j <= N_r}, K_r)
           { 0,         otherwise
```

### Equation 15 — Affinity score via sigmoid (not softmax)

```
s_{i,t} = Sigmoid(u_t^T e_i)
```

Variables:
- `s_{i,t}` — token-to-expert affinity score, scalar in [0, 1]
- `u_t` — FFN input vector for token t
- `e_i` — router weight row for expert i

Paper quote (verbatim):
> "Slightly different from DeepSeek-V2, DeepSeek-V3 uses the sigmoid function to compute
> the affinity scores, and applies a normalization among all selected affinity scores to
> produce the gating values."

Sigmoid scores are independent per-expert — they do not compete in a zero-sum way as
they would under softmax.

### Equation 16 — Top-K selection WITH load-balance bias (used during training)

```
g'_{i,t} = { s_{i,t},  if s_{i,t} + b_i in TopK({s_{j,t} + b_j | 1 <= j <= N_r}, K_r)
           { 0,         otherwise
```

Variables:
- `b_i` — per-expert auxiliary bias (not gradient-trained)

**Critical distinction (verbatim):**
> "Note that the bias term is only used for routing. The gating value, which will be
> multiplied with the FFN output, is still derived from the original affinity score s_{i,t}."

That is:
- **Selection** (which experts to activate): uses `s_{i,t} + b_i`  (biased)
- **Weight computation** (how much each expert contributes): uses `s_{i,t}`  (unbiased)

### Bias update rule — auxiliary-loss-free load balancing

From paper Section 2.1 (verbatim):
> "During training, we keep monitoring the expert load on the whole batch of each training
> step. At the end of each step, we will decrease the bias term by γ if its corresponding
> expert is overloaded, and increase it by γ if its corresponding expert is underloaded,
> where γ is a hyper-parameter called bias update speed."

```
for each expert i at end of training step:
    if expert_i is overloaded:
        b_i -= gamma
    elif expert_i is underloaded:
        b_i += gamma
```

Updated via heuristic sign rule, NOT via gradient. No auxiliary loss needed.

---

## Reference implementation

**Source:** `sources/code/deepseek_v3_moe.py`
**Attribution:** DeepSeek-AI, `https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py`
**Fetched:** 2026-03-17

### Gate.forward — routing with unbiased weight composition

```python
def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Step 1+2: compute logits and apply sigmoid  (paper Eq. 15)
    scores = F.linear(x, self.weight)             # (B, N_r)
    scores = scores.sigmoid()                     # s_{i,t} = Sigmoid(u_t^T e_i)

    # Step 3: save UNBIASED scores for weight computation  (paper Eq. 13)
    original_scores = scores

    # Step 4: add bias ONLY for routing decision  (paper Eq. 16)
    if self.bias is not None:
        scores = scores + self.bias               # biased scores for selection only

    # Step 6: select top-K experts by BIASED score  (paper Eq. 16)
    indices = torch.topk(scores, self.topk, dim=-1)[1]

    # Step 7: gather UNBIASED scores for selected experts  (paper Eq. 13)
    weights = original_scores.gather(1, indices)  # NOT scores (unbiased)

    # Step 8: normalize over selected experts  (paper Eq. 13)
    if self.score_func == "sigmoid":
        weights /= weights.sum(dim=-1, keepdim=True)

    # Step 9: apply route_scale
    weights *= self.route_scale

    return weights.type_as(x), indices
```

Key structural notes from the reference `Gate.__init__`:
- Router weight: `nn.Parameter(torch.empty(n_routed_experts, dim))` — one row `e_i` per expert
- Bias: `nn.Parameter(torch.empty(n_routed_experts, dtype=torch.float32))` — one scalar per expert
- Bias is a `nn.Parameter` in the reference but is intended to be updated outside backprop;
  in the reference code the bias update function is commented out (inference-only file)

### Expert — single routed FFN in DeepSeek-V3

```python
class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

In DeepSeek-V3, each expert is a full independent SwiGLU MLP. Our MoL FFN replaces this
with rank-4 LoRA adapters over a single shared base weight — a fundamental structural
difference described in the next section.

---

## Our implementation

**File:** `phase1/model.py`

| Class | Lines | Role |
|---|---|---|
| `LoRAAdapter` | 214–224 | Single rank-r LoRA: `output = (x @ A @ B) * scale` |
| `MoLFFN` | 227–332 | Full MoL FFN layer |
| `MoLFFN.forward` | 274–318 | Routing + composition + load balance update |

### LoRAAdapter (phase1/model.py:214–224)

B matrix initialized to zeros (`torch.zeros(rank, d_out)`) so the adapter is a no-op at
initialization. A matrix initialized with normal distribution scaled by `1/sqrt(d_in)`.
Scale factor is `1/rank`.

### MoLFFN routing (phase1/model.py:274–287)

Follows the DeepSeek-V3 Gate pattern exactly:

```python
logits = self.router(x)                             # [B, L, n_experts]
scores = torch.sigmoid(logits)                      # unbiased affinity (Eq. 15)
biased = scores + self.expert_bias                  # biased for selection only (Eq. 16)
_, topk_idx = biased.topk(self.top_k, dim=-1)      # select with biased scores (Eq. 16)
topk_scores = scores.gather(2, topk_idx)            # gather UNBIASED scores (Eq. 13)
topk_weights = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-8)  # normalize (Eq. 13)
```

One-hot construction for differentiable gradient flow (phase1/model.py:285–287):
```python
one_hot = F.one_hot(topk_idx, self.n_experts).to(x.dtype)  # [B, L, k, n_experts]
expert_weights = (one_hot * topk_weights.unsqueeze(-1)).sum(dim=2)  # [B, L, n_experts]
```

### MoLFFN composition (phase1/model.py:289–306)

Weight-space composition over all three SwiGLU projections:

```
W_eff · x = W_base · x + ΔW_shared · x + Σ_k w_k · (ΔW_k · x)
```

Applied before the nonlinearity for gate and up, and after for down:

```python
gate = self.base_gate(x) + self.shared_gate(x)
up   = self.base_up(x)   + self.shared_up(x)
for e in range(self.n_experts):
    w_e = expert_weights[:, :, e].unsqueeze(-1)
    gate = gate + w_e * self.expert_gate[e](x)
    up   = up   + w_e * self.expert_up[e](x)
hidden = F.silu(gate) * up                         # nonlinearity ONCE on combined projections
output = self.base_down(hidden) + self.shared_down(hidden)
for e in range(self.n_experts):
    w_e = expert_weights[:, :, e].unsqueeze(-1)
    output = output + w_e * self.expert_down[e](hidden)
```

### Load balance update (phase1/model.py:308–317)

```python
if self.training:
    with torch.no_grad():
        counts = torch.zeros(self.n_experts, device=x.device)
        for k in range(self.top_k):
            counts.scatter_add_(0, topk_idx[:, :, k].reshape(-1),
                                torch.ones(B * L, device=x.device))
        avg = counts.mean()
        self.expert_bias += self.bias_step * (avg - counts).sign()
```

This is the DeepSeek-V3 sign rule applied per-batch. `bias_step = 0.01` (γ); DeepSeek-V3
used 0.001 — scaled up for toy experiments (noted in source comment at line 272).

### Intentional deviations from DeepSeek-V3

| Deviation | Reason |
|---|---|
| Experts are rank-4 LoRA adapters over shared base, not full independent MLPs | Parameter efficiency; MoLE design (architecture v0.3.1 Sections 7.1, 8) |
| Three expert sets (gate, up, down) per layer instead of one per expert | LoRA correction applied to each SwiGLU projection independently |
| Shared LoRA (always active) alongside conditional experts | Analogous to DeepSeek-V3 shared experts (N_s), but implemented as LoRA not full MLP |
| `expert_bias` stored as `register_buffer`, not `nn.Parameter` | Prevents gradient flow through the bias; matches intended design (bias is heuristic-updated, not learned) |
| `bias_step = 0.01` instead of DeepSeek-V3's `0.001` | Toy scale requires faster load-balance correction |
| No node/group-limited routing | Single-machine toy model; group routing is a distributed training optimization |
| Normalization denominator has `+ 1e-8` epsilon | Numerical stability guard; absent in reference but safe addition |
| Router is `nn.Linear(d, n_experts, bias=False)` | No router bias; router bias is a separate `expert_bias` buffer |
| Default: 8 experts, top-2, rank 4 | Per architecture spec; DeepSeek-V3 uses 256 routed experts, top-8 |

---

## Verification checklist

1. **Sigmoid not softmax**: Confirm `torch.sigmoid(logits)` is used at `model.py:279`, not `F.softmax`.

2. **Bias for selection only**: Confirm `biased = scores + self.expert_bias` at line 280 is used only for `topk()` at line 281, and that `scores.gather(2, topk_idx)` at line 282 gathers from the original `scores` tensor, not `biased`.

3. **Normalization over selected experts**: Confirm `topk_scores.sum(dim=-1, keepdim=True)` at line 283 sums only the top-k scores, not all N_r scores.

4. **B matrix zero init**: Confirm `LoRAAdapter.__init__` initializes `self.B = nn.Parameter(torch.zeros(rank, d_out))` at line 220. At initialization, every LoRA adapter must be a no-op (output = 0).

5. **One-hot gradient flow**: Confirm `F.one_hot(topk_idx, ...)` at line 286 is used so that gradients flow through `topk_weights` (the soft weights), not through the discrete index selection.

6. **Bias not gradient-trained**: Confirm `self.expert_bias` is a `register_buffer` (not `nn.Parameter`) and that the update at line 316 runs inside `torch.no_grad()`.

7. **Load balance sign rule matches paper**: Confirm the bias update is `bias += bias_step * (avg - counts).sign()`, which increases bias for underloaded experts and decreases it for overloaded ones — matching the paper's stated rule.

8. **Composition order**: Confirm LoRA corrections for gate and up projections are accumulated before `F.silu()` is applied (line 299), not after. Down projection corrections are applied after.

9. **Scale factor**: Confirm `LoRAAdapter.scale = 1.0 / rank` at line 221. With rank=4, scale=0.25.

10. **Shared adapter always active**: Confirm `self.shared_gate(x)`, `self.shared_up(x)`, `self.shared_down(hidden)` are added unconditionally (lines 290–292, 302) regardless of routing outcome.

11. **Expert weight zero-sum check**: After training for at least 100 steps, run `model.get_mol_stats()` (line 321) and verify `expert_balance` > 0.7 (entropy ratio), confirming load balancing is functioning.

12. **No bias in router linear**: Confirm `self.router = nn.Linear(d, n_experts, bias=False)` at line 267. Router bias is handled separately via `expert_bias` buffer, not baked into the linear layer.
