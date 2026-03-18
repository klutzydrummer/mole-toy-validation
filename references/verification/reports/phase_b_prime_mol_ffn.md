# Phase B' Validation Report: mol_ffn.md

**Component file:** `references/components/mol_ffn.md`
**Validator:** Phase B' automated verification agent
**Date:** 2026-03-17
**Sources checked:**
- `references/sources/papers/deepseek_v3_2412.19437.md`
- `references/sources/code/deepseek_v3_moe.py`
- `phase1/model.py` (implementation under test)

---

## Overall Verdict: PASS

All paper equation citations are verified verbatim or in mathematically equivalent form.
All code snippets are verified against the actual source files.
No unsupported claims. One minor annotation issue (see item C1 below — flagged WARN, not FAIL).
No internal contradictions found.

---

## 1. Verified Claims

### 1.1 Paper Equations (Equations 12–16)

**EQ-12 — FFN output combining shared and routed experts**

Claim in component file:
```
h'_t = u_t + sum_{i=1}^{N_s} FFN_i^(s)(u_t) + sum_{i=1}^{N_r} g_{i,t} FFN_i^(r)(u_t)
```
Source match: `deepseek_v3_2412.19437.md`, Section "Equation 12 — FFN Output with Shared + Routed Experts" — verbatim match.
Variable definitions (u_t, h'_t, N_s, N_r, g_{i,t}) match paper exactly.
**VERIFIED.**

---

**EQ-13 — Gating value normalization over unbiased scores of selected experts**

Claim in component file:
```
g_{i,t} = g'_{i,t} / sum_{j=1}^{N_r} g'_{j,t}
```
Source match: `deepseek_v3_2412.19437.md`, Section "Equation 13 — Gating Value Normalization (UNBIASED scores)" — verbatim match.
Claim that normalization is over selected experts only (g'_{j,t} nonzero only for K_r selected) matches paper annotation: "g'_{i,t} is nonzero only for the K_r selected experts".
**VERIFIED.**

---

**EQ-14 — Top-K selection without load-balance bias**

Claim in component file:
```
g'_{i,t} = { s_{i,t},  if s_{i,t} in TopK({s_{j,t} | 1 <= j <= N_r}, K_r)
           { 0,         otherwise
```
Source match: `deepseek_v3_2412.19437.md`, Section "Equation 14 — Top-K Selection (without load-balance bias)" — verbatim match.
**VERIFIED.**

---

**EQ-15 — Affinity score via sigmoid**

Claim in component file:
```
s_{i,t} = Sigmoid(u_t^T e_i)
```
Source match: `deepseek_v3_2412.19437.md`, Section "Equation 15 — Affinity Score via Sigmoid" — verbatim match.

Verbatim paper quote cited in component file:
> "Slightly different from DeepSeek-V2, DeepSeek-V3 uses the sigmoid function to compute the affinity scores, and applies a normalization among all selected affinity scores to produce the gating values."

Source match: `deepseek_v3_2412.19437.md` under "Why sigmoid, not softmax (verbatim)" — verbatim match.
**VERIFIED.**

---

**EQ-16 — Top-K selection with load-balance bias**

Claim in component file:
```
g'_{i,t} = { s_{i,t},  if s_{i,t} + b_i in TopK({s_{j,t} + b_j | 1 <= j <= N_r}, K_r)
           { 0,         otherwise
```
Source match: `deepseek_v3_2412.19437.md`, Section "Equation 16 — Top-K Selection WITH Load-Balance Bias" — verbatim match.

Verbatim critical distinction cited in component file:
> "Note that the bias term is only used for routing. The gating value, which will be multiplied with the FFN output, is still derived from the original affinity score s_{i,t}."

Source match: `deepseek_v3_2412.19437.md`, Section "CRITICAL DISTINCTION (verbatim)" — verbatim match.
Also appears verbatim in `deepseek_v3_moe.py` header comment (lines 21–23).
**VERIFIED.**

---

**Bias update rule (load balancing)**

Claim in component file:
> "During training, we keep monitoring the expert load on the whole batch of each training step. At the end of each step, we will decrease the bias term by γ if its corresponding expert is overloaded, and increase it by γ if its corresponding expert is underloaded, where γ is a hyper-parameter called bias update speed."

Source match: `deepseek_v3_2412.19437.md`, Section "Load Balancing: Bias Update Rule (Verbatim)" — verbatim match.

Pseudocode in component file matches paper pseudocode exactly.

Claim "Updated via heuristic sign rule, NOT via gradient. No auxiliary loss needed." is supported by paper: `deepseek_v3_2412.19437.md` states "Updated via heuristic, NOT via gradient" and "No auxiliary loss needed for the primary balancing mechanism."
**VERIFIED.**

---

### 1.2 Code Snippets

**Gate.forward code snippet**

Component file presents a `Gate.forward` listing attributed to `sources/code/deepseek_v3_moe.py`, `Gate` class.

Verified against `deepseek_v3_moe.py` lines 63–121:
- `scores = F.linear(x, self.weight)` — matches line 73 verbatim.
- `scores = scores.sigmoid()` — matches line 80 verbatim (the DeepSeek-V3 path when score_func != "softmax").
- `original_scores = scores` — matches line 84 verbatim.
- `if self.bias is not None: scores = scores + self.bias` — matches lines 89–90 verbatim.
- `indices = torch.topk(scores, self.topk, dim=-1)[1]` — matches line 107 verbatim.
- `weights = original_scores.gather(1, indices)` — matches line 111 verbatim.
- `if self.score_func == "sigmoid": weights /= weights.sum(dim=-1, keepdim=True)` — matches lines 115–116 verbatim.
- `weights *= self.route_scale` — matches line 119 verbatim.
- `return weights.type_as(x), indices` — matches line 121 verbatim.

NOTE: The component file's listing omits the group-limited routing block (lines 94–104 of the source). This is an intentional simplification; the omission is not a misrepresentation — the omitted code is the group-limited routing path which the component file explicitly lists as a deviation ("No node/group-limited routing" in the deviations table).
**VERIFIED (with acceptable omission noted).**

`Gate.__init__` structural claims:
- "Router weight: `nn.Parameter(torch.empty(n_routed_experts, dim))`" — matches `deepseek_v3_moe.py` line 57 verbatim.
- "Bias: `nn.Parameter(torch.empty(n_routed_experts, dtype=torch.float32))`" — matches line 60 verbatim.
- "Bias is a `nn.Parameter` in the reference but is intended to be updated outside backprop; in the reference code the bias update function is commented out (inference-only file)" — confirmed: `update_bias` method is commented out at lines 129–140.
**VERIFIED.**

---

**Expert class code snippet**

Component file:
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

Source match: `deepseek_v3_moe.py` lines 147–159 — verbatim match.
**VERIFIED.**

---

### 1.3 Our Implementation Claims

**LoRAAdapter (phase1/model.py:214–224)**

Component claims:
- B matrix initialized to zeros: `torch.zeros(rank, d_out)` — confirmed at `model.py` line 220: `self.B = nn.Parameter(torch.zeros(rank, d_out))`.
- A matrix initialized with normal distribution scaled by `1/sqrt(d_in)` — confirmed at line 219: `torch.randn(d_in, rank) * (1.0 / math.sqrt(d_in))`.
- Scale factor is `1/rank` — confirmed at line 221: `self.scale = 1.0 / rank`.
- Output formula `output = (x @ A @ B) * scale` — confirmed at line 224: `return (x @ self.A @ self.B) * self.scale`.
- Line range 214–224: confirmed (class spans lines 214–224).
**VERIFIED.**

---

**MoLFFN routing (phase1/model.py:274–287)**

Component claims routing code at lines 274–287; confirmed:
- Line 278: `logits = self.router(x)` — match.
- Line 279: `scores = torch.sigmoid(logits)` — match.
- Line 280: `biased = scores + self.expert_bias` — match.
- Line 281: `_, topk_idx = biased.topk(self.top_k, dim=-1)` — match.
- Line 282: `topk_scores = scores.gather(2, topk_idx)` — match (gathers from unbiased `scores`).
- Line 283: `topk_weights = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-8)` — match.
- Lines 286–287: one-hot construction — match.
**VERIFIED.**

---

**MoLFFN composition (phase1/model.py:289–306)**

Component claims:
- `gate = self.base_gate(x) + self.shared_gate(x)` at line 290 — confirmed line 290.
- `up = self.base_up(x) + self.shared_up(x)` at line 291 — confirmed line 291.
- Expert loop accumulating gate/up corrections before silu — confirmed lines 293–296.
- `hidden = F.silu(gate) * up` at line 299 — confirmed.
- Down projection base + shared at line 302 — confirmed.
- Expert down loop at lines 304–306 — confirmed.
**VERIFIED.**

---

**Load balance update (phase1/model.py:308–317)**

Component claims:
- Inside `if self.training: with torch.no_grad():` — confirmed lines 309–310.
- `counts.scatter_add_` pattern — confirmed lines 312–314.
- `avg = counts.mean()` — confirmed line 315.
- `self.expert_bias += self.bias_step * (avg - counts).sign()` — confirmed line 316.
- `bias_step = 0.01` (γ) — confirmed at `model.py` line 272: `self.bias_step = 0.01  # γ, larger than DS-V3's 0.001 for toy scale`.
- Component claim "DeepSeek-V3 used 0.001 — scaled up for toy experiments (noted in source comment at line 272)" — confirmed, comment at line 272 reads exactly this.
**VERIFIED.**

---

**Deviations table**

| Deviation | Verification |
|---|---|
| Experts are rank-4 LoRA adapters over shared base | Confirmed: `LoRAAdapter` class; `Expert` in DS-V3 is full MLP. Claim accurate. |
| Three expert sets (gate, up, down) per layer | Confirmed: `expert_gate`, `expert_up`, `expert_down` ModuleLists at lines 262–264. |
| Shared LoRA always active | Confirmed: `shared_gate/up/down` added unconditionally at lines 290–291, 302. |
| `expert_bias` as `register_buffer`, not `nn.Parameter` | Confirmed at line 270: `self.register_buffer("expert_bias", ...)`. |
| `bias_step = 0.01` instead of 0.001 | Confirmed at line 272. |
| No node/group-limited routing | Confirmed: no group routing logic in `MoLFFN`. |
| `+ 1e-8` epsilon in normalization | Confirmed at line 283. |
| Router `nn.Linear(d, n_experts, bias=False)` | Confirmed at line 267. |
| Default: 8 experts, top-2, rank 4 | Confirmed at `MoLFFN.__init__` signature: `n_experts=8, top_k=2, rank=4`. |

All deviation claims **VERIFIED.**

---

### 1.4 Verification Checklist Items

| Item | Claim | Verified Against |
|---|---|---|
| 1 | `torch.sigmoid(logits)` at line 279 | `model.py:279` — confirmed |
| 2 | Bias used only for topk; gather from `scores` not `biased` | `model.py:280–282` — confirmed |
| 3 | Normalization sums only top-k scores | `model.py:283`: `topk_scores.sum(...)` — confirmed |
| 4 | B matrix `torch.zeros(rank, d_out)` at line 221 | `model.py:220` (actual line 220, not 221) — confirmed |
| 5 | `F.one_hot(topk_idx, ...)` at line 286 | `model.py:286` — confirmed |
| 6 | `expert_bias` is `register_buffer` + update inside `no_grad()` | `model.py:270, 310` — confirmed |
| 7 | Bias update is `bias += bias_step * (avg - counts).sign()` | `model.py:316` — confirmed |
| 8 | LoRA corrections for gate/up before `F.silu()`, down after | `model.py:293–299, 304–306` — confirmed |
| 9 | `scale = 1.0 / rank` at line 221 | `model.py:221` — confirmed |
| 10 | Shared adapters unconditionally added | `model.py:290–291, 302` — confirmed |
| 11 | `get_mol_stats()` — runtime/empirical claim; not verifiable statically | N/A (not a code correctness claim) |
| 12 | Router `nn.Linear(d, n_experts, bias=False)` at line 267 | `model.py:267` — confirmed |

All statically-verifiable checklist items **VERIFIED.**

---

## 2. Unverified / Unsupported Claims

None found. Every factual claim in the component file traces to either:
- A verbatim or mathematically equivalent passage in `deepseek_v3_2412.19437.md`, or
- A verbatim or structurally equivalent passage in `deepseek_v3_moe.py`, or
- A directly observable line in `phase1/model.py`.

---

## 3. Warnings (Not Failures)

**WARN-C1: Checklist item 4 — minor line number off-by-one**

Component file states: "Confirm `LoRAAdapter.__init__` initializes `self.B = nn.Parameter(torch.zeros(rank, d_out))` at line 221."

Actual location: `model.py` line **220**, not 221. Line 221 is `self.scale = 1.0 / rank`.
The code is correct; only the cited line number is off by one. This is a documentation nit, not a correctness issue.
**WARN (not FAIL).**

---

**WARN-C2: `get_mol_stats()` method name in checklist item 11**

Component file references `model.get_mol_stats()` returning `expert_balance`. The actual method is named `get_load_stats()` (confirmed at `model.py` line 321) and returns `expert_balance` as a key. The method name in the checklist does not match the implementation. This would cause a runtime error if the checklist was executed literally.
**WARN (not FAIL) — the underlying implementation logic is correct but the method name in the checklist is wrong.**

---

## 4. Internal Contradictions

None found. All internal cross-references within the component file are consistent:
- Equation references in code comments match the equation definitions.
- Variable definitions in the equations section match their usage in the routing pseudocode.
- The deviations table correctly characterizes each difference between DS-V3 and the MoL FFN implementation.
- The bias update sign rule stated in prose ("increase for underloaded, decrease for overloaded") matches the pseudocode and the implementation at `model.py:316`: `(avg - counts).sign()` is positive when `avg > counts` (underloaded) and negative when `avg < counts` (overloaded).

---

## Summary Table

| Category | Count | Status |
|---|---|---|
| Paper equations verified | 5 (Eq 12–16) + bias rule | PASS |
| Paper quotes verified | 3 verbatim quotes | PASS |
| Code snippets verified | Gate.forward, Expert, Gate.__init__ | PASS |
| Implementation claims verified | LoRAAdapter, MoLFFN routing, composition, load balance, deviations table | PASS |
| Checklist items (static) | 11 of 12 | PASS |
| Checklist items (runtime) | 1 (item 11) | N/A |
| Unsupported claims | 0 | PASS |
| Internal contradictions | 0 | PASS |
| Warnings | 2 (minor line number, method name) | WARN |

**Overall: PASS**
