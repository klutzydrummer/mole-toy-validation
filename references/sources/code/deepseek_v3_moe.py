# SOURCE: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
# PAPER:  "DeepSeek-V3 Technical Report" — arXiv:2412.19437 (DeepSeek-AI, Dec 2024)
# FETCHED: 2026-03-17
#
# This file contains the MoE routing section verbatim from the official DeepSeek-V3
# inference code, lightly annotated with equation references for cross-checking against
# the paper. Used as ground truth for verifying MoLE MoL FFN routing correctness.
#
# KEY DESIGN POINTS (for MoL FFN verification):
#   1. score_func = "sigmoid"  (not softmax)  — paper Eq. 15
#   2. original_scores = sigmoid(linear(x, weight))  — these are UNBIASED
#   3. bias b_i added to scores ONLY for top-k selection — paper Eq. 16
#   4. weights = original_scores.gather(indices)  — UNBIASED scores used for weight computation
#   5. weights normalized: weights /= weights.sum()  — paper Eq. 13
#   6. weights *= route_scale  — replaces softmax's implicit normalization
#   7. Bias update rule (NOT in inference code — training only):
#      if expert overloaded:  b_i -= gamma
#      if expert underloaded: b_i += gamma
#
# PAPER QUOTES:
#   "Note that the bias term is only used for routing. The gating value, which will be
#    multiplied with the FFN output, is still derived from the original affinity score s_{i,t}."
#
#   "DeepSeek-V3 uses the sigmoid function to compute the affinity scores, and applies a
#    normalization among all selected affinity scores to produce the gating values."

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ============================================================
# Gate — MoE Router with auxiliary-loss-free load balancing
# ============================================================
# Paper section: 2.1 Basic Architecture, "Auxiliary-Loss-Free Load Balancing"
# Implements equations 12-16 from arXiv:2412.19437

class Gate(nn.Module):
    """
    Gating mechanism for MoE routing.

    Routing uses biased scores (s + b) for top-k selection.
    Weight computation uses original (unbiased) scores s.
    Bias b_i is updated outside of backprop based on per-expert load.
    """
    def __init__(self, args):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts          # K_r in paper
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func             # "sigmoid" for DeepSeek-V3
        self.route_scale = args.route_scale
        # Router projection: maps d_model -> n_routed_experts
        # Each row e_i is the centroid vector for expert i (paper Eq. 15: u_t^T e_i)
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        # Auxiliary bias b_i: only created for the 671B model (dim=7168)
        # Updated by sign rule each step, NOT via gradient
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32)) \
            if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch*seq, d_model) — token representations u_t

        Returns:
            weights: (batch*seq, topk) — normalized unbiased gating values g_{i,t}
            indices: (batch*seq, topk) — selected expert indices
        """
        # Step 1: compute logits = u_t^T e_i  (paper Eq. 15, pre-activation)
        scores = F.linear(x, self.weight)             # (B, N_r)

        # Step 2: apply score function — sigmoid for DeepSeek-V3  (paper Eq. 15)
        # s_{i,t} = Sigmoid(u_t^T e_i)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()                 # <-- DeepSeek-V3 path

        # Step 3: save UNBIASED scores — these are used for weight computation (paper Eq. 13)
        # "the gating value ... is still derived from the original affinity score s_{i,t}"
        original_scores = scores

        # Step 4: add per-expert bias ONLY for routing decision  (paper Eq. 16)
        # Selection uses: s_{i,t} + b_i
        # Weight computation uses: s_{i,t}  (NOT s_{i,t} + b_i)
        if self.bias is not None:
            scores = scores + self.bias               # biased scores for selection only

        # Step 5: group-limited routing (node-limited routing in paper)
        # Restricts tokens to at most topk_groups groups of experts
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                # When bias present: use top-2 sum per group for group selection
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            # Mask out non-selected groups with -inf before global top-k
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)

        # Step 6: select top-K_r experts by BIASED score  (paper Eq. 16)
        indices = torch.topk(scores, self.topk, dim=-1)[1]  # (B, K_r)

        # Step 7: gather UNBIASED scores for selected experts  (paper Eq. 13)
        # weights_raw = s_{i,t} for selected i  (NO bias)
        weights = original_scores.gather(1, indices)        # (B, K_r)

        # Step 8: normalize over selected experts  (paper Eq. 13)
        # g_{i,t} = g'_{i,t} / sum_j g'_{j,t}
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)

        # Step 9: apply route_scale (replaces softmax normalization for sigmoid mode)
        weights *= self.route_scale

        return weights.type_as(x), indices

    # ----------------------------------------------------------------
    # Bias update rule — called each training step, NOT via backprop
    # Paper: "we will decrease the bias term by γ if its corresponding
    #         expert is overloaded, and increase it by γ if its
    #         corresponding expert is underloaded"
    # ----------------------------------------------------------------
    # def update_bias(self, expert_load: torch.Tensor, gamma: float):
    #     """
    #     expert_load: (N_r,) float tensor, mean tokens routed per expert this step
    #     gamma: bias update speed hyperparameter
    #     """
    #     if self.bias is None:
    #         return
    #     target_load = expert_load.mean()
    #     overloaded  = expert_load > target_load   # bool mask
    #     underloaded = expert_load < target_load   # bool mask
    #     self.bias.data[overloaded]  -= gamma
    #     self.bias.data[underloaded] += gamma


# ============================================================
# Expert — single routed FFN expert
# ============================================================

class Expert(nn.Module):
    """
    Single FFN expert. In DeepSeek-V3 this is a standard SwiGLU MLP.
    In MoLE, this corresponds to a LoRA adapter over a shared base weight.
    """
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ============================================================
# MoE — full Mixture-of-Experts layer
# ============================================================

class MoE(nn.Module):
    """
    Mixture-of-Experts layer.

    Combines N_s shared experts (always active) with N_r routed experts
    (top K_r selected per token). Paper Equation 12:

        h'_t = u_t + sum_{i=1}^{N_s} FFN_i^(s)(u_t) + sum_{i=1}^{N_r} g_{i,t} FFN_i^(r)(u_t)
    """
    def __init__(self, args):
        super().__init__()
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([
            Expert(args.dim, args.moe_inter_dim)
            for i in range(self.n_routed_experts)
        ])
        self.shared_experts = None  # MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)                        # (B*T, d)

        # Get routing weights and indices (g_{i,t} and selected expert indices)
        weights, indices = self.gate(x)                 # (B*T, K_r), (B*T, K_r)

        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()

        # Accumulate weighted expert outputs  (paper Eq. 12: g_{i,t} * FFN_i^(r)(u_t))
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)        # which tokens selected this expert
            y[idx] += expert(x[idx]) * weights[idx, top, None]

        # Add shared expert output if present  (paper Eq. 12: sum FFN_i^(s)(u_t))
        if self.shared_experts is not None:
            z = self.shared_experts(x)
            y = y + z

        return y.view(shape)
