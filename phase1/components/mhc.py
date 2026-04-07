"""
mHC: Manifold-Constrained Hyper-Connections (arXiv:2512.24880).
KromHC H_res factorization (arXiv:2601.21579).

Spec: references/components/mhc.md
Sources: references/sources/papers/mhc_2512.24880.md,
         references/sources/code/mhc_hyper_connections.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Full mHC layer update (arXiv:2512.24880, tokenbender/mHC):
#
#   x_{l+1} = H_res · x_l + H_post^T · F(H_pre · x_l, W_l)
#
# Three matrices per sub-layer:
#   H_res  [n, n] - doubly stochastic, mixes existing streams
#   H_pre  [n]    - non-negative, combines streams into branch input
#   H_post [n]    - non-negative, distributes branch output per-stream
#
# H_res implementation: KromHC (arXiv:2601.21579) — exact doubly stochastic via
# Kronecker-product factorization. For n=4=2×2: two 2×2 factors, each parameterized
# as a single scalar a ∈ (0,1) via softmax: U = a·I + (1-a)·Swap. The Kronecker
# product of two doubly stochastic matrices is doubly stochastic (Theorem 4.2).
# Replaces the approximate Sinkhorn-Knopp from original mHC. No iterations needed.
# Init: factor_logits = [0, -8] → softmax → a≈1 → U≈I → H_res = I⊗I ≈ I_4.


class KromHCResidual(nn.Module):
    """
    Exact doubly stochastic H_res for n=4 via Kronecker-product factorization.
    References: arXiv:2601.21579 (KromHC), Theorem 4.2 + Eq. 14.
    Factorization: H_res = U1 ⊗ U2, each U_k a 2×2 doubly stochastic matrix.

    For n=2: 2! = 2 permutation matrices (I, Swap).
    Each factor: U_k = a_k·I + (1-a_k)·Swap, a_k = softmax([logit_0, logit_1])[0].
    Kronecker product: (2×2) ⊗ (2×2) → 4×4, exactly doubly stochastic.

    Static variant only (input-independent). Dynamic extension not implemented here.
    """

    def __init__(self):
        super().__init__()
        # Two factor logit pairs — each length-2 vector over {identity, swap}.
        # Init [0, -8]: softmax → [≈1, ≈0] → a≈1 → U≈I → H_res = I⊗I ≈ I_4.
        self.factor1_logits = nn.Parameter(torch.tensor([0.0, -8.0]))
        self.factor2_logits = nn.Parameter(torch.tensor([0.0, -8.0]))
        # Register as buffers so .to(device) moves them once and torch.compile can cache them.
        self.register_buffer("I2", torch.eye(2))
        self.register_buffer("S2", torch.tensor([[0., 1.], [1., 0.]]))

    def forward(self):
        """Returns a 4×4 exactly doubly stochastic matrix."""
        dtype = self.factor1_logits.dtype
        I2 = self.I2.to(dtype=dtype)
        S2 = self.S2.to(dtype=dtype)

        a1 = F.softmax(self.factor1_logits, dim=0)[0]  # scalar ∈ (0,1)
        a2 = F.softmax(self.factor2_logits, dim=0)[0]

        U1 = a1 * I2 + (1 - a1) * S2  # 2×2 doubly stochastic
        U2 = a2 * I2 + (1 - a2) * S2  # 2×2 doubly stochastic

        # Kronecker product: U1 ⊗ U2 = 4×4 exactly doubly stochastic
        return torch.kron(U1, U2)  # [4, 4]


class HyperConnection(nn.Module):
    """
    Single mHC connection wrapping one branch (attention or FFN).

    Implements: x_{l+1} = H_res · x_l + H_post^T · F(H_pre · x_l)

    H_res is computed via KromHCResidual (arXiv:2601.21579): exact doubly stochastic
    by Kronecker-product factorization. Requires n=4 (factored as 2×2).

    H_pre and H_post use softmax (not sigmoid as in paper Eq.8), matching reference
    code tokenbender/mHC-manifold-constrained-hyper-connections.

    Args:
        n: number of streams — must be 4 (KromHC factorization is 2×2)
        d: model dimension (unused here; kept for API symmetry)
    """

    def __init__(self, n: int, d: int):
        super().__init__()
        assert n == 4, f"KromHC requires n=4 (got n={n}). Factorization is fixed as 2×2."
        self.n = n
        self.d = d

        # H_res: exact doubly stochastic via KromHC (replaces approximate Sinkhorn).
        self.krom_res = KromHCResidual()

        # H_pre: combines n streams → 1 branch input (non-negative via softmax).
        # Init: one randomly chosen stream = 0, rest = -8 → near one-hot selector.
        # Matches tokenbender/mHC H_pre_logits init (init_residual_index pattern).
        init_h_pre = torch.full((n,), -8.0)
        init_h_pre[torch.randint(n, (1,)).item()] = 0.0
        self.pre_logits = nn.Parameter(init_h_pre)

        # H_post: distributes branch output → n streams (non-negative via softmax).
        # Init: zeros → uniform softmax (1/n per stream) at start.
        # Matches tokenbender/mHC H_post_logits init.
        self.post_logits = nn.Parameter(torch.zeros(n))

    def forward(self, streams, branch_fn):
        """
        streams: [B, L, n, d]
        branch_fn: callable that takes [B, L, d] and returns [B, L, d]
        returns: [B, L, n, d]
        """
        B, L, n, d = streams.shape

        # ---- H_res: mix existing streams (exact doubly stochastic, KromHC) ----
        H_res = self.krom_res()  # [4, 4], exactly doubly stochastic
        mixed = torch.einsum("ij, bljd -> blid", H_res, streams)

        # ---- H_pre: combine streams into branch input ----
        pre_weights = F.softmax(self.pre_logits, dim=0)  # [n], non-negative, sums to 1
        branch_input = torch.einsum("n, blnd -> bld", pre_weights, streams)  # [B, L, d]

        # ---- Branch function (attention or FFN) ----
        branch_output = branch_fn(branch_input)  # [B, L, d]

        # ---- H_post: distribute branch output per-stream ----
        # softmax (not softplus): normalized to sum to 1, matching tokenbender/mHC.
        post_weights = F.softmax(self.post_logits, dim=0)  # [n], sums to 1
        # H_post^T · branch_output: each stream gets branch_output * post_weights[i]
        distributed = branch_output.unsqueeze(2) * post_weights.view(1, 1, n, 1)  # [B, L, n, d]

        # ---- Combine: H_res · x + H_post^T · F(H_pre · x) ----
        return mixed + distributed
