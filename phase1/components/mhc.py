"""
mHC: Manifold-Constrained Hyper-Connections (arXiv:2512.24880).
go-mHC: Generalized Orthostochastic H_res (arXiv:2604.02309).

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
# H_res implementation: go-mHC (arXiv:2604.02309) — input-conditional, exactly doubly
# stochastic via Cayley transform + block Frobenius projection. Replaces KromHC
# (arXiv:2601.21579) which was static and limited to n=4.
#
# Pipeline per token position (arXiv:2604.02309, Section 3.3):
#   Z = x_agg @ W_res + b_res        project mean-stream to skew params
#   A = skew(Z)                       ns×ns skew-symmetric (ns = n*s)
#   Q = (I + A)^{-1}(I - A)          Cayley → Q ∈ SO(ns)
#   H_res[i,j] = (1/s)||Q_{ij}||²_F  block Frobenius → exactly doubly stochastic
#
# Init: W_res=0, b_res=0 → A=0 → Q=I → H_res=I_n (identity at start).
# Recommended s=2 (arXiv:2604.02309 Section 4).


class GoMHCResidual(nn.Module):
    """
    Exactly doubly stochastic H_res via generalized orthostochastic matrices.
    Reference: arXiv:2604.02309 (go-mHC), Section 3.3.

    Input-conditional: H_res depends on the current stream state (unlike static KromHC).

    Construction per forward call:
      1. x_agg = mean(streams, dim=n)         [B, L, d]   — aggregate streams
      2. Z = W_res(x_agg)                     [B, L, ns*(ns-1)//2]
      3. A = skew(Z)                           [B, L, ns, ns] skew-symmetric
      4. Q = solve(I+A, I-A)                  [B, L, ns, ns] ∈ SO(ns)
      5. H_res[i,j] = (1/s)||Q[i*s:(i+1)*s, j*s:(j+1)*s]||²_F  → [B, L, n, n]

    H_res is exactly doubly stochastic for all inputs:
      - Row sums = 1: each s-row block of Q is orthonormal → ||block_row||²_F = s
      - Col sums = 1: same argument on columns (Q orthogonal)
      - Non-negative: squared norms ≥ 0

    Args:
        n: number of streams
        d: model dimension
        s: expressivity parameter (default 2; s=1 → orthostochastic, s→∞ → full Birkhoff)
    """

    def __init__(self, n: int, d: int, s: int = 2):
        super().__init__()
        self.n = n
        self.s = s
        ns = n * s
        n_skew = ns * (ns - 1) // 2  # free params in ns×ns skew-symmetric matrix

        # Project mean-pooled streams to skew-symmetric matrix params.
        # Zero init: W_res=0, b_res=0 → Z=0 → A=0 → Q=I → H_res=I_n at start.
        self.W_res = nn.Linear(d, n_skew, bias=True)
        nn.init.zeros_(self.W_res.weight)
        nn.init.zeros_(self.W_res.bias)

        # Upper-triangular (row < col) indices for the ns×ns skew-symmetric matrix.
        idx = torch.triu_indices(ns, ns, offset=1)
        self.register_buffer("skew_row", idx[0])
        self.register_buffer("skew_col", idx[1])
        self.register_buffer("I_ns", torch.eye(ns))

    def _skew(self, z):
        """
        z: [B, L, n_skew] → [B, L, ns, ns] skew-symmetric.
        A[..., i, j] = z[..., k] and A[..., j, i] = -z[..., k]
        for each upper-triangular (i,j) pair at index k.
        """
        ns = self.n * self.s
        B, L, _ = z.shape
        A = torch.zeros(B, L, ns, ns, dtype=z.dtype, device=z.device)
        A[:, :, self.skew_row, self.skew_col] = z
        A[:, :, self.skew_col, self.skew_row] = -z
        return A

    def forward(self, streams):
        """
        streams: [B, L, n, d]
        Returns H_res: [B, L, n, n], exactly doubly stochastic.
        """
        B, L, n, d = streams.shape
        ns = n * self.s

        # Aggregate streams → [B, L, d]
        x_agg = streams.mean(dim=2)

        # Project to skew-symmetric params
        z = self.W_res(x_agg)  # [B, L, n_skew]

        # Build skew-symmetric matrix A: [B, L, ns, ns]
        A = self._skew(z)

        # Cayley transform: Q = (I+A)^{-1}(I-A) ∈ SO(ns)
        # torch.linalg.solve(M, B) solves M @ X = B → X = M^{-1} B
        # (I+A) @ Q = (I-A)  →  Q = (I+A)^{-1}(I-A)
        # (I+A) is always non-singular: A skew-symmetric → eigenvalues purely imaginary → -1 ∉ spectrum
        dtype = A.dtype
        eye = self.I_ns.to(dtype=dtype, device=A.device).expand(B, L, ns, ns)
        Q = torch.linalg.solve(eye + A, eye - A)  # [B, L, ns, ns]

        # Block Frobenius projection → [B, L, n, n] exactly doubly stochastic.
        # Reshape Q as [B, L, n, s, n, s]: Q_blocks[b,l,i,:,j,:] = Q[b,l,i*s:(i+1)*s, j*s:(j+1)*s]
        Q_blocks = Q.reshape(B, L, n, self.s, n, self.s)
        H_res = (Q_blocks ** 2).sum(dim=(3, 5)) / self.s  # [B, L, n, n]

        return H_res


class HyperConnection(nn.Module):
    """
    Single mHC connection wrapping one branch (attention or FFN).

    Implements: x_{l+1} = H_res · x_l + H_post^T · F(H_pre · x_l)

    H_res is computed via GoMHCResidual (arXiv:2604.02309): input-conditional,
    exactly doubly stochastic via Cayley transform + block Frobenius projection.
    Works for any n (unlike KromHC which required n=4=2×2).

    H_pre and H_post use softmax (not sigmoid as in paper Eq.8), matching reference
    code tokenbender/mHC-manifold-constrained-hyper-connections.

    Args:
        n: number of streams
        d: model dimension
    """

    def __init__(self, n: int, d: int):
        super().__init__()
        self.n = n
        self.d = d

        # H_res: input-conditional exactly doubly stochastic via go-mHC.
        self.go_res = GoMHCResidual(n, d, s=2)

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

        # ---- H_res: mix existing streams (input-conditional, exactly doubly stochastic) ----
        H_res = self.go_res(streams)  # [B, L, n, n]
        mixed = torch.einsum("blij, bljd -> blid", H_res, streams)

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
