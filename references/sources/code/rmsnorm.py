"""
RMSNorm — Reference Implementation
====================================
Sources:
  - Zhang & Sennrich, "Root Mean Square Layer Normalization"
    arXiv:1910.07467  https://arxiv.org/abs/1910.07467
  - bzhangGo/rmsnorm rmsnorm_torch.py  (original authors' repo)
  - Meta LLaMA model.py (production variant)

Formula:
    RMSNorm(x) = g · x / RMS(x)
    RMS(x)     = sqrt( mean(x², dim=-1, keepdim=True) + ε )

Key difference from LayerNorm: NO mean subtraction (no re-centering).
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Implementation 1: Original authors' version  (bzhangGo/rmsnorm)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Reference: Zhang & Sennrich, arXiv:1910.07467

    Args:
        d:     Model/feature dimension.
        p:     Partial RMSNorm fraction in [0, 1].  Set to -1 (default) to
               use all dimensions (standard RMSNorm).
        eps:   Small constant added inside the sqrt for numerical stability.
        bias:  If True, adds a learnable bias term.  Typically False because
               RMSNorm does not enforce re-centering invariance.
    """

    def __init__(self, d: int, p: float = -1.0, eps: float = 1e-8, bias: bool = False):
        super().__init__()
        self.d = d
        self.p = p
        self.eps = eps
        self.bias = bias

        # Learnable scale parameter g, init to 1
        self.scale = nn.Parameter(torch.ones(d))

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p < 0.0 or self.p > 1.0:
            # Standard RMSNorm: use full vector
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            # Partial RMSNorm (pRMSNorm): estimate RMS from first p fraction
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)
            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        # RMS(x) = norm(x) / sqrt(d)
        # Equivalently: sqrt(mean(x^2))
        rms_x = norm_x * (d_x ** (-0.5))
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset
        return self.scale * x_normed


# ---------------------------------------------------------------------------
# Implementation 2: LLaMA-style (rsqrt form, float32 cast)
# ---------------------------------------------------------------------------

class RMSNormLlama(nn.Module):
    """
    LLaMA-style RMSNorm using torch.rsqrt.

    Uses rsqrt for a single fused operation and casts to float32 for the
    normalization step to avoid precision loss, then casts back.

    This is the de facto standard in modern LLM codebases.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # x.pow(2).mean(-1, keepdim=True) + eps  ==  mean(x^2) + eps
        # rsqrt(v) = 1 / sqrt(v)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for stability, then cast back to original dtype
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# ---------------------------------------------------------------------------
# Correctness verification helpers
# ---------------------------------------------------------------------------

def verify_no_mean_centering(dim: int = 64, tol: float = 1e-5) -> bool:
    """
    Verify that RMSNorm does NOT subtract the mean (unlike LayerNorm).
    A vector shifted by a constant should NOT normalize to the same output.
    """
    norm = RMSNormLlama(dim=dim)
    norm.eval()
    with torch.no_grad():
        x = torch.randn(4, dim)
        shift = torch.ones(4, dim) * 5.0   # large constant shift
        x_shifted = x + shift

        out_x = norm(x)
        out_shifted = norm(x_shifted)

        # LayerNorm would give identical outputs; RMSNorm should differ
        are_equal = torch.allclose(out_x, out_shifted, atol=tol)
        return not are_equal   # True = correctly different (no mean centering)


def verify_formula(dim: int = 32, tol: float = 1e-5) -> bool:
    """
    Verify output matches the formula: g * x / sqrt(mean(x^2) + eps).
    """
    norm = RMSNormLlama(dim=dim)
    norm.eval()
    x = torch.randn(2, 4, dim)
    with torch.no_grad():
        out = norm(x)
        expected_rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps)
        expected = norm.weight * (x / expected_rms)
        return torch.allclose(out.float(), expected, atol=tol)


def verify_scale_invariance(dim: int = 32, tol: float = 1e-4) -> bool:
    """
    RMSNorm is scale-equivariant: RMSNorm(c*x) / c == RMSNorm(x) / 1
    (because the RMS scales with c, so the ratio x/RMS(x) is invariant to scale).
    The output satisfies: RMSNorm(c*x) = c * RMSNorm(x) only if g=1.
    Actually the key property is: the normalized part x/RMS(x) is scale-invariant.
    """
    norm = RMSNormLlama(dim=dim)
    # Set weight to 1 to isolate normalization
    nn.init.ones_(norm.weight)
    norm.eval()
    x = torch.randn(4, dim)
    c = 3.7
    with torch.no_grad():
        out_x  = norm(x)
        out_cx = norm(c * x)
        # x/RMS(x) == (cx)/RMS(cx)
        return torch.allclose(out_x, out_cx, atol=tol)


if __name__ == "__main__":
    print(f"No mean centering (should be True):  {verify_no_mean_centering()}")
    print(f"Formula correct   (should be True):  {verify_formula()}")
    print(f"Scale invariance  (should be True):  {verify_scale_invariance()}")

    # Smoke test: both implementations should agree
    dim = 64
    x = torch.randn(2, 16, dim)
    n1 = RMSNorm(d=dim, eps=1e-6)
    n2 = RMSNormLlama(dim=dim, eps=1e-6)
    nn.init.ones_(n1.scale)
    nn.init.ones_(n2.weight)
    with torch.no_grad():
        out1 = n1(x)
        out2 = n2(x)
    print(f"Both impls agree: {torch.allclose(out1.float(), out2.float(), atol=1e-5)}")
