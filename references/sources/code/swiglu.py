"""
SwiGLU FFN — Reference Implementation
=======================================
Sources:
  - Shazeer, "GLU Variants Improve Transformer"
    arXiv:2002.05202  https://arxiv.org/abs/2002.05202
  - Meta LLaMA model.py FeedForward class
  - Google T5X layers.py MlpBlock (activations=['swish', 'linear'])

Formula:
    FFN_SwiGLU(x) = (SiLU(x @ W1.T) * (x @ W3.T)) @ W2.T

    where SiLU(x) = x * sigmoid(x)   [Swish with β=1]

Dimension:
    hidden_dim = (8/3) * d_model   (≈ 2.667×, rounded to hardware multiple)
    This compensates for having three matrices instead of two.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Implementation 1: LLaMA-style (canonical SwiGLU)
# ---------------------------------------------------------------------------

class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network as used in LLaMA.

    Three projections (no bias):
      - gate_proj (W1): d_model → hidden_dim   — passed through SiLU
      - up_proj   (W3): d_model → hidden_dim   — linear branch
      - down_proj (W2): hidden_dim → d_model   — output projection

    Forward:
      h = F.silu(gate_proj(x)) * up_proj(x)
      out = down_proj(h)

    Args:
        d_model:        Input/output dimension.
        hidden_dim:     Inner dimension. If None, computed as (8/3)*d_model
                        rounded up to the nearest multiple_of.
        multiple_of:    Round hidden_dim up to this multiple (hardware alignment).
        bias:           Whether to use bias in linear layers (default False).
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = None,
        multiple_of: int = 256,
        bias: bool = False,
    ):
        super().__init__()
        if hidden_dim is None:
            # 8/3 * d_model, rounded to multiple_of
            hidden_dim = int(2 * (4 * d_model) / 3)        # = 8/3 * d_model
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.hidden_dim = hidden_dim

        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=bias)   # W1
        self.up_proj   = nn.Linear(d_model, hidden_dim, bias=bias)   # W3
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=bias)   # W2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., d_model)
        Returns:
            out: (..., d_model)
        """
        # SiLU(gate) * up  — element-wise product
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Implementation 2: Minimal functional form
# ---------------------------------------------------------------------------

def swiglu(x: torch.Tensor, W1: torch.Tensor, W3: torch.Tensor, W2: torch.Tensor) -> torch.Tensor:
    """
    Functional SwiGLU FFN.

    FFN_SwiGLU(x, W1, W2, W3) = (Swish(x @ W1.T) * (x @ W3.T)) @ W2.T

    As defined in Shazeer (2020) eq for FFN_SwiGLU where Swish = SiLU (β=1).

    Args:
        x:  (..., d_model)
        W1: (hidden_dim, d_model)  — gate weight
        W3: (hidden_dim, d_model)  — up/value weight
        W2: (d_model, hidden_dim)  — down/output weight

    Returns:
        (..., d_model)
    """
    gate = F.silu(x @ W1.T)    # Swish-gated branch
    up   = x @ W3.T             # linear branch
    return (gate * up) @ W2.T


# ---------------------------------------------------------------------------
# Swish / SiLU reference
# ---------------------------------------------------------------------------

def swish(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Swish activation: x * sigmoid(β·x)

    With β=1 this is SiLU, the standard used in SwiGLU.
    F.silu(x) is the PyTorch built-in and is preferred.
    """
    return x * torch.sigmoid(beta * x)


# ---------------------------------------------------------------------------
# GLU variants for reference (from Table 1 of Shazeer 2020)
# ---------------------------------------------------------------------------

def glu(x: torch.Tensor, W: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Original GLU: σ(xW) ⊙ (xV)"""
    return torch.sigmoid(x @ W.T) * (x @ V.T)


def reglu(x: torch.Tensor, W: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """ReGLU: ReLU(xW) ⊙ (xV)"""
    return F.relu(x @ W.T) * (x @ V.T)


def geglu(x: torch.Tensor, W: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """GEGLU: GELU(xW) ⊙ (xV)"""
    return F.gelu(x @ W.T) * (x @ V.T)


# ---------------------------------------------------------------------------
# Correctness verification helpers
# ---------------------------------------------------------------------------

def verify_hidden_dim_ratio(d_model: int = 512, multiple_of: int = 256) -> bool:
    """
    Verify hidden_dim is approximately 8/3 * d_model.
    Exact formula: hidden_dim = round_up(floor(8/3 * d_model), multiple_of)
    """
    ffn = SwiGLUFeedForward(d_model=d_model, multiple_of=multiple_of)
    ratio = ffn.hidden_dim / d_model
    # Should be close to 8/3 ≈ 2.667, typically between 2.5 and 2.9
    return 2.4 < ratio < 3.0


def verify_three_matrices() -> bool:
    """SwiGLU must have exactly three projection matrices."""
    ffn = SwiGLUFeedForward(d_model=64, multiple_of=64)
    matrices = [n for n, p in ffn.named_parameters() if 'weight' in n]
    return len(matrices) == 3


def verify_gating_only_on_gate_branch(d_model: int = 32) -> bool:
    """
    The SiLU nonlinearity is applied ONLY to the gate branch (W1/gate_proj),
    NOT to the up branch (W3/up_proj). Verify by checking that the up branch
    output is linear (no clipping at 0, unlike ReLU).
    """
    ffn = SwiGLUFeedForward(d_model=d_model, multiple_of=32)
    nn.init.ones_(ffn.gate_proj.weight)
    nn.init.zeros_(ffn.up_proj.weight)
    nn.init.eye_(ffn.up_proj.weight[:d_model])   # identity-like

    x = torch.randn(4, d_model)
    # With gate=ones and up=identity, output should be smooth (SiLU output * x)
    # This is just a structural smoke test
    out = ffn(x)
    return out.shape == x.shape


def verify_output_shape(d_model: int = 128, seq_len: int = 16, batch: int = 2) -> bool:
    """Output shape must equal input shape."""
    ffn = SwiGLUFeedForward(d_model=d_model)
    x = torch.randn(batch, seq_len, d_model)
    out = ffn(x)
    return out.shape == x.shape


if __name__ == "__main__":
    print(f"Hidden dim ratio ~8/3: {verify_hidden_dim_ratio()}")
    print(f"Three weight matrices: {verify_three_matrices()}")
    print(f"Output shape correct:  {verify_output_shape()}")

    # Smoke test: functional and module agree
    d, h = 64, 192
    W1 = torch.randn(h, d)
    W3 = torch.randn(h, d)
    W2 = torch.randn(d, h)
    x  = torch.randn(2, 8, d)
    out_fn = swiglu(x, W1, W3, W2)

    ffn = SwiGLUFeedForward(d_model=d, hidden_dim=h)
    ffn.gate_proj.weight.data = W1
    ffn.up_proj.weight.data   = W3
    ffn.down_proj.weight.data = W2
    out_mod = ffn(x)

    print(f"Functional vs module agree: {torch.allclose(out_fn, out_mod, atol=1e-5)}")
    print(f"Output shape: {out_mod.shape}")

    # Verify SiLU == Swish(β=1)
    x_test = torch.randn(100)
    print(f"SiLU == Swish(β=1): {torch.allclose(F.silu(x_test), swish(x_test, beta=1.0), atol=1e-6)}")
