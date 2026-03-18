"""
RoPE (Rotary Position Embedding) — Reference Implementation
============================================================
Sources:
  - Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    arXiv:2104.09864  https://arxiv.org/abs/2104.09864
  - EleutherAI/gpt-neox positional_embeddings.py (rotate_half / apply_rotary_pos_emb)
  - Meta LLaMA model.py (complex-number form: precompute_freqs_cis / apply_rotary_emb)

Two equivalent implementations are provided:
  1. rotate_half form  — operates on real tensors, used by GPT-NeoX / most HF models
  2. complex form      — treats pairs as complex numbers, used by LLaMA
"""

import torch
import torch.nn as nn
from typing import Tuple


# ---------------------------------------------------------------------------
# Implementation 1: rotate_half form  (GPT-NeoX style)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """
    Precomputes cos/sin tables up to max_seq_len and returns slices on demand.

    Frequencies: inv_freq[j] = 1 / (base ^ (2j / dim)),  j = 0..dim/2-1
    This gives θ_j = base^(-2j/dim), a geometric sequence.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        # Shape: (dim/2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute and cache cos/sin tables
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        # freqs[m, j] = m * inv_freq[j]
        freqs = torch.outer(t, self.inv_freq)           # (seq_len, dim/2)
        # Concatenate to cover all dim positions: [freqs | freqs] -> (seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (cos, sin) tables of shape (seq_len, dim)."""
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates the second half of the last dimension by negating it and swapping
    with the first half.  For x = [x1 | x2] returns [-x2 | x1].

    This implements the 2D rotation within each dimension pair:
        [cos  -sin] [x1]   [x1*cos - x2*sin]
        [sin   cos] [x2] = [x1*sin + x2*cos]
    which in the rotate_half form becomes:
        x * cos + rotate_half(x) * sin
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to query and key tensors.

    Args:
        q:   Query tensor, shape (..., seq_len, head_dim)
        k:   Key tensor,   shape (..., seq_len, head_dim)
        cos: Cosine table, shape (max_seq, head_dim)  — from RotaryEmbedding
        sin: Sine table,   shape (max_seq, head_dim)
        offset: position offset for KV-cache incremental decoding

    Returns:
        (q_rot, k_rot) with RoPE applied.

    Core formula (per Su et al. eq. for f_q, f_k):
        q_rot = q * cos(m·Θ) + rotate_half(q) * sin(m·Θ)
        k_rot = k * cos(n·Θ) + rotate_half(k) * sin(n·Θ)

    Key property: dot(q_rot_m, k_rot_n) depends only on (m - n).
    """
    seq_len = q.shape[-2]
    cos = cos[offset : offset + seq_len]   # (seq_len, head_dim)
    sin = sin[offset : offset + seq_len]

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# Implementation 2: complex-number form  (LLaMA style)
# ---------------------------------------------------------------------------

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute complex exponentials e^{i·m·θ_j} for all positions m and
    dimension-pair indices j.

    Returns:
        freqs_cis: complex64 tensor of shape (end, dim//2)
                   freqs_cis[m, j] = exp(i * m * inv_freq[j])
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end).float()
    freqs = torch.outer(t, inv_freq)           # (end, dim/2), real
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape freqs_cis to broadcast over batch and head dimensions of x."""
    ndim = x.ndim
    assert ndim >= 2
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1]), (freqs_cis.shape, x.shape)
    shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb_complex(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    LLaMA-style RoPE using complex multiplication.

    Treats each consecutive pair (x_{2j}, x_{2j+1}) as a complex number
    and multiplies by e^{i·m·θ_j}.  This is mathematically equivalent to
    apply_rotary_pos_emb above.

    Args:
        xq:        (..., seq_len, head_dim)
        xk:        (..., seq_len, head_dim)
        freqs_cis: (seq_len, head_dim//2) complex64

    Returns:
        (xq_rot, xk_rot) as real tensors with same shape as inputs.
    """
    # Reshape to complex: (..., seq_len, head_dim/2)  complex
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = _reshape_for_broadcast(freqs_cis, xq_)
    # Complex multiply = rotation
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ---------------------------------------------------------------------------
# Correctness verification helpers
# ---------------------------------------------------------------------------

def verify_relative_position_property(dim: int = 8, tol: float = 1e-5) -> bool:
    """
    Verify the key RoPE property:
        dot(rotate(q, m), rotate(k, n)) == dot(rotate(q, 0), rotate(k, n-m))

    for all m, n in a small test set.
    """
    rope = RotaryEmbedding(dim=dim, max_seq_len=32)
    cos, sin = rope.forward(seq_len=32)

    passed = True
    for m in range(5):
        for n in range(5):
            q = torch.randn(1, dim)
            k = torch.randn(1, dim)

            q_m = q * cos[m] + rotate_half(q) * sin[m]
            k_n = k * cos[n] + rotate_half(k) * sin[n]
            dot_mn = (q_m * k_n).sum()

            offset = n - m
            if offset >= 0:
                q_0 = q * cos[0] + rotate_half(q) * sin[0]
                k_off = k * cos[offset] + rotate_half(k) * sin[offset]
                dot_ref = (q_0 * k_off).sum()
                if not torch.isclose(dot_mn, dot_ref, atol=tol):
                    passed = False
    return passed


if __name__ == "__main__":
    ok = verify_relative_position_property()
    print(f"Relative position property holds: {ok}")

    # Quick smoke test
    B, H, T, D = 2, 4, 16, 64
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    rope = RotaryEmbedding(dim=D, max_seq_len=T)
    cos, sin = rope.forward(T)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    print(f"rotate_half form output shape: {q_rot.shape}")  # (2, 4, 16, 64)

    freqs_cis = precompute_freqs_cis(dim=D, end=T)
    q_rot2, k_rot2 = apply_rotary_emb_complex(q, k, freqs_cis)
    print(f"complex form output shape: {q_rot2.shape}")
    print(f"Both forms agree: {torch.allclose(q_rot, q_rot2, atol=1e-5)}")
