"""
Shared primitives used by all Phase 1 components.

These are foundational building blocks with no inter-component dependencies.
Do not add math that belongs to a specific component spec here.
"""

import math  # noqa: F401 — re-exported for components that need it

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope(d_head: int, max_len: int = 4096, theta: float = 10000.0):
    pos = torch.arange(max_len, dtype=torch.float32)
    freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
    angles = pos.unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    """x: [B, n_heads, L, d_head]"""
    d_half = x.shape[-1] // 2
    cos = cos[:x.shape[2], :d_half].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[2], :d_half].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., :d_half], x[..., d_half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, d: int, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            d_ff = int(d * 8 / 3)
            d_ff = ((d_ff + 63) // 64) * 64
        self.d_ff = d_ff
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
