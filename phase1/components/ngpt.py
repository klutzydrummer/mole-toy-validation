"""
nGPT: Normalized Transformer with Representation Learning on the Hypersphere.
Reference: arXiv:2410.01131

Spec: references/components/ngpt.md  (to be added)
Sources: references/sources/papers/ngpt_2410.01131.md  (to be added)

The core nGPT modifications applied here are:
  1. l2_norm — L2 normalization along last dim; used for hidden states and Q/K vectors
  2. Q and K L2-normalized per head after RoPE (dot products become cosine similarities)
  3. Attention scale inverted: s_qk replaces 1/√d_h; implemented as q * s_qk with scale=1.0
  4. normalize_ngpt_weights — post-optimizer-step column normalization of all weight matrices

The update rule (Eq. 10–11) and alpha parameters live in TransformerBlock._forward_ngpt.

Three attention classes mirror their non-nGPT counterparts:
  NGPTCausalAttention      mirrors CausalSelfAttention       (attention_rope_norms.py)
  NGPTMLACausalAttention   mirrors MLACausalAttention        (mla_attention.py)
  NGPTDiffCausalAttention  mirrors DifferentialCausalAttention (diff_attention.py)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from phase1.components._shared import apply_rope, precompute_rope


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    """L2-normalize along last dimension. Used for hidden states and Q/K vectors."""
    return F.normalize(x, dim=-1)


class NGPTCausalAttention(nn.Module):
    """
    nGPT causal self-attention (arXiv:2410.01131).

    Identical to CausalSelfAttention with two modifications:
      1. Q and K are L2-normalized per head after RoPE.
         Dot products become cosine similarities in [−1, 1].
      2. Attention logits scaled by learnable s_qk (init √d_h) rather than 1/√d_h.
         Implemented as: SDPA(q * s_qk, k, v, scale=1.0) = softmax(q @ k.T * s_qk) @ v.

    The inverted scale compensates for normalized Q/K: cosine similarities in [−1, 1]
    need to be multiplied by √d_h to recover the softmax variance of unnormalized vectors.
    s_qk is learnable — it starts at √d_h and adapts during training.
    """

    def __init__(self, d: int, n_heads: int, max_len: int = 4096):
        super().__init__()
        assert d % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d // n_heads

        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

        # s_qk: learnable attention scale. Init = √d_h (≈ 8.0 at d=512, n_heads=8).
        # Gradient flows through q * self.s_qk before the SDPA call.
        self.s_qk = nn.Parameter(torch.tensor(math.sqrt(d // n_heads)))

        cos, sin = precompute_rope(self.d_head, max_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)
        # L2-normalize Q and K per head — dot products become cosine similarities
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        # q * s_qk scales logits; scale=1.0 disables SDPA's default 1/√d_h
        out = F.scaled_dot_product_attention(q * self.s_qk, k, v, is_causal=True, scale=1.0)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out(out)


class NGPTMLACausalAttention(nn.Module):
    """
    nGPT Multi-Head Latent Attention (arXiv:2410.01131 + arXiv:2405.04434).

    Mirrors MLACausalAttention with nGPT modifications:
      - kv_norm and q_norm removed (input is already unit-norm in the nGPT forward path,
        so RMSNorm at the latent bottleneck becomes a redundant scalar multiplication)
      - Full concatenated q = [q_c; q_rope] and k = [k_c; k_rope] normalized per head
        after construction (normalizing the full head vector, not parts separately,
        preserves the relative content/positional contribution scale)
      - Inverted attention scale: s_qk init = √(d_h + d_h_R) for the full head dim

    All other MLA mechanics unchanged: shared KV latent, decoupled RoPE (Eq. 14–19),
    separate Q latent, d_c = d//2.
    """

    def __init__(self, d: int, n_heads: int,
                 d_c: int = None, d_c_q: int = None, max_len: int = 4096):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.d_h_R = self.d_head // 2
        self.d = d

        if d_c is None:
            d_c = d // 2
        if d_c_q is None:
            d_c_q = d // 2
        self.d_c, self.d_c_q = d_c, d_c_q

        # KV compression (Eq. 9–11) — no kv_norm; input is already unit-norm
        self.W_DKV = nn.Linear(d, d_c, bias=False)
        self.W_UK  = nn.Linear(d_c, n_heads * self.d_head, bias=False)
        self.W_UV  = nn.Linear(d_c, n_heads * self.d_head, bias=False)

        # Q compression (Eq. 12–13) — no q_norm; input is already unit-norm
        self.W_DQ  = nn.Linear(d, d_c_q, bias=False)
        self.W_UQ  = nn.Linear(d_c_q, n_heads * self.d_head, bias=False)

        # Decoupled RoPE projections (Eq. 14–15)
        self.W_QR  = nn.Linear(d_c_q, n_heads * self.d_h_R, bias=False)
        self.W_KR  = nn.Linear(d, self.d_h_R, bias=False)

        self.out = nn.Linear(d, d, bias=False)

        # s_qk: init √(d_h + d_h_R) — scaled for the full concatenated head dim
        full_head_dim = self.d_head + self.d_h_R
        self.s_qk = nn.Parameter(torch.tensor(math.sqrt(full_head_dim)))

        cos, sin = precompute_rope(self.d_h_R, max_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x):
        B, L, D = x.shape
        nh, dh, d_h_R = self.n_heads, self.d_head, self.d_h_R

        # KV path — no norm; input already on unit sphere
        c_kv = self.W_DKV(x)                                                    # [B, L, d_c]
        k_c  = self.W_UK(c_kv).reshape(B, L, nh, dh).transpose(1, 2)           # [B, nh, L, dh]
        v    = self.W_UV(c_kv).reshape(B, L, nh, dh).transpose(1, 2)           # [B, nh, L, dh]

        # Q path — no norm; input already on unit sphere
        c_q  = self.W_DQ(x)                                                     # [B, L, d_c_q]
        q_c  = self.W_UQ(c_q).reshape(B, L, nh, dh).transpose(1, 2)            # [B, nh, L, dh]

        # Decoupled RoPE (Eq. 14–15)
        q_rope = apply_rope(
            self.W_QR(c_q).reshape(B, L, nh, d_h_R).transpose(1, 2),
            self.rope_cos, self.rope_sin
        )  # [B, nh, L, d_h_R]
        k_rope = apply_rope(
            self.W_KR(x).reshape(B, L, 1, d_h_R).transpose(1, 2),
            self.rope_cos, self.rope_sin
        ).expand(B, nh, L, d_h_R)  # [B, nh, L, d_h_R]

        # Concatenate content and positional (Eq. 16–17)
        q = torch.cat([q_c, q_rope], dim=-1)   # [B, nh, L, dh + d_h_R]
        k = torch.cat([k_c, k_rope], dim=-1)   # [B, nh, L, dh + d_h_R]

        # L2-normalize full concatenated head vector per head
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        out = F.scaled_dot_product_attention(q * self.s_qk, k, v, is_causal=True, scale=1.0)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out(out)


class NGPTDiffCausalAttention(nn.Module):
    """
    nGPT Differential Causal Attention V2 (arXiv:2410.01131 + Jan 2026 V2).

    Mirrors DifferentialCausalAttention with nGPT modifications:
      - Q (shape [B, 2nh, L, dh]) and K (shape [B, nh, L, dh]) L2-normalized per head
        after RoPE. Each of the 2nh Q half-heads is independently normalized along dim=-1.
      - Inverted attention scale: s_qk init = √d_h, applied as q * s_qk with scale=1.0.
      - λ computation (sigmoid(W_lambda · x)) unchanged — it is a per-token scalar blend,
        not a projection that needs to be on the sphere.

    All V2 mechanics preserved: doubled Q, GQA pairing via repeat_interleave,
    sigmoid λ (not exp), no per-head RMSNorm, W_lambda bias=True.
    """

    def __init__(self, d: int, n_heads: int, layer_idx: int = 0, max_len: int = 4096):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.d = d

        self.W_Q = nn.Linear(d, 2 * d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        # W_lambda: bias=True so sigmoid(0) = 0.5 at init — reasonable starting λ
        self.W_lambda = nn.Linear(d, n_heads, bias=True)
        self.out = nn.Linear(d, d, bias=False)

        # s_qk: init √d_h (each Q half-head has dimension d_h)
        self.s_qk = nn.Parameter(torch.tensor(math.sqrt(d // n_heads)))

        cos, sin = precompute_rope(self.d_head, max_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x):
        B, L, D = x.shape
        nh, dh = self.n_heads, self.d_head

        q = self.W_Q(x).reshape(B, L, 2 * nh, dh).transpose(1, 2)   # [B, 2nh, L, dh]
        k = self.W_K(x).reshape(B, L, nh, dh).transpose(1, 2)        # [B, nh,  L, dh]
        v = self.W_V(x).reshape(B, L, nh, dh).transpose(1, 2)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # L2-normalize each half-head independently (F.normalize along dim=-1 = per-head)
        # Q: [B, 2nh, L, dh] — each of the 2nh half-heads independently unit-normed
        # K: [B, nh,  L, dh] — each of the nh heads independently unit-normed
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # GQA pairing: repeat each KV head twice to match doubled Q
        k_rep = k.repeat_interleave(2, dim=1)   # [B, 2nh, L, dh]
        v_rep = v.repeat_interleave(2, dim=1)

        attn = F.scaled_dot_product_attention(
            q * self.s_qk, k_rep, v_rep, is_causal=True, scale=1.0
        )  # [B, 2nh, L, dh]

        attn1 = attn[:, 0::2]   # [B, nh, L, dh]
        attn2 = attn[:, 1::2]

        # Token-specific λ (Diff V2): sigmoid projection, not exp reparameterization
        lam = torch.sigmoid(self.W_lambda(x)).transpose(1, 2).unsqueeze(-1)  # [B, nh, L, 1]

        out = (attn1 - lam * attn2).transpose(1, 2).reshape(B, L, D)
        return self.out(out)


@torch.no_grad()
def normalize_ngpt_weights(model: nn.Module) -> None:
    """
    Normalize all weight matrices row-wise after each optimizer step.

    Called post-step for nGPT configs only. Ensures weight row-vectors live on the unit
    hypersphere consistently with normalized hidden states (arXiv:2410.01131, Section 2.3).
    The paper normalizes rows of weight matrices: each row w_i of W ∈ ℝ^{n×d} is replaced
    by w_i / ||w_i||_2 (i.e., normalize along the last/embedding dimension, dim=-1).

    Covers: all nn.Linear weight matrices and nn.Embedding weight matrix.
    Skips tied parameters (embed/lm_head share the same Parameter object) via a seen-id
    set. Skips 1-D parameters (RMSNorm scales, biases) — only 2-D weight matrices.
    """
    seen = set()
    for m in model.modules():
        if isinstance(m, nn.Linear | nn.Embedding):
            w = m.weight
            if id(w) not in seen:
                seen.add(id(w))
                w.data.copy_(F.normalize(w.data, dim=-1))
