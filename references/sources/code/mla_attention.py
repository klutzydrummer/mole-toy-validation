# SOURCE: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
# PAPER:  "DeepSeek-V3 Technical Report" — arXiv:2412.19437 (December 2024)
#         MLA originally introduced in "DeepSeek-V2" — arXiv:2405.04434 (May 2024)
#
# This file contains the verbatim MLA (Multi-Head Latent Attention) class from the
# DeepSeek-V3 inference code, lightly adapted to remove tensor-parallel boilerplate
# (ColumnParallelLinear → nn.Linear, RowParallelLinear → nn.Linear) for readability.
# The second listing preserves the original source exactly as fetched.
#
# KEY CONCEPTS:
#   - Two inference modes:
#       "naive": expand latent → full K, V per head; cache full K/V tensors
#       default ("absorption"): cache only kv_norm(kv) latent + k_pe;
#                               absorb wkv_b into Q projection at runtime
#   - RoPE decoupling:
#       q split into q_nope (content, no RoPE) + q_pe (positional, RoPE applied)
#       kv_a output split into kv latent + k_pe (shared RoPE key, single head)
#       Only _pe components receive rotary embeddings
#   - KV absorption trick (default inference):
#       q_nope ← einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :qk_nope_head_dim])
#       This folds W^{UK} into the query so only the latent c_t^{KV} is stored,
#       not the expanded per-head keys.
#   - Cache layout (absorption mode):
#       kv_cache: (B, T, kv_lora_rank)      — content latent, position-independent
#       pe_cache: (B, T, qk_rope_head_dim)  — shared RoPE key, position-dependent
#
# DIMENSIONS (DeepSeek-V2 reference values):
#   dim              = 5120   (hidden dimension)
#   n_heads          = 128
#   qk_nope_head_dim = 128    (d_h, content component per head)
#   qk_rope_head_dim = 64     (d_h^R, positional component per head = d_h/2)
#   qk_head_dim      = 192    (d_h + d_h^R, full head dim for scale)
#   v_head_dim       = 128
#   kv_lora_rank     = 512    (d_c = 4*d_h, KV compression bottleneck)
#   q_lora_rank      = 1536   (d_c', query compression; 0 = no query compression)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to input tensor x."""
    # x: (B, S, H, D) or (B, S, 1, D)
    # freqs_cis: (S, D//2) complex
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(-2), 2)
    x_out = torch.stack([
        xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
        xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    return x_out.flatten(-2).type_as(x)


# ============================================================
# MLA — Multi-Head Latent Attention (simplified, no tensor parallel)
# Adapted from DeepSeek-V3 inference/model.py for standalone use.
# ============================================================

class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Layer.

    Implements the attention mechanism from DeepSeek-V2/V3 with low-rank KV compression
    and decoupled RoPE. Two inference modes:
      - naive: full K/V expansion and caching (simpler, higher memory)
      - absorption (default): cache only (kv_lora_rank + qk_rope_head_dim) per position
    """
    def __init__(self, args):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads  # simplified (no tensor parallel)
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        # Query projection: optionally low-rank (q_lora_rank > 0) or direct
        if self.q_lora_rank == 0:
            # Direct: h_t → q  (no query compression)
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            # Low-rank query: h_t → c_t^Q → q  (Eq. 12-13 from DeepSeek-V2)
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank)          # W^{DQ}
            self.q_norm = nn.RMSNorm(self.q_lora_rank)                 # RMSNorm on latent
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim)  # W^{UQ}

        # KV compression: h_t → [c_t^{KV} (kv_lora_rank) | k_t^R (qk_rope_head_dim)]
        # wkv_a implements W^{DKV} for the content latent + W^{KR} for the shared RoPE key
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank)                   # RMSNorm on KV latent

        # KV up-projection: c_t^{KV} → [k_nope (per head) | v (per head)]
        # Implements W^{UK} (first qk_nope_head_dim) and W^{UV} (last v_head_dim) jointly
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))

        # Output projection: W^O
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim)

        # Attention scale: 1/sqrt(d_h + d_h^R) = 1/sqrt(qk_head_dim)
        self.softmax_scale = self.qk_head_dim ** -0.5

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor], attn_impl: str = "absorption"):
        """
        Args:
            x:          (B, S, dim)
            start_pos:  int, position offset for KV cache indexing
            freqs_cis:  (S, qk_rope_head_dim//2) precomputed RoPE frequencies
            mask:       (S, T) causal mask or None
            attn_impl:  "naive" or "absorption"
        Returns:
            (B, S, dim)
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        # --- Query projection ---
        if self.q_lora_rank == 0:
            q = self.wq(x)                                    # (B, S, n_heads * qk_head_dim)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))         # low-rank path: Eq. 12-13
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)

        # Split into content (no RoPE) and positional (RoPE) components — Eq. 16
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)              # apply RoPE to positional part

        # --- KV compression ---
        # wkv_a produces: [c_t^{KV} (kv_lora_rank) | k_t^R (qk_rope_head_dim)]
        kv = self.wkv_a(x)                                    # (B, S, kv_lora_rank + qk_rope_head_dim)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # k_pe is the shared RoPE key (one vector per position, shared across all heads) — Eq. 15
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)  # (B, S, 1, qk_rope_head_dim)

        if attn_impl == "naive":
            # --- Naive mode: expand full K and V, cache them ---
            q = torch.cat([q_nope, q_pe], dim=-1)             # (B, S, n_heads, qk_head_dim)
            kv = self.wkv_b(self.kv_norm(kv))                 # expand latent → K+V
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            # Cache full K and V (expensive: n_heads * qk_head_dim per position)
            k_cache = k   # would store in self.k_cache[:bsz, start_pos:end_pos] in stateful version
            v_cache = v   # would store in self.v_cache[:bsz, start_pos:end_pos]
            scores = torch.einsum("bshd,bthd->bsht", q, k_cache) * self.softmax_scale

        else:
            # --- Absorption mode (default): cache only latent + k_pe ---
            # Absorb W^{UK} into Q: q_nope_absorbed = q_nope @ W^{UK}.T
            # This lets us compute scores directly against cached c_t^{KV}
            wkv_b = self.wkv_b.weight.view(self.n_local_heads, -1, self.kv_lora_rank)
            # q_nope_absorbed shape: (B, S, n_heads, kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])

            # Cache only the normalized latent and shared RoPE key
            kv_cache = self.kv_norm(kv)      # (B, S, kv_lora_rank)  — c_t^{KV}
            pe_cache = k_pe.squeeze(2)       # (B, S, qk_rope_head_dim) — k_t^R

            # Attention scores: content part + positional part — Eq. 18
            scores = (
                torch.einsum("bshc,btc->bsht", q_nope, kv_cache) +   # content scores
                torch.einsum("bshr,btr->bsht", q_pe,   pe_cache)     # positional scores
            ) * self.softmax_scale

        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)

        if attn_impl == "naive":
            x = torch.einsum("bsht,bthd->bshd", scores, v_cache)
        else:
            # Output in latent space, then fold W^{UV} (absorbed via wkv_b[-v_head_dim:])
            x = torch.einsum("bsht,btc->bshc", scores, kv_cache)                      # (B,S,H,kv_lora_rank)
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])       # (B,S,H,v_head_dim)

        x = self.wo(x.flatten(2))  # (B, S, dim)
        return x


# ============================================================
# VERBATIM SOURCE (from DeepSeek-V3 inference/model.py, tensor-parallel version)
# URL: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
# ============================================================
#
# class MLA(nn.Module):
#     """
#     Multi-Head Latent Attention (MLA) Layer.
#     """
#     def __init__(self, args: ModelArgs):
#         super().__init__()
#         self.dim = args.dim
#         self.n_heads = args.n_heads
#         self.n_local_heads = args.n_heads // world_size
#         self.q_lora_rank = args.q_lora_rank
#         self.kv_lora_rank = args.kv_lora_rank
#         self.qk_nope_head_dim = args.qk_nope_head_dim
#         self.qk_rope_head_dim = args.qk_rope_head_dim
#         self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
#         self.v_head_dim = args.v_head_dim
#
#         if self.q_lora_rank == 0:
#             self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
#         else:
#             self.wq_a = Linear(self.dim, self.q_lora_rank)
#             self.q_norm = RMSNorm(self.q_lora_rank)
#             self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
#         self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
#         self.kv_norm = RMSNorm(self.kv_lora_rank)
#         self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
#         self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
#         self.softmax_scale = self.qk_head_dim ** -0.5
#         if args.max_seq_len > args.original_seq_len:
#             mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
#             self.softmax_scale = self.softmax_scale * mscale * mscale
#
#         if attn_impl == "naive":
#             self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
#             self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
#         else:
#             self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
#             self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)
#
#     def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
#         bsz, seqlen, _ = x.size()
#         end_pos = start_pos + seqlen
#         if self.q_lora_rank == 0:
#             q = self.wq(x)
#         else:
#             q = self.wq_b(self.q_norm(self.wq_a(x)))
#         q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
#         q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
#         q_pe = apply_rotary_emb(q_pe, freqs_cis)
#         kv = self.wkv_a(x)
#         kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
#         k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
#         if attn_impl == "naive":
#             q = torch.cat([q_nope, q_pe], dim=-1)
#             kv = self.wkv_b(self.kv_norm(kv))
#             kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
#             k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
#             k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
#             self.k_cache[:bsz, start_pos:end_pos] = k
#             self.v_cache[:bsz, start_pos:end_pos] = v
#             scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
#         else:
#             wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size)
#             wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
#             q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
#             self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
#             self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
#             scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
#                       torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
#         if mask is not None:
#             scores += mask.unsqueeze(1)
#         scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
#         if attn_impl == "naive":
#             x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
#         else:
#             x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
#             x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
#         x = self.wo(x.flatten(2))
#         return x
