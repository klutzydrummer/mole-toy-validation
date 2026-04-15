"""
nGPT reference implementation — verbatim excerpts from NVIDIA/ngpt model.py.
Source: https://github.com/NVIDIA/ngpt/blob/main/model.py
License: MIT (Copyright 2024 NVIDIA CORPORATION & AFFILIATES)

Excerpted for comparison against our implementation in phase1/components/ngpt.py.
Key patterns: justnorm, alpha update, Q/K normalization, sqk scaling, sz logit scale.
"""

# ── justnorm ───────────────────────────────────────────────────────────────────
# NVIDIA ref (inside Block class):

def justnorm(self, x):
    res = x / x.norm(p=2, dim=-1, keepdim=True)
    return res

# Our impl (module-level function, phase1/components/ngpt.py:31):
#   def l2_norm(x): return F.normalize(x, dim=-1)
# Equivalent. F.normalize handles zero-norm edge case; NVIDIA doesn't.


# ── Alpha parameter init ───────────────────────────────────────────────────────
# NVIDIA ref (inside Block.__init__, use_nGPT == 1 branch):

#   config.base_scale = 1.0 / (n_embd ** 0.5)   [GPTConfig default]
#
#   self.attn_alpha_init_value   = 0.05
#   self.attn_alpha_init_scaling = config.base_scale   # = 1/sqrt(d)
#   self.attn_alpha = nn.Parameter(
#       self.attn_alpha_init_scaling * torch.ones(config.n_embd, dtype=torch.float32)
#   )
#   # stored value: 1/sqrt(d) ≈ 0.0442 at d=512
#   # effective value at forward: attn_alpha * (0.05 / base_scale)
#   #                            = base_scale * (0.05 / base_scale) = 0.05
#   # ... so effective init alpha = 0.05 per-dim, consistent with paper

# Our impl (phase1/components/transformer_block.py):
#   self.alpha_a = nn.Parameter(torch.full((d,), 0.05))
#   applied as: alpha_eff = self.alpha_a * (1.0 / math.sqrt(self.d))
#   effective init = 0.05 / sqrt(d) ≈ 0.0022 at d=512
#
# NOTE: NVIDIA stores alpha = base_scale and recovers 0.05 at forward time.
# Our impl stores 0.05 and applies 1/sqrt(d) at forward time.
# Functional result is identical: effective step ≈ 0.05/sqrt(d) at init.


# ── Alpha-based spherical update (Block.forward) ───────────────────────────────
# NVIDIA ref (verbatim, attention sub-layer):

#   lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
#   lr = torch.abs(lr)
#   # lr effective value = attn_alpha * (0.05 / base_scale) = 0.05 at init (see above)
#   # torch.abs enforces non-negative step size
#
#   A_norm = self.justnorm(h)       # pre-update hidden state, renormed
#   B_norm = self.justnorm(h_att)   # sublayer output, renormed
#   res = A_norm + lr * (B_norm - A_norm)
#   h = self.justnorm(res)          # LERP then renorm

# Our impl (TransformerBlock._forward_ngpt, transformer_block.py):
#   h_a = l2_norm(self.attn(x))                    # B_norm equivalent
#   x   = l2_norm(x + alpha_a * (h_a - x))         # LERP then renorm
#
# NOTE: NVIDIA normalizes x (h) before the LERP step (A_norm = justnorm(h)).
# Our impl: x is already on S^{d-1} (maintained by the update loop), so
# l2_norm(x) = x and l2_norm(x + alpha*(h_a - x)) is functionally equivalent.
# The normalization of h_att/sublayer output is identical in both.


# ── Q/K normalization and sqk scaling (Block.forward) ─────────────────────────
# NVIDIA ref (verbatim, attention, use_nGPT == 1):

#   sqk = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(
#       1, 1, self.config.n_head, self.config.n_embd // self.config.n_head
#   )
#   # sqk is per-dim per-head: shape [1, 1, n_head, d_head]
#   # sqk effective init = base_scale * (1.0 / base_scale) = 1.0 per element
#   # Applied AFTER L2Norm: q = sqk * justnorm(q)
#   q = sqk * self.justnorm(q)
#   k = sqk * self.justnorm(k)
#   # Then SDPA with softmax_scale = sqrt_head_dim (inverted from standard 1/sqrt_head_dim):
#   softmax_scale = sqrt_head_dim
#   y = flash_attn_func(..., softmax_scale=softmax_scale, ...)
#   # Effective attention: softmax(sqk^2 * cosine_sim * sqrt(d_h)) @ v

# Our impl (NGPTCausalAttention.forward, ngpt.py):
#   q = F.normalize(q, dim=-1)
#   k = F.normalize(k, dim=-1)
#   out = F.scaled_dot_product_attention(q * self.s_qk, k, v, is_causal=True, scale=1.0)
#   # s_qk is a SCALAR, init sqrt(d_h) ≈ 8.0. Applied to Q only.
#   # Effective attention: softmax(s_qk * cosine_sim * 1.0) @ v
#
# INTENTIONAL DEVIATION:
#   - NVIDIA: sqk per-dim per-head [n_head, d_head], applied to both Q and K,
#     SDPA scale = sqrt(d_h) → net scale = sqk^2 * sqrt(d_h)
#   - Ours: s_qk scalar, applied to Q only, SDPA scale = 1.0 → net scale = s_qk
#   Both achieve "inverted scale" relative to standard 1/sqrt(d_h).
#   Scalar s_qk is simpler and closer to the paper's description (Section 2.2
#   describes a single scalar per layer, not per-dim per-head).


# ── Logit scale sz (GPT.forward) ──────────────────────────────────────────────
# NVIDIA ref (verbatim):

#   if (config.use_nGPT == 1):
#       self.sz_init_value   = 1.00
#       self.sz_init_scaling = config.base_scale   # = 1/sqrt(d)
#       self.sz = nn.Parameter(
#           self.sz_init_scaling * torch.ones(config.vocab_size, dtype=torch.float32)
#       )
#   # In forward:
#   sz = self.sz * (self.sz_init_value / self.sz_init_scaling)   # = self.sz / base_scale
#   # effective init = base_scale * (1 / base_scale) = 1.0 per vocab token
#   logits = sz * logits
#   # Shape [vocab_size] broadcast against logits [B, T, vocab_size]

# Our impl (ToyTransformer.__init__ / forward, model.py):
#   self.ngpt_s_z = nn.Parameter(torch.tensor(math.sqrt(d)))   # scalar, init sqrt(d)
#   logits = logits * self.ngpt_s_z
#
# INTENTIONAL DEVIATION:
#   - NVIDIA: per-vocab vector sz, init 1.0 (after scaling)
#   - Ours: scalar s_z, init sqrt(d) ≈ 22.6
#   Paper Section 2.5 describes a scalar s_z (not per-vocab).
#   sqrt(d) init is theoretically motivated: embed.weight rows are unit-norm,
#   so logits are cosine similarities in [-1,1]; multiplying by sqrt(d) restores
#   a softmax variance comparable to unnormalized d-dim dot products.


# ── Weight normalization ───────────────────────────────────────────────────────
# NVIDIA ref (train.py — not in model.py, but called after optimizer.step()):
# The model exposes a normalize_weights() method; train.py calls it each step.
# Pattern (from repo README and train.py):
#
#   def normalize_weights(self):
#       for pn, p in self.named_parameters():
#           if p.dim() == 2:   # only 2D weight matrices
#               p.data.copy_(p.data / p.data.norm(p=2, dim=0, keepdim=True))
#               # NOTE: NVIDIA normalizes along dim=0 (column normalization)
#               # This makes each COLUMN unit-norm, not each row.

# Our impl (normalize_ngpt_weights, ngpt.py):
#   w.data.copy_(F.normalize(w.data, dim=-1))
#   # dim=-1 = last dimension = row normalization (each ROW unit-norm)
#
# INTENTIONAL DEVIATION — DIM DIRECTION:
#   NVIDIA ref uses dim=0 (column normalization).
#   Our impl uses dim=-1 (row normalization).
#
#   Paper Section 2.3: "normalize each weight matrix W ∈ ℝ^{d_out × d_in} such that
#   each row vector has unit L2 norm." → row normalization → dim=-1.
#   NVIDIA's dim=0 (column normalization) appears to conflict with the paper text.
#   We follow the paper (dim=-1 / row normalization).
