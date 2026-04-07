# Phase 3 Interface Spec: OuterModel ↔ ToyTransformer

**Status:** DRAFT — prerequisite for any Phase 3 code (see COMPOSITION_GATE.md §7)
**Date:** 2026-04-06

This document defines the tensor contract between the Phase 2 outer encoder (`OuterModel`)
and the Phase 1 inner transformer (`ToyTransformer`) for Phase 3 composition. No Phase 3
code should be written until this spec is reviewed and the decisions below are confirmed.

---

## 1. Data flow

```
x [B, L]  (token ids)
  │
  ▼
OuterModel.embed(x) → h [B, L, d]
  │
  ▼
OuterModel.encoder(h) → encoder_out [B, L, d]      ← Zone E
  │
  ▼
OuterModel.router(encoder_out) → concept_tokens [B, M_max, d]   ← BoundaryRouter
                                  boundary_idx   [B, M_max]     (0 = padding)
                                  concept_mask   [B, M_max]     (True = valid slot)
                                  boundary_probs [B, L]
                                  compression_ratio (scalar)
  │
  ▼
ToyTransformer(concept_tokens, key_padding_mask=~concept_mask)   ← NEW in Phase 3
  → processed_concepts [B, M_max, d]
  │
  ▼
SimpleDecoder(processed_concepts, boundary_idx, concept_mask, encoder_out)
  → logits [B, L, vocab]                                         ← Zone D
```

---

## 2. Inner transformer input contract

**Tensor:** `concept_tokens [B, M_max, d]`
- Valid slots (where `concept_mask[b, m] = True`): contain the concept token vector from `BoundaryRouter`
- Padding slots (where `concept_mask[b, m] = False`): contain **zeros** (current behavior in `BoundaryRouter`)
- `M_max` varies per batch — it is the maximum boundary count across all sequences in the batch

**Key padding mask:** `~concept_mask [B, M_max]` (True = slot is padding, False = valid)
- Must be passed to all attention modules in `ToyTransformer` to prevent attention over zero-padded slots
- Zero-padded slots corrupt inner transformer output near boundaries for shorter sequences if not masked

**Decision (required):** Does the inner transformer use **causal** or **bidirectional** attention over concept tokens?
- **Recommended: causal** — consistent with the autoregressive H-Net design; concept token at position m should not attend to future concept tokens m+1, m+2, ...
- **Alternative: bidirectional** — concept tokens within a document could benefit from full context; but this changes the generative model's factorization
- **Status: UNDECIDED** — must be decided before Phase 3 code is written

---

## 3. Inner transformer output contract

**Tensor:** `processed_concepts [B, M_max, d]`
- Same shape as input: `[B, M_max, d]`
- Padding slot outputs should be ignored downstream (SimpleDecoder uses `concept_mask` to select valid slots)
- `ToyTransformer` must not be weight-tied to the outer embedding or lm_head — it operates in concept-token space, not token-id space

---

## 4. Required code changes to attention modules

All three Phase 1 attention modules must accept an optional `key_padding_mask` parameter.
Currently they only use `is_causal=True` and have no padding mask path.

### 4a. CausalSelfAttention (`phase1/components/attention_rope_norms.py`)

Current signature:
```python
def forward(self, x):
    B, L, D = x.shape
    ...
    F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

Required change:
```python
def forward(self, x, key_padding_mask=None):
    # key_padding_mask: [B, L] bool, True = ignore this key position
    # Combine with causal mask via attn_mask parameter
```

`F.scaled_dot_product_attention` accepts `attn_mask [B, n_heads, L_q, L_k]` (additive float
mask, -inf to block) or a bool mask. The causal mask and key_padding_mask must be combined:
```python
# causal: upper triangle = -inf
# padding: columns where key_padding_mask=True = -inf
# combined = causal_mask + padding_mask (broadcast over heads)
```

### 4b. MLACausalAttention (`phase1/components/mla_attention.py`)
Same change — add `key_padding_mask=None` parameter. MLA's KV compression operates on
the full sequence before splitting to heads; the mask applies at the attention score stage,
same as standard attention.

### 4c. DifferentialCausalAttention / DiffMLAAttention (`phase1/components/diff_attention.py`)
Same change. The differential mechanism subtracts two attention maps — the mask must be
applied to **both** maps before subtraction, not to the difference.

### 4d. ToyTransformer.forward signature change
```python
def forward(self, x, key_padding_mask=None):
    # Thread key_padding_mask through all TransformerBlock.forward calls
```

`TransformerBlock.forward` must similarly thread the mask through to `self.attn.forward`.

---

## 5. SimpleDecoder integration

`SimpleDecoder` currently receives concept tokens via the router output. In Phase 3 it
must receive **processed** concept tokens (post-inner-transformer) instead of raw router
output. The plug-back index (`boundary_idx`) and `concept_mask` are unchanged — they come
from `BoundaryRouter` and are passed through to `SimpleDecoder` unmodified.

**No change required to SimpleDecoder itself** — only the tensor passed as `concept_tokens`
changes from raw boundary tokens to inner-transformer-processed tokens.

---

## 6. Phase 3 model wrapper sketch

```python
class Phase3Model(nn.Module):
    def __init__(self, outer_config, inner_config, ...):
        self.outer = OuterModel(config=outer_config, ...)  # best encoder from Phase 2
        self.inner = ToyTransformer(config=inner_config, ...)  # composed Phase 1 model
        # Note: lm_head lives in outer (token space); inner operates in concept space
        # Remove inner.lm_head and inner.norm_out from the inner model's forward path,
        # or override forward to stop before lm_head

    def forward(self, x):
        h = self.outer.embed(x)
        encoder_out = self.outer.encoder(h)
        concept_tokens, boundary_idx, concept_mask, boundary_probs, cr = \
            self.outer.router(encoder_out)

        # Inner transformer on concept tokens (causal, with padding mask)
        processed = self.inner(concept_tokens, key_padding_mask=~concept_mask)
        # processed: [B, M_max, d] — inner transformer output, no lm_head applied

        logits = self.outer.decoder(processed, boundary_idx, concept_mask, encoder_out)
        return logits, boundary_probs, cr
```

---

## 7. Open decisions (must resolve before Phase 3 code)

| Decision | Options | Recommendation | Status |
|----------|---------|---------------|--------|
| Causal vs. bidirectional inner attention | causal / bidirectional | causal (matches H-Net) | UNDECIDED |
| lm_head placement | outer only / inner only / both | outer only | DECIDED |
| RoPE positions for concept tokens | use boundary_idx / use 0..M / no RoPE | use 0..M (ordinal position among concepts) | UNDECIDED |
| Inner transformer depth | same as Phase 1 (8L) / shallower | 8L (same) | UNDECIDED |
| Weight sharing outer↔inner | yes / no | no (different representation spaces) | DECIDED |

**RoPE position note:** Using `boundary_idx` positions for concept tokens would give each
concept token its source position in the original sequence — this could help the inner
transformer model inter-concept positional relationships accurately, but requires passing
`boundary_idx` into all attention modules' RoPE computations. Using ordinal 0..M is simpler
and is what H-Net implicitly assumes.

---

## 8. Verification requirement

Any Phase 3 attention module change that adds `key_padding_mask` to a tracked component
(`attention_rope_norms`, `mla_attention`, `diff_attention`) will mark that component stale
in `utils/verify.py`. Full Phase B' verification must be re-run for each modified component
before any Phase 3 training begins.
