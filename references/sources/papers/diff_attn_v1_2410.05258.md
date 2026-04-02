# Differential Transformer V1

## Citation

**Title:** Differential Transformer
**Authors:** Tianzhu Ye, Li Dong, Yuqing Xia, Yutao Sun, Yi Zhu, Gao Huang, Furu Wei
**ArXiv:** https://arxiv.org/abs/2410.05258
**Submitted:** October 7, 2024
**Venue:** ICLR 2025
**BibTeX key:** ye2024differential

```bibtex
@inproceedings{ye2024differential,
  title={Differential Transformer},
  author={Ye, Tianzhu and Dong, Li and Xia, Yuqing and Sun, Yutao and Zhu, Yi and Huang, Gao and Wei, Furu},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```

---

## Abstract (verbatim excerpt)

"We introduce Diff Transformer, which amplifies attention to the relevant context while
canceling noise. Specifically, the differential attention mechanism computes attention
scores as the difference between two separate softmax attention maps. This approach
resembles the differential amplifier concept in electronics, which cancels out common-mode
noise."

---

## Core differential attention mechanism (V1)

### Notation

- h: number of attention heads (V1 uses h/2 differential heads, each operating over n/2 standard heads)
- d_h: head dimension (same as standard transformer)
- λ: scalar scalar noise-canceling coefficient

### V1 head layout

- Q1, Q2: each of shape [N, h/2, d_h] — split from a single [N, h, d_h] Q projection
- K1, K2: each of shape [N, h/2, d_h] — split from a single [N, h, d_h] K projection
- V: shape [N, h_kv, 2·d_h] — **double-wide** value per head

### V1 lambda parameterization (static, per-head, per-layer)

```
λ = exp(λ_{q1} · λ_{k1}) − exp(λ_{q2} · λ_{k2}) + λ_init
```

Where:
- λ_{q1}, λ_{k1}, λ_{q2}, λ_{k2} ∈ ℝ^{d_h} are learned per-head vectors
- λ_init is a per-layer constant: `λ_init(l) = 0.8 − 0.6 · exp(−0.3 · (l − 1))`
  - l = layer index (1-based): layer 1 → λ_init ≈ 0.2; layer 8 → λ_init ≈ 0.71

### V1 per-head RMSNorm

After the differential operation, V1 applies per-head RMSNorm scaled by `(1 − λ_init)`:

```
attn = rmsnorm(attn1 − λ · attn2) × (1 − λ_init)
```

### V1 pseudocode (from V2 blog reference)

```python
def DiffAttnV1(
        layer_index, q1, q2, k1, k2, v,
        lam_q1, lam_k1, lam_q2, lam_k2,
):
    """
    q1, q2: (N, h/2, d)
    k1, k2: (N, h_kv/2, d)
    v:      (N, h_kv/2, 2d)
    lam_*: (d,)
    """
    attn1 = flash_attn_func(q1, k1, v)
    attn2 = flash_attn_func(q2, k2, v)

    lam_init = 0.8 - 0.6 * exp(-0.3 * layer_index)
    lam1 = exp(sum(lam_q1 * lam_k1))
    lam2 = exp(sum(lam_q2 * lam_k2))
    lam = lam1 - lam2 + lam_init
    attn = attn1 - lam * attn2

    attn = rmsnorm(attn)
    attn = attn * (1 - lam_init)
    return attn
```

**Source:** Reproduced verbatim from the V2 blog comparison section
(https://huggingface.co/blog/microsoft/diff-attn-v2, accessed 2026-03-31).

---

## Key limitations of V1 (from V2 blog motivation section)

1. **Custom kernels required**: V is double-wide `[N, h_kv/2, 2d]`, so standard FlashAttention
   cannot be used without modification. Two separate forward passes over double-wide V are needed.

2. **Per-head RMSNorm instability**: At seq_len=8192, uniform attention gives RMS(c) ≈ 1/√8192,
   so RMSNorm amplifies by ~√8192 ≈ 90.5×. This causes gradient instability during large-scale
   pretraining with large learning rates.

3. **Static λ initialization**: λ_init(l) = 0.8 − 0.6·exp(−0.3·(l−1)) is a fixed scalar per
   layer with no token-specific or dynamic component. Hard to adapt per-context.

These three issues are all addressed in V2. See `diff_attn_v2_2026_01.md`.

---

## Note on V1 vs V2 for this codebase

**Our implementation uses V2 only.** V1 is documented here for reference and to explain
the design decisions visible in V2's changes. The `DifferentialCausalAttention` and
`DiffMLAAttention` classes in `phase1/model.py` implement V2 semantics; no V1 code
is present in the codebase.
