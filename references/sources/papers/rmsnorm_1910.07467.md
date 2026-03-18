# RMSNorm: Root Mean Square Layer Normalization

## Citation

Zhang, B., & Sennrich, R. (2019).
**Root Mean Square Layer Normalization.**
arXiv:1910.07467 [cs.LG].
https://arxiv.org/abs/1910.07467

## Abstract

The authors propose RMSNorm as a computationally efficient alternative to
LayerNorm. Their key insight is that "re-centering invariance in LayerNorm is
dispensable," allowing them to simplify the normalization process using root mean
square calculations instead. This modification maintains performance comparable to
LayerNorm while reducing computational overhead by 7–64% across different
architectures. The paper also introduces pRMSNorm, which estimates the RMS from
only a portion of the inputs.

## Key Equations

### LayerNorm (for contrast)

Standard LayerNorm normalizes by both mean and variance:

```
LayerNorm(x) = g · (x - μ) / sqrt(σ² + ε) + b

where:
  μ = (1/d) Σ x_i          (mean centering)
  σ² = (1/d) Σ (x_i - μ)²  (variance)
  g  = learnable scale parameter (shape: d)
  b  = learnable bias parameter  (shape: d)
  ε  = small constant for numerical stability
```

### RMSNorm Formula

RMSNorm drops mean centering entirely. The formula is:

```
RMSNorm(x) = g · x / RMS(x)

where:
  RMS(x) = sqrt( (1/d) Σ x_i² + ε )
          = sqrt( x.pow(2).mean(-1, keepdim=True) + ε )

  g = learnable scale parameter (shape: d), init to ones
  ε = small constant (typically 1e-6 or 1e-8)
```

Equivalently using `rsqrt`:

```
RMSNorm(x) = g · x · rsqrt( mean(x², dim=-1, keepdim=True) + ε )
```

### Key Differences from LayerNorm

| Property           | LayerNorm     | RMSNorm         |
|--------------------|---------------|-----------------|
| Mean centering     | Yes (μ subtracted) | **No**     |
| Variance scaling   | Yes           | Replaced by RMS |
| Learnable scale g  | Yes           | Yes             |
| Learnable bias b   | Yes           | **No** (typically omitted) |
| Numerical formula  | (x-μ)/σ       | x/RMS(x)        |

### Epsilon Placement

Epsilon is added **inside the square root** (to the mean of squares), not
added to the RMS after the sqrt:

```
# Correct:
rms = sqrt(mean(x^2) + eps)

# Incorrect:
rms = sqrt(mean(x^2)) + eps
```

### Partial RMSNorm (pRMSNorm)

An optional variant estimates the RMS from the first `p` fraction of elements:

```
pRMSNorm(x) = g · x / RMS(x[:p*d])
```

This is disabled (full RMSNorm) when p = -1 or p is outside [0, 1].

## Implementation Notes for Verification

A correct RMSNorm implementation must satisfy:

1. **No mean subtraction**: unlike LayerNorm, do not compute or subtract the
   mean of x.

2. **RMS formula**: `rms = sqrt(x.pow(2).mean(-1, keepdim=True) + eps)`
   or equivalently `torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)`.

3. **Epsilon inside the sqrt**: adding epsilon after the sqrt is incorrect.

4. **Scale parameter g**: shape `(d,)`, initialized to ones, applied as
   element-wise multiplication after normalization.

5. **No bias by default**: RMSNorm does not enforce re-centering, so a bias
   is usually omitted (unlike LayerNorm).

6. **Cast for numerical stability**: LLaMA casts to float32 for the norm
   computation, then casts back: `x * torch.rsqrt(...).type_as(x)`.

7. **Normalization dimension**: always the last dimension (feature dimension),
   not sequence or batch dimensions.
