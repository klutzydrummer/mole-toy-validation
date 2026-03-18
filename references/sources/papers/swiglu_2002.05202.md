# SwiGLU: GLU Variants Improve Transformer

## Citation

Shazeer, N. (2020).
**GLU Variants Improve Transformer.**
arXiv:2002.05202 [cs.LG].
https://arxiv.org/abs/2002.05202

## Abstract

The paper investigates gated linear units (GLUs) and their variations as
alternatives to standard activation functions in Transformer FFN layers. GLUs
"consist of the component-wise product of two linear projections, one of which
is first passed through a sigmoid function." The research explores replacing
sigmoid with other nonlinearities (ReLU, GELU, Swish) and tests the resulting
variants in T5-style Transformer FFN sublayers. Certain GLU variants — in
particular SwiGLU — outperform the conventionally used ReLU or GELU activations.

## Key Equations

### Original GLU (Dauphin et al., 2017)

```
GLU(x, W, V, b, c) = σ(xW + b) ⊙ (xV + c)

where:
  σ = sigmoid
  ⊙ = element-wise multiplication
  W, V = weight matrices (projection to hidden_dim)
  b, c = biases
```

### Swish Activation

Swish (Ramachandran et al., 2017), also called SiLU:

```
Swish(x) = x · σ(x) = x · sigmoid(x)

In PyTorch: F.silu(x)
```

### SwiGLU Definition

SwiGLU replaces the sigmoid gate with Swish:

```
SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊙ (xV + c)
                       = (xW · σ(xW)) ⊙ (xV)
```

In practice biases are omitted (as in LLaMA), giving:

```
SwiGLU(x, W, V) = F.silu(x @ W.T) * (x @ V.T)
```

### FFN with SwiGLU

The full FFN sublayer using SwiGLU is:

```
FFN_SwiGLU(x, W1, W2, W3) = (Swish(x @ W1.T) * (x @ W3.T)) @ W2.T
```

Where:
- W1: gate projection  (d_model → d_ffn)
- W3: value projection (d_model → d_ffn)  [called W3 in LLaMA, V in the paper]
- W2: output projection (d_ffn → d_model)

This requires **three** weight matrices instead of the two used in standard FFN.

### Dimension Expansion Factor

Standard Transformer FFN uses hidden_dim = 4 × d_model.

SwiGLU uses **two** up-projections instead of one. To keep parameter count
comparable to a 4× standard FFN with two matrices, the hidden dimension is
reduced to (2/3) × 4 × d_model = **8/3 × d_model ≈ 2.667 × d_model**.

LLaMA's concrete calculation:

```python
hidden_dim = int(2 * hidden_dim / 3)  # hidden_dim starts as 4 * dim
# then round up to nearest multiple_of (e.g. 256) for hardware efficiency
```

Parameter count comparison (d = d_model, h = hidden_dim):
- Standard FFN: 2 × d × h  matrices  (W1: d→h, W2: h→d)
- SwiGLU FFN:   3 × d × h' matrices  (W1, W2, W3: all d↔h')
- For equal params: 3h' = 2h → h' = 2h/3; with h=4d: h' = 8d/3

### All GLU Variants Listed in the Paper

```
FFN_ReLU(x, W1, W2, b, V)   = max(0, xW1 + b) · W2
FFN_GELU(x, W1, W2, b, V)   = GELU(xW1 + b) · W2
FFN_Swish(x, W1, W2, b, V)  = Swish(xW1 + b) · W2

GLU(x, W, V, b, c)          = σ(xW + b) ⊙ (xV + c)
ReGLU(x, W, V, b, c)        = ReLU(xW + b) ⊙ (xV + c)
GEGLU(x, W, V, b, c)        = GELU(xW + b) ⊙ (xV + c)
SwiGLU(x, W, V, b, c)       = Swish(xW + b) ⊙ (xV + c)   ← best performer
```

## Implementation Notes for Verification

A correct SwiGLU FFN implementation must satisfy:

1. **Three weight matrices**: gate (W1), up/value (W3), down (W2). Some
   implementations name them gate_proj, up_proj, down_proj.

2. **Formula**: `output = down_proj(silu(gate_proj(x)) * up_proj(x))`
   The SiLU/Swish is applied to the gate branch only; the up branch is linear.

3. **Dimension**: hidden_dim should be (8/3) × d_model, rounded up to a
   multiple of some alignment value (256 or 64 are common). Do NOT use 4×.

4. **No bias**: LLaMA-style implementations omit bias terms in all three
   projections.

5. **Element-wise multiply**: the gate and value projections are combined with
   element-wise (Hadamard) product `*`, not dot product or addition.

6. **Swish = SiLU**: `F.silu(x)` in PyTorch is the correct function.
   `torch.sigmoid(x) * x` is equivalent but slower.
