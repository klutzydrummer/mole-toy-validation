# Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models

## Full Citation

**Authors:** Soham De, Samuel L. Smith, Anushan Fernando, Aleksandar Botev, George Cristian-Muraru,
Albert Gu, Ruba Haroun, Leonard Berrada, Yutian Chen, Srivatsan Srinivasan, Guillaume Desjardins,
Arnaud Doucet, David Budden, Yee Whye Teh, Razvan Pascanu, Nando De Freitas, and Caglar Gulcehre

**Title:** Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models

**Year:** 2024

**arXiv ID:** 2402.19427

**Submitted:** February 29, 2024

**arXiv abstract URL:** https://arxiv.org/abs/2402.19427

**arXiv HTML URL:** https://arxiv.org/html/2402.19427

**Official code:** https://github.com/google-deepmind/recurrentgemma

---

## Abstract

"Recurrent neural networks (RNNs) have fast inference and scale efficiently on long sequences, but
they are difficult to train and hard to scale. We propose Hawk, an RNN with gated linear recurrences,
and Griffin, a hybrid model that mixes gated linear recurrences with local attention. Hawk exceeds
the reported performance of Mamba on downstream tasks, while Griffin matches the performance of
Llama-2 despite being trained on over 6 times fewer tokens. We also show that Griffin can extrapolate
on sequences significantly longer than those seen during training. Our models match the hardware
efficiency of Transformers during training, and during inference they have lower latency and
significantly higher throughput. We scale Griffin up to 14B parameters, and explain how to shard our
models for efficient distributed training."

---

## Section 2.4: Real-Gated Linear Recurrent Unit (RG-LRU)

The RG-LRU is the core recurrent building block used in both Hawk and Griffin. It is defined by the
following four equations.

### Equations (Section 2.4)

**Equation (1) — Recurrence gate:**

    r_t = σ(W_a x_t + b_a),    recurrence gate

**Equation (2) — Input gate:**

    i_t = σ(W_x x_t + b_x),    input gate

**Equation (3) — Recurrent weight:**

    a_t = a^(c · r_t)

**Equation (4) — Hidden state update:**

    h_t = a_t ⊙ h_{t-1} + √(1 − a_t²) ⊙ (i_t ⊙ x_t)

Where:
- σ is the sigmoid function
- ⊙ denotes element-wise multiplication
- c is a scalar constant set to **8**
- a is a learnable diagonal parameter, parameterized as **a = σ(Λ)** where Λ is a learnable vector
- All operations are element-wise (the recurrent weight a is diagonal)

### Output

"The output of the layer is y_t = h_t"

### Parameter Parameterization

"The recurrent weight a in Equation (4) is diagonal. Hence all operations are element-wise. We
parameterize a in Equation (3) as a = σ(Λ), where Λ is a learnable parameter."

"This guarantees that 0 ≤ a ≤ 1, ensuring that the recurrence is stable. The variable c is a
scalar-valued constant set to 8."

### Initialization

"We initialize both W_a and W_x using LeCun init. We initialize Λ such that a^c is uniformly
distributed between 0.9 and 0.999 at the start of training, similar to Orvieto et al. [2023b]."

In the official PyTorch implementation this is done by sampling `a^c` (i.e., `a_param` after
softplus) uniformly in [0.9², 0.999²] in squared-radius space (proportional to ring area), then
taking the log and applying the inverse softplus transform. See `rnn_param_init` in the code.

### Gate Independence (Efficiency Property)

"The layer has gates on both the input x and the recurrent weight a. However, neither gate depends
on the recurrent state h_{t-1}, which ensures that the computation can be executed efficiently on
device."

### Real vs. Complex

"Unlike the original LRU layer, we do not use complex algebra in the recurrence. While using complex
recurrences would lead to a more expressive layer [Orvieto et al., 2023a] we found that complex
recurrences were not beneficial for language modelling in practice, as also observed by Gu and Dao
[2023]."

---

## Appendix A: Log-Space Computation and Gate Behaviour

### Equation (6) — Log-space computation of a_t

For numerical stability, a_t is computed in log-space rather than directly as a power:

    log a_t = log a^(c·r_t)
            = log σ(Λ)^(c·r_t)
            = −c · softplus(Λ) ⊙ r_t

This is mathematically equivalent to Equation (3) but avoids overflow/underflow in the power
operation. In code: `log_a = -8.0 * gate_a * softplus(a_param)`, then `a = exp(log_a)`.

### Gate Behaviour Analysis

"Our gate is quite different than other standard gates in the literature... Ours on the other hand
is biased towards retaining information, and does not allow to fully discard the contribution of
h_{t-1}."

"For example, the selection mechanism proposed in Mamba [Gu and Dao, 2023] is comparable to the
update gate of GRUs which interpolates between the previous state and the current observation x_t.
Its effect on the hidden state allows it to reset its state and forget any information it holds from
the past, similar to the forget gate in the LSTM."

"In contrast, our recurrence gate can approximately interpolate between the standard LRU update from
Orvieto et al. [2023a] and the previous hidden state, which allows it to effectively discard the
input and preserve all information from the previous history."

"We believe the key role of this gate is to enable the model to achieve super-exponential memory by
reducing the influence of uninformative inputs."

The gate behavior is analyzed using the functions:

    α(r_t) = a_t = a^(c·r_t)
    β(r_t) = √(1 − α(r_t)²)

---

## The √(1 − a_t²) Normalization Term

The multiplier `√(1 − a_t²)` applied to the gated input `i_t ⊙ x_t` in Equation (4) is a
**gamma normalization** term. It ensures that the input contribution is scaled to maintain the
approximate norm of the hidden state h_t.

Intuition: if h_{t-1} has unit variance and x_t has unit variance, and a_t ∈ [0,1], then:

    Var[h_t] = a_t² · Var[h_{t-1}] + (1 − a_t²) · Var[i_t ⊙ x_t]
             ≈ a_t² + (1 − a_t²)
             = 1

So the coefficient `√(1 − a_t²)` is chosen precisely so that h_t has approximately unit variance
under the assumption that h_{t-1} and the gated input have unit variance. This is analogous to
the normalization in the original LRU paper (Orvieto et al. 2023).

In the official implementation, the gradient of `sqrt` is clipped at `_MAX_SQRT_GRADIENT = 1000.0`
via a custom autograd function `SqrtBoundDerivative` to prevent NaN gradients in bfloat16 training.

At sequence boundaries (segment_pos == 0), the multiplier is set to 1.0 (reset: no recurrent
contribution), so `h_t = i_t ⊙ x_t` at the start of a new document.

---

## Key Hyperparameters (from paper)

- Scalar constant: **c = 8**
- a parameterization: **a = σ(Λ)**, so a ∈ (0, 1)
- Λ initialization: a^c uniform in **[0.9, 0.999]** at training start
- W_a, W_x: **LeCun initialization**
- The recurrent state h_t is stored in **float32** even during bfloat16 training
- Gate layers are **BlockDiagonalLinear** (block-diagonal weight matrices, num_heads blocks)
- The constant c=8 means the effective per-step decay `a^c` is in [0.9, 0.999] by initialization

---

## Architecture Context

The RG-LRU is embedded inside a "recurrent block" that has the following structure:

    input x  →  [temporal conv (width 4)]  →  split into two branches:
        branch 1: RG-LRU → linear projection
        branch 2: GeLU gate
    element-wise product of branches → output

This is the residual block used in both Hawk and Griffin. Griffin interleaves these recurrent blocks
with local sliding-window attention blocks (window size 1024 by default).

---

## Related Work / References Cited in Paper

- Orvieto et al. [2023a]: "Resurrecting Recurrent Neural Networks for Long Sequences" (LRU paper,
  complex-valued recurrences)
- Orvieto et al. [2023b]: initialization strategy for LRU (uniform a^c in [0.9, 0.999])
- Gu and Dao [2023]: Mamba (structured state space model, selection mechanism)
- GRU: update gate compared to RG-LRU's recurrence gate
- LSTM: forget gate compared to RG-LRU's recurrence gate

---

## Notes for Implementation Verification

The critical equations to verify in any RG-LRU implementation:

1. **Gate computation:**
   ```
   r_t = sigmoid(W_a @ x_t + b_a)   # recurrence gate (controls a_t)
   i_t = sigmoid(W_x @ x_t + b_x)   # input gate (scales x_t)
   ```

2. **Recurrent coefficient (log-space for stability):**
   ```
   log_a_t = -c * softplus(Λ) * r_t      # c = 8
   a_t = exp(log_a_t)
   a_t_squared = exp(2 * log_a_t)
   ```

3. **Hidden state update:**
   ```
   gated_x = i_t * x_t
   gamma = sqrt(1 - a_t_squared)          # norm-preserving multiplier
   h_t = a_t * h_{t-1} + gamma * gated_x
   ```

4. **Sequence boundary reset:** at segment start, set `a_t = 0` (zero out the recurrent
   contribution), and set `gamma = 1.0` (so `h_t = gated_x`).

5. **State dtype:** accumulate h_t in **float32** regardless of input dtype.

6. **Gradient stability:** clip gradient of sqrt at `_MAX_SQRT_GRADIENT = 1000.0`.
