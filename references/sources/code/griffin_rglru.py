# Source: https://github.com/google-deepmind/recurrentgemma/blob/main/recurrentgemma/torch/layers.py
# Official Google DeepMind RecurrentGemma implementation of the RG-LRU (Real-Gated Linear Recurrent Unit)
# from: "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models"
# De et al., 2024. arXiv:2402.19427
#
# License: Apache 2.0 (https://www.apache.org/licenses/LICENSE-2.0)
# Copyright 2024 DeepMind Technologies Limited.
#
# This file contains the verbatim RG-LRU forward pass and all supporting code extracted from
# recurrentgemma/torch/layers.py. Lightly trimmed to focus on RG-LRU; Conv1D and Einsum are
# omitted. Everything else is verbatim.

# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Base layers."""

from collections.abc import Sequence
import math
from typing import overload, Literal

import einops
from recurrentgemma.torch import array_typing as at
import torch
from torch import nn


_MAX_SQRT_GRADIENT = 1000.0


class BlockDiagonalLinear(nn.Module):
  """Block-diagonal linear layer."""

  def __init__(
      self,
      width: int,
      num_blocks: int,
      w_init_variance_scale: float = 1.0,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    """Initializes the BlockDiagonalLinear.

    Args:
      width: The number of dimensions of the input and output.
      num_blocks: The number of diagonal blocks in the layer.
      w_init_variance_scale: A parameters that scales the variance of the
        initialization of the weights.
      device: On what device to initialize parameters. Needed to allow for
        initializing the module without parameter initialzation.
      dtype: What dtype to use for initialziation.
    """
    super().__init__()
    self.width = width
    self.num_blocks = num_blocks
    self.w_init_variance_scale = w_init_variance_scale
    self.block_width = self.width // self.num_blocks

    # Parameters.
    self.w = nn.Parameter(torch.empty(
        [self.num_blocks, self.block_width, self.block_width],
        device=device,
        dtype=dtype
    ))
    self.b = nn.Parameter(torch.empty(
        [self.num_blocks, self.block_width], device=device, dtype=dtype
    ))

    # Initialization.
    self.reset_parameters()

  def reset_parameters(self) -> None:
    """Resets the parameters of the module."""
    self.w_init_(self.w)
    torch.nn.init.zeros_(self.b)

  def w_init_(self, w: torch.Tensor) -> None:
    """Initializes the weight `w` of the layer."""
    std = math.sqrt(self.w_init_variance_scale / self.block_width)
    torch.nn.init.normal_(w, mean=0.0, std=std)

  @at.typed
  def forward(self, x: at.ExpandedActivations) -> at.ExpandedActivations:
    """Calls the BlockDiagonalLinear."""
    # Split x to blocks.
    x = einops.rearrange(x, "... (h i) -> ... h i", h=self.num_blocks)

    # Linear layer over each block + bias.
    y = torch.einsum("... h i, h i j -> ... h j", x, self.w) + self.b

    # Flatten the output.
    return einops.rearrange(y, "... h j -> ... (h j)", h=self.num_blocks)


@at.typed
def rnn_scan(
    x: at.ExpandedActivations,
    a: at.ExpandedActivations,
    reset: at.Reset,
    h0: at.RNNState | None,
    acc_dtype: torch.dtype = torch.float32,
) -> tuple[at.ExpandedActivations, at.RNNState]:
  """Runs the recurrence of a linear RNN.

  Args:
    x: The input sequence.
    a: The diagonal of the recurrence matrix `A`.
    reset: Indicator of document boundaries, e.g. when to reset the hidden state
      of the RNN.
    h0: The initial hidden state.
    acc_dtype: The data type for the accumulation.

  Returns:
    The output of the linear recurrence.
  """
  assert x.ndim == 3
  assert a.shape == x.shape[-a.ndim :]
  assert a.dtype == x.dtype
  assert type(a) is type(x)
  assert h0 is None or h0.dtype == acc_dtype

  # Multiply `a` by the reset.
  a = a * ~reset[..., None]

  if x.shape[1] == 1:
    # Using scan in sampling mode.
    if h0 is None:
      return x, x[:, 0].type(acc_dtype)

    else:
      y = a.type(acc_dtype) * h0[:, None] + x.type(acc_dtype)
      return y.type(x.dtype), y[:, -1]

  else:
    # Using scan in linear mode.
    if h0 is not None:
      h_t = h0
    else:
      h_t = torch.zeros(x[:, 0].shape, dtype=acc_dtype, device=x.device)

    y = torch.zeros_like(x)
    for t in range(x.shape[1]):
      h_t = a[:, t].type(acc_dtype) * h_t + x[:, t].type(acc_dtype)
      y[:, t] = h_t.type(x.dtype)

  return y, h_t


def rnn_param_init(
    tensor: torch.Tensor,
    min_rad: float,
    max_rad: float,
    transform: str = "softplus",
    eps: float = 1e-8,
) -> torch.Tensor:
  """Initializes the `A` parameter of the RG-LRU uniformly on a ring."""
  with torch.no_grad():
    # Proportional to area in a ring.
    # 0.5 * jnp.log(unif * (max_rad**2 - min_rad**2) + min_rad**2 + 1e-8)
    tensor.uniform_(min_rad ** 2 + eps, max_rad ** 2 + eps)
    tensor.log_().mul_(0.5)

    if transform == "softplus":
      # Inverse transform.
      # jnp.log(jnp.exp(-a_real) - 1.0).astype(dtype)
      return tensor.neg_().exp_().sub_(1.0).log_()
    else:
      raise NotImplementedError()


class SqrtBoundDerivative(torch.autograd.Function):
  """Computes a square root with a gradient clipped at `_MAX_SQRT_GRADIENT`."""

  @staticmethod
  def forward(ctx, x: torch.Tensor) -> torch.Tensor:
    """The forward pass, which is a normal `sqrt`."""
    ctx.save_for_backward(x)
    return torch.sqrt(x)

  @staticmethod
  def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
    """The backward pass, which clips the `sqrt` gradient."""
    (x,) = ctx.saved_tensors
    clipped_x_times_4 = torch.clip(4.0 * x, min=1 / (_MAX_SQRT_GRADIENT**2))
    return grad_output / torch.sqrt(clipped_x_times_4)


class RGLRU(nn.Module):
  """A Real-Gated Linear Recurrent Unit (RG-LRU) layer."""

  def __init__(
      self,
      width: int,
      num_heads: int,
      w_init_variance_scale: float = 1.0,
      device: str | torch.device | None = None,
      dtype: torch.dtype | None = None,
  ):
    """Initializes the RG-LRU.

    Args:
      width: The number of dimensions of the input and output.
      num_heads: The number of diagonal blocks in the input and A gate layers.
      w_init_variance_scale: Initialization parameter for the
        BlockDiagonalLinear layers of the gates. See the `BlockDiagonalLinear`
        layer for details.
      device: On what device to initialize parameters. Needed to allow for
        initializing the module without parameter initialzation.
      dtype: What dtype to use for initialziation.
    """
    super().__init__()
    self.width = width
    self.num_heads = num_heads
    self.w_init_variance_scale = w_init_variance_scale

    # Parameters and layers.
    self.a_param = nn.Parameter(torch.empty(
        [self.width], device=device, dtype=dtype
    ))
    self.input_gate = BlockDiagonalLinear(
        width=self.width,
        num_blocks=self.num_heads,
        w_init_variance_scale=w_init_variance_scale,
        device=device,
        dtype=dtype,
    )
    self.a_gate = BlockDiagonalLinear(
        width=self.width,
        num_blocks=self.num_heads,
        w_init_variance_scale=self.w_init_variance_scale,
        device=device,
        dtype=dtype,
    )

    # Initialization
    self.reset_parameters()

  def reset_parameters(self) -> None:
    """Resets the parameters of the module."""
    self.input_gate.reset_parameters()
    self.a_gate.reset_parameters()
    self.a_param_init(self.a_param)

  def a_param_init(self, w: torch.Tensor) -> torch.Tensor:
    """Initializes the `A` parameter of the RG-LRU."""
    return rnn_param_init(w, min_rad=0.9, max_rad=0.999)

  @overload
  def forward(
      self,
      x: at.ExpandedActivations,
      segment_pos: at.SegmentPos,
      cache: at.RNNState | None = None,
      return_cache: Literal[True] = True,
  ) -> tuple[at.ExpandedActivations, at.RNNState]:
    ...

  @overload
  def forward(
      self,
      x: at.ExpandedActivations,
      segment_pos: at.SegmentPos,
      cache: at.RNNState | None = None,
      return_cache: Literal[False] = False,
  ) -> tuple[at.ExpandedActivations, None]:
    ...

  @at.typed
  def forward(
      self,
      x: at.ExpandedActivations,
      segment_pos: at.SegmentPos,
      cache: at.RNNState | None = None,
      return_cache: bool = True,
  ) -> tuple[at.ExpandedActivations, at.RNNState | None]:
    """Calls the RG-LRU.

    Args:
      x: Sequence of input activations.
      segment_pos: Position of each token in the sequence.
      cache: The previous hidden state of the RG-LRU.
      return_cache: Whether to compute and return the updated cache.

    Returns:
      Output of the block together with the updated hidden state.
    """

    bs, l, _ = x.shape
    assert segment_pos.shape == (bs, l)
    reset = segment_pos == 0

    # Gates for x and a.
    gate_x = torch.sigmoid(self.input_gate(x))
    gate_a = torch.sigmoid(self.a_gate(x))

    # Compute the parameter `A` of the recurrence.
    log_a = -8.0 * gate_a * nn.functional.softplus(self.a_param)
    a = torch.exp(log_a)
    a_square = torch.exp(2 * log_a)

    # Gate the input.
    gated_x = x * gate_x

    # Apply gamma normalization to the input. We need to clip the derivatives of
    # `sqrt` in order to prevent NaNs during training in bfloat16.
    multiplier = SqrtBoundDerivative.apply(1 - a_square)
    multiplier = reset[..., None] + ~reset[..., None] * multiplier
    normalized_x = gated_x * multiplier.type(x.dtype)

    y, last_h = rnn_scan(
        x=normalized_x,
        a=a,
        reset=reset,
        h0=cache,
    )

    if not return_cache:
      return y, None

    return y, last_h

  @classmethod
  def init_cache(
      cls,
      batch_size: int,
      width: int,
      device: str | torch.device | None = None,
  ) -> at.RNNState:
    """Returns an empty initialized cache for the RG-LRU."""
    # RG-LRU cache always in float32.
    return torch.zeros((batch_size, width), dtype=torch.float32, device=device)
