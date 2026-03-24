"""
Accelerator detection and SDPA kernel configuration.

Usage in train.py:
    from utils.device import get_device, configure_sdpa
    device   = get_device()
    amp_info = configure_sdpa(device)   # sets backends, returns log string
"""

import importlib.util

import torch


def get_device() -> torch.device:
    """
    Return the best available device: CUDA > XLA (TPU) > CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if importlib.util.find_spec("torch_xla") is not None:
        import torch_xla.core.xla_model as xm  # type: ignore[import]
        return xm.xla_device()
    return torch.device("cpu")


def configure_sdpa(device: torch.device) -> str:
    """
    Set PyTorch SDPA backend flags for the detected accelerator and return a
    one-line description suitable for logging.

    CUDA sm >= 80 (A100, H100, ...):
        flash + mem_efficient + math — FlashAttention-2 available; PyTorch picks best.
    CUDA sm < 80 (T4 sm75, V100 sm70, ...):
        mem_efficient + math — FlashAttention-2 unsupported; mem_efficient covers sm70+.
    XLA / TPU:
        No CUDA flags set. XLA dispatches fused attention internally through its own
        compiler passes; attempting to configure CUDA backends would be a no-op or error.
    CPU:
        math only — the only SDPA path available without a GPU.
    """
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability()
        sm  = cap[0] * 10 + cap[1]
        name = torch.cuda.get_device_name()
        if cap[0] >= 8:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            return f"flash+mem_efficient+math | {name} (sm{sm})"
        else:
            # FlashAttention-2 requires sm80 — disable to avoid silent fallback warnings.
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            return f"mem_efficient+math | {name} (sm{sm}, flash requires sm80+)"
    elif device.type == "xla":
        # TPU: torch_xla intercepts F.scaled_dot_product_attention and
        # lowers it to an XLA fused attention op. No CUDA backend flags apply.
        return "xla-fused (TPU)"
    else:
        return "math (CPU)"


def get_amp_config(device: torch.device):
    """
    Return (use_amp, amp_dtype, use_grad_scaler) for the given device.

    - CUDA sm >= 80: bfloat16 (native BF16 tensor cores, no GradScaler needed)
    - CUDA sm < 80:  float16  (native FP16 tensor cores, GradScaler required)
    - XLA / TPU:     bfloat16 (TPU natively supports BF16; no GradScaler)
    - CPU:           no AMP
    """
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:
            return True, torch.bfloat16, False
        else:
            return True, torch.float16, True
    elif device.type == "xla":
        return True, torch.bfloat16, False
    else:
        return False, torch.float32, False
