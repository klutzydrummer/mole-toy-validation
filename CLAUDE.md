# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Phase 1 toy validation of the MoLE architecture (Mixture-of-LoRAs Encoder) at ~5M params on enwik8 character-level LM. Four configs: `baseline`, `mhc`, `mol`, `compose`.

## Commands

```bash
bash setup.sh                              # one-time: downloads enwik8, prepares splits
bash run_experiments.sh                    # run all 4 configs in sequence
bash run_experiments.sh baseline           # run single config
python phase1/train.py --config mol --resume   # manual resume
```

## Key invariants — verify these before any change

### train.py ↔ run_experiments.sh argument contract
Every argument passed in `run_experiments.sh` must exist in the `argparse` block of `train.py`. The `train()` function signature and the `argparse` block must stay in sync. **This was the source of the first crash** (`--log_interval` existed in `train()` but not in argparse). Before adding any new CLI arg, add it to BOTH places.

To verify: `grep -o '\-\-[a-z_]*' run_experiments.sh | sort` must be a subset of `grep "add_argument" phase1/train.py | grep -o '"\-\-[a-z_]*"' | tr -d '"' | sort`

### mHC implementation correctness (arXiv:2512.24880, tokenbender/mHC reference)

The formula is: `x_{l+1} = H_res·x + H_post^T · F(H_pre·x)`

Critical checks:
- `sinkhorn_log`: log-space, `tau=0.05`, `n_iters=10`. Must handle both `[n,n]` (static) and `[B,L,n,n]` (dynamic) inputs via the same function.
- `H_res` init: `diag=0, off-diag=-8` (near-identity after sinkhorn)
- `H_pre` init: one random entry `0`, rest `-8` (near one-hot)
- `H_post` init: all zeros (uniform softmax = `1/n` per stream at start)
- `H_post` activation: **`softmax`**, NOT `softplus`. Softplus is unbounded and violates the normalization constraint.
- `H_pre` must read from the **original** `streams`, not from `mixed` (after H_res application). Both H_res and H_pre operate on the same input `x`.
- `_init_weights` uses `isinstance(m, nn.Linear)`. `LoRAAdapter` stores weights as `nn.Parameter`, so `_init_weights` does NOT touch them. `B=zeros` init is safe.

### MoL routing correctness (DeepSeek-V3, deepseek-ai/DeepSeek-V3 inference/model.py)
- `scores = sigmoid(logits)` — unbiased affinity
- `biased = scores + expert_bias` — bias added ONLY for topk selection
- `topk_weights = scores.gather(topk_idx)` — weights use **unbiased** scores, not biased
- Load balance sign: `bias += gamma * sign(avg - counts)`. Underloaded → avg > counts → positive sign → bias increases → more likely selected. This is correct.

## How to audit this codebase

Do NOT make changes without first reading every affected file completely. The correct workflow:

1. **Read all files** (`phase1/model.py`, `phase1/train.py`, `utils/data.py`, `utils/metrics.py`, `run_experiments.sh`, `setup.sh`)
2. **Cross-check argument contracts** (see invariant above)
3. **Verify against the reference implementations** in `/home/brandon/Projects/MoLE/reference/` — especially `mhc_hyper_connections.py` and `deepseek_v3_mla_moe.py`
4. **Fix root causes** — never paper over a mismatch by removing the flag or suppressing the error

## Architecture

```
phase1/model.py     ToyTransformer: baseline / mhc / mol / compose
phase1/train.py     Training loop: AMP, torch.compile, resume, eval
utils/data.py       enwik8 loader: CharDataset, get_dataloader
utils/metrics.py    TrainLogger (JSONL), ParamCounter, ce_to_bpc
setup.sh            Download enwik8, prepare .npy splits
run_experiments.sh  Run all 4 configs, auto-resume, print summary table
```

Base config: `d=256, n_layers=8, n_heads=8, SwiGLU, RoPE, RMSNorm, weight-tying`

mHC adds: `n=2 streams`, stream expand at embed → stream collapse (learned softmax weights) before lm_head.

MoL replaces SwiGLU with: `8 experts, top-2, rank-4 LoRA`, shared LoRA always active, per-expert LoRA weighted by unbiased router scores.
