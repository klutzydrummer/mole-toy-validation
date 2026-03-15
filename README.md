# Architecture v0.3.1 — Toy Validation (Phase 1)

Validates mHC residual streams and Mixture-of-LoRAs FFN at ~5M params
on enwik8 character-level LM.

## Deploying on Lightning.ai

### First time setup

1. Push this repo to GitHub (one time):
   ```bash
   # From your local machine, inside this directory:
   git remote add origin https://github.com/<you>/mole-toy-validation.git
   git push -u origin main
   ```

2. In your Lightning.ai Studio, open a terminal and run:
   ```bash
   git clone https://github.com/<you>/mole-toy-validation.git
   cd mole-toy-validation
   pip install -r requirements.txt
   bash setup.sh
   ```

### Running experiments

Run all 4 configs in sequence (~4-5 hours on T4):
```bash
bash run_experiments.sh
```

Or run one config at a time:
```bash
bash run_experiments.sh baseline
bash run_experiments.sh mhc
bash run_experiments.sh mol
bash run_experiments.sh compose
```

### Handling interrupted GPU sessions

If your Lightning.ai session pauses mid-run, the script auto-resumes from the last checkpoint:
```bash
bash run_experiments.sh mol    # picks up where it left off
```

Or manually:
```bash
python phase1/train.py --config mol --resume
```

### Syncing results back

From the Studio terminal:
```bash
git add checkpoints/*_summary.json checkpoints/*.jsonl
git commit -m "phase1 results"
git push
```

---

## Architecture

Four configs sharing the same base (d=256, 8 layers, 8 heads, SwiGLU, RoPE, RMSNorm):

**baseline**: Vanilla transformer. Control. Expected ~1.2–1.4 BPC.

**mhc**: Adds n=2 manifold-constrained hyper-connection streams. Each sub-layer
uses the full mHC update `x = H_res · x + H_post^T · F(H_pre · x)` with three
separate matrices per the mHC paper (arXiv:2512.24880). H_post's per-stream
weights cause streams to diverge even from identical initialization.

**mol**: Adds MoL FFN (8 rank-4 LoRA experts, top-2 sigmoid routing). Weight
composition: corrections added before silu (gate/up) and after (down).
Routing follows DeepSeek-V3: bias for selection only, unbiased for weights.

**compose**: Both mHC + MoL.

## Key implementation decisions grounded in references

| Decision | Reference |
|----------|-----------|
| mHC: H_res + H_pre + H_post (not just H_res) | tokenbender/mHC, lucidrains/hyper-connections, HC paper Fig 2b |
| MoL: bias for selection, unbiased for weights | deepseek-ai/DeepSeek-V3 inference/model.py Gate class |
| MoL: silu applied once to combined projections | Architecture doc Section 8 step 10, Section 7.1 |
| MoL: one_hot for routing weights (not scatter_) | Differentiable gradient flow through topk_weights |
| Sinkhorn: clamp(-20,20) before exp | Prevent float32 overflow |
| LoRA B=0 init via nn.Parameter | Safe from _init_weights; standard LoRA convention |

## Files
```
phase1/model.py         — ToyTransformer (all 4 configs)
phase1/train.py         — Training loop (AMP, compile, resume)
utils/data.py           — enwik8 loader
utils/metrics.py        — BPC, logging
setup.sh                — One-time environment + data setup
run_experiments.sh      — Run all 4 configs with auto-resume
requirements.txt        — Python dependencies
```
