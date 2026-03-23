# MoLE Toy Validation

Systematic ablation study of three architectural components for language model efficiency:
**mHC** (manifold-constrained hyper-connections), **MoL** (mixture-of-LoRAs FFN routing),
and **HDC** (hierarchical dynamic compression). Validated at 28M parameters before scaling.

Training runs on cloud (Lightning.ai T4). All results are reproducible from this repo.

---

## Status

| Phase | Configs | Status |
|-------|---------|--------|
| Phase 1 — component ablation at 28M params | baseline, mhc, mol, compose + wide/single variants | In progress |
| Phase 2 — HDC compression pipeline | hdc_rulebased, hdc_gate, hdc_stride, r2/r8 variants | Pending Phase 1 |
| Phase 3 — 1B scale validation | baseline_1b, mol_1b, mhc_lite_1b, compose_1b | Hypothetical — see PHASE3_STUDY_DESIGN.md |

---

## Phase 1 Results (partial — runs in progress)

| Config | Best val BPC | Params | Notes |
|--------|-------------|--------|-------|
| compose | 3.5316 | 31.1M | Best overall |
| mol | 3.5702 | 31.1M | — |
| mhc | 3.5736 | 27.8M | Lost to baseline — see mhc.md for root cause |
| baseline | 3.5875 | 27.8M | Control |

**Note:** These are preliminary single-seed results. The mol vs. baseline comparison above
conflates architecture with capacity (mol has 3.35M more params). The controlled comparisons
are `mol vs. baseline_wide` (Q1, same total params) and `mol vs. mol_single` (Q2, same LoRA
budget). Multi-seed reruns of all primary configs are pending. See `STUDY_DESIGN.md` §1.

Full results in `checkpoints/report.md` (regenerate: `python utils/reporter.py`).

---

## Architecture

Base config: `d=512, n_layers=8, n_heads=8, SwiGLU, RoPE, RMSNorm, weight-tying`
Dataset: WikiText-103-raw (CC-BY-SA 3.0, derived from Wikipedia), BPE tokenizer (vocab=4096, sentencepiece), seq_len=256

### Phase 1 configs

**baseline** — Vanilla transformer. Control.

**mhc** — Adds n=4 hyper-connection streams (arXiv:2512.24880) with **KromHC** exact
doubly stochastic H_res (arXiv:2601.21579). Each sublayer: `x = H_res · x + H_post^T · F(H_pre · x)`.
H_res is computed via Kronecker-product factorization (two 2×2 factors, `U1 ⊗ U2`) —
exactly doubly stochastic with no Sinkhorn iterations. Phase 1 original run used n=2 and
approximate SK; this re-run fixes both. See `references/components/mhc.md`.

**mol** — Replaces the SwiGLU FFN with Mixture-of-LoRAs routing (arXiv:2412.19437 §3).
8 rank-8 LoRA experts, top-2 sigmoid routing. Routing bias used for selection only
(unbiased for weight computation), matching DeepSeek-V3 Gate class.

**compose** — mHC + MoL simultaneously.

**baseline_wide** — Dense SwiGLU with d_ff scaled to match mol's total params (~31.1M).
Parameter-controlled baseline for Q1: does routing help beyond raw capacity?

**mol_single** — Single LoRA at the same total rank as mol (rank=64). Tests whether
routing over low-rank adapters beats a single high-rank adapter.

### Phase 2 configs (HDC pipeline)

Wraps the Phase 1 mol inner network with Zone E (content-aware compression via
CausalRecurrenceLayer + BoundaryRouter) and Zone D (reconstruction).

Run order: `hdc_rulebased → hdc_gate → hdc_stride → hdc_r2/r8`

Phase 2 gates: smoke test (`utils/smoke_test.py`) + model hash verification (`utils/verify.py`)
must pass before any Phase 2 run. See CLAUDE.md for the 9.7 BPC plateau post-mortem.

---

## Running experiments

```bash
bash run_experiments.sh              # all Phase 1 configs
bash run_experiments.sh phase1       # Phase 1 only
bash run_experiments.sh phase2       # Phase 2 only (requires smoke test + verify gates)
bash run_experiments.sh baseline     # single config
```

Auto-resume: if a checkpoint exists, the script resumes automatically.

Manual resume:
```bash
python phase1/train.py --config mol --resume
python phase2/train.py --config hdc_gate --resume
```

---

## Key implementation decisions

| Decision | Reference |
|----------|-----------|
| mHC H_res: log-space Sinkhorn, τ=0.05, 10 iters | tokenbender/mHC reference code |
| mHC H_pre/H_post: softmax (not sigmoid as in paper Eq. 8) | tokenbender/mHC reference code — matches our implementation |
| MoL: routing bias for selection only, unbiased for weights | deepseek-ai/DeepSeek-V3 `Gate` class |
| MoL: one_hot for routing weights (not scatter_) | Differentiable gradient flow through topk_weights |
| CausalRecurrenceLayer: log_a_init=3.0 (Zone E), 0.0 (Zone D) | Prevents 9.7 BPC plateau — see smoke_test.py |
| AMP dtype: float16 on T4 (sm<80), bfloat16 on Ampere+ | Auto-detected via `get_device_capability()` |

---

## Files

```
phase1/model.py         ToyTransformer — baseline / mhc / mol / compose
phase1/train.py         Training loop — AMP, torch.compile, auto-resume
phase2/model.py         HDCModel — ZoneE + InnerTransformer + ZoneD
phase2/train.py         Dual-loss training loop (loss_ntp + λ_comp * loss_comp)
utils/data.py           TokenDataset, get_dataloader (WikiText-103)
utils/metrics.py        TrainLogger (JSONL), ParamCounter, ce_to_bpc
utils/smoke_test.py     Phase 2 pre-flight: 1000-step health check
utils/verify.py         Hash-based model staleness check
utils/reporter.py       Auto-generates checkpoints/report.md
run_experiments.sh      Orchestrates all configs with gates and auto-resume
references/components/  Authoritative specs for each architectural component
references/sources/     Reference papers and verbatim reference implementations
references/verification/ Phase B' verification reports
```

---

## Methodology documentation

- `STUDY_DESIGN.md` — experimental protocol, controlled comparisons, falsifiability
- `TRAINING_METHODOLOGY.md` — training procedure, AMP, optimizer, LR schedule
- `PHASE3_STUDY_DESIGN.md` — planned 1B scale validation with KromHC/mHC-lite
- `GRANT_READINESS.md` — pre-application audit checklist for Google TRC
- `references/components/` — per-component specs with equations, deviations, and verification checklists

---

## Deploying on Lightning.ai

```bash
# First time — in Studio terminal:
git clone https://github.com/<you>/mole-toy-validation.git
cd mole-toy-validation
pip install -r requirements.txt
bash setup.sh

# Run experiments:
bash run_experiments.sh

# Sync results back:
git add checkpoints/
git commit -m "phase1 results"
git push
```
