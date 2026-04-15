# MoLE Toy Validation

Systematic ablation study of architectural components for language model efficiency:
**MoL** (mixture-of-LoRAs FFN routing), **mHC** (manifold-constrained hyper-connections
via go-mHC, arXiv:2604.02309), **nGPT** (hyperspherical hidden states, arXiv:2410.01131),
and **HDC** (hierarchical dynamic compression). Validated at 28M parameters before scaling.

Training runs on cloud (Lightning.ai T4). All results are reproducible from this repo.

---

## Study structure

The project is organized into 5 Phase 1 study groups + Phase 2 outer encoder study.
See `STUDY_DESIGN.md` for research questions, controls, and execution order.

| Study | Target | Configs | Research question |
|-------|--------|---------|------------------|
| A — MoLE core | `study_mole` | baseline, baseline_wide, mol, mol_single, mhc, compose | Q1: routing vs. capacity; Q2: routing vs. high-rank adapter |
| B — Attention | `study_attention` | mla, diff_attn, diff_mla | Q4: diff attn; Q5: MLA KV compression cost |
| C — mHC compose | `study_mhc_compose` | diff_mhc, mla_mhc, diff_mla_mhc | Q6: mHC + attention variant compositions |
| D — nGPT | `study_ngpt` | ngpt, ngpt_mla, ngpt_diff_attn | Q7: hyperspherical constraint benefit |
| E — Multi-sphere | `study_sphere` | ngpt_mhc_a, ngpt_mhc_c | Q8: multi-sphere vs. wrap-sublayer mHC |
| F — Outer encoder | `phase2` | 10 outer_* configs | Q3: content-aware vs. fixed-stride compression |

---

## Phase 1 Results

All Phase 1 runs are being re-run from scratch with the current codebase (go-mHC,
arXiv:2604.02309). Prior seed42 results are superseded — they were produced by an
unconfirmed implementation mix during the KromHC → go-mHC migration in April 2026 and
are not reproducible from the current repo.

Results will be posted here as studies complete. See `checkpoints/report.md` for live
training state (regenerate: `python utils/reporter.py`).

See `STUDY_DESIGN.md` §5 for statistical requirements (3 seeds for primary claims).

---

## Architecture

Base config: `d=512, n_layers=8, n_heads=8, SwiGLU, RoPE, RMSNorm, weight-tying`
Dataset: WikiText-103-raw (CC-BY-SA 3.0), BPE tokenizer (vocab=4096, sentencepiece), seq_len=256

### Phase 1 component specs

Authoritative correctness criteria live in `references/components/`:

| Component | Spec | Key references |
|-----------|------|---------------|
| mHC (go-mHC) | `references/components/mhc.md` | arXiv:2512.24880, arXiv:2604.02309 |
| MoL routing | `references/components/mol_ffn.md` | arXiv:2412.19437 |
| MLA attention | `references/components/mla_attention.md` | arXiv:2405.04434 |
| Diff Attn V2 | `references/components/diff_attention.md` | arXiv:2410.05258 |
| nGPT | `references/components/ngpt.md` | arXiv:2410.01131 |

### Phase 2 configs (outer encoder study, Q3)

Wraps the Phase 1 mol inner network with Zone E (content-aware compression via
CausalRecurrenceLayer + BoundaryRouter) and Zone D (reconstruction). 10 configs.

Phase 2 gates: smoke test (`utils/smoke_test.py`) + model hash verification (`utils/verify.py`)
must pass before any Phase 2 run. See CLAUDE.md for the 9.7 BPC plateau post-mortem.

---

## Running experiments

```bash
bash run_experiments.sh study_mole         # Study A: MoLE core (Q1, Q2)
bash run_experiments.sh study_attention    # Study B: attention variants
bash run_experiments.sh study_mhc_compose  # Study C: mHC + attention compositions
bash run_experiments.sh study_ngpt         # Study D: nGPT hyperspherical
bash run_experiments.sh study_sphere       # Study E: multi-sphere compositions
bash run_experiments.sh phase1_scaling     # scaling study: 5 configs × d=256,768
bash run_experiments.sh phase2             # Phase 2: outer encoder study
bash run_experiments.sh phase1             # all 17 Phase 1 configs
bash run_experiments.sh baseline           # single config
```

Auto-resume: if a checkpoint exists, the script resumes automatically.

Manual resume:
```bash
python phase1/train.py --config mol --resume
python phase2/train.py --config outer_crl --resume
```

---

## Key implementation decisions

| Decision | Reference |
|----------|-----------|
| mHC H_res: go-mHC Cayley transform + block Frobenius projection | arXiv:2604.02309, Dandachi & Diggs-Galligan |
| mHC H_pre/H_post: softmax (not sigmoid) | mHC reference code |
| MoL: routing bias for selection only, unbiased for weights | deepseek-ai/DeepSeek-V3 `Gate` class |
| nGPT: row normalization (dim=-1, not NVIDIA ref's dim=0) | arXiv:2410.01131 Section 2.3 |
| CausalRecurrenceLayer: log_a_init=3.0 (Zone E), 0.0 (Zone D) | Prevents 9.7 BPC plateau — see smoke_test.py |
| AMP dtype: float16 on T4 (sm<80), bfloat16 on Ampere+ | Auto-detected via `get_device_capability()` |

---

## Files

```
phase1/components/      Component implementations: attention, mla, diff_attention, mhc, mol_ffn, ngpt
phase1/model.py         Facade: ToyTransformer (17 CONFIGS)
phase1/train.py         Training loop — AMP, torch.compile, auto-resume
phase2/components/      Components: causal_recurrence, zone_e, zone_d, boundary_router
phase2/model.py         Facade: OuterModel (10 CONFIGS)
phase2/train.py         H-Net ratio loss, alpha warmup, boundary/chunk metrics
utils/data.py           TokenDataset, get_dataloader (WikiText-103)
utils/metrics.py        TrainLogger (JSONL), ParamCounter, ce_to_bpc
utils/smoke_test.py     Phase 2 pre-flight: 1000-step health check
utils/verify.py         Component-level staleness checker (9 named components)
utils/reporter.py       Auto-generates checkpoints/report.md
run_experiments.sh      Orchestrates all configs with study groups, gates, and auto-resume
references/components/  Authoritative specs for each architectural component
references/sources/     Reference papers and verbatim reference implementations
references/verification/ Phase B' verification reports and last_verified.json
```

---

## Methodology documentation

- `STUDY_DESIGN.md` — experimental protocol, 8 research questions, controlled comparisons
- `TRAINING_METHODOLOGY.md` — training procedure, AMP, optimizer, LR schedule
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
bash run_experiments.sh study_mole         # start with core study
bash run_experiments.sh phase1             # or run all Phase 1

# Sync results back:
git add checkpoints/
git commit -m "phase1 results"
git push
```
