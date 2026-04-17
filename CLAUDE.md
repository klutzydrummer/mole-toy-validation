# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Phase 1 toy validation of the MoLE architecture (Mixture-of-LoRAs Encoder) on WikiText-103 (BPE, vocab=4096). Phase 2 (HDC) wraps the mol inner network with Zone E and Zone D for content-aware compression.

**Phase 1 configs (27 total across 6 study groups, d=512 ~28M params):**

| Study | Configs | Research question |
|-------|---------|------------------|
| A — MoLE core | `baseline`, `baseline_wide`, `mol`, `mol_single`, `mhc`, `compose` | Q1: routing vs. capacity; Q2: routing vs. high-rank adapter |
| B — Attention | `mla`, `diff_attn`, `diff_attn_matched`, `diff_mla` | Q4: diff attn mechanism (controlled: `diff_attn_matched` vs `baseline`); Q5: MLA KV compression cost |
| C — mHC compose | `diff_mhc`, `mla_mhc`, `diff_mla_mhc` | Q6: mHC + attention variant compositions |
| D — nGPT | `ngpt`, `ngpt_mla`, `ngpt_diff_attn` | Q7: hyperspherical constraint benefit |
| E — Multi-sphere | `ngpt_mhc_a`, `ngpt_mhc_c` | Q8: multi-sphere vs. wrap-sublayer mHC |
| F — Compositions | `ngpt_diff_mla`, `diff_attn_matched_mhc`, `mol_mla`, `mol_diff_attn_matched`, `ngpt_mol`, `mol_diff_mla`, `ngpt_diff_attn_matched`, `ngpt_diff_mla_mhc`, `mol_diff_mla_mhc` | Composition gap fills: nGPT grid, controlled mHC, MoL×attention, direction-based routing, three-way |

Use `bash run_experiments.sh study_<name>` to run a single study group (e.g. `study_ngpt`).
See `STUDY_DESIGN.md` for research questions, controls, and execution order.

**Phase 1 scaling study (15 configs: 5 × d=256/768/1024):**
- Configs: `baseline`, `mla`, `diff_attn`, `diff_mla`, `mol` at d=256 (n_heads=4, ~6M), d=768 (n_heads=12, ~58M), d=1024 (n_heads=16, ~111M)
- 4 scale points spanning 20× size range enable curve-fitting, not just direction detection
- d_c/d=25% is held constant; ratio and absolute dimension are confounded — cannot isolate them
- d=1024 runs auto-receive `--use_8bit_adam --grad_checkpoint` (set in run_phase1_scale for d≥1024)
- Checkpoints use prefix `{cfg}_d{d}_seed42` to avoid collision with d=512 runs
- Run: `bash run_experiments.sh phase1_scaling`

---

## Operations — applies to every session

**IMPORTANT: When verifying any technical claim, always prefer authoritative online sources (official docs, published papers, reference implementations) over built-in knowledge. Use WebSearch/WebFetch to confirm before asserting facts. AI knowledge has a cutoff and can be wrong; published and empirically verified sources are ground truth.**

Training runs on cloud (Lightning.ai), never locally. Never run training or model commands on this machine.

Read every affected file completely before making any change. Fix root causes — never suppress errors or remove flags to make checks pass.

---

## Local pre-push checks — run these before every `git push`

**IMPORTANT: Python is not in the local shell PATH. Use `nix-shell` to enter the dev environment.**

```bash
nix-shell --run "python utils/shape_check.py"   # tensor shape validation (Phase 1 + 2)
nix-shell --run "ruff check ."                  # lint
```

`shell.nix` provides CPU-only torch + torchinfo + ruff. A single forward pass at full production dims (d=512, seq=256, batch=1) costs ~200 MB RAM — safe to run locally.

`shape_check.py` validates:
- Output shape contracts for all 37 configs (17 Phase 1 + 10 scaling + 10 Phase 2)
- No NaN/Inf in logits
- boundary_probs in [0, 1] for all Phase 2 configs

---

## Commands

```bash
bash run_experiments.sh                               # run all configs (Phase 1 + Phase 2)
bash run_experiments.sh phase1                        # Phase 1 only (17 configs, d=512)
bash run_experiments.sh phase1_scaling                # scaling study (5 configs × d=256,768,1024)
bash run_experiments.sh phase2                        # Phase 2 only (10 outer encoder configs)
bash run_experiments.sh baseline                      # single Phase 1 config
bash run_experiments.sh outer_crl                     # single Phase 2 config
python phase1/train.py --config mol --resume          # manual resume (Phase 1)
python phase2/train.py --config outer_crl --resume    # manual resume (Phase 2)
python utils/smoke_test.py                            # Phase 2 pre-flight health check
python utils/smoke_test.py --check-only               # check stored smoke test result
python utils/verify.py status                         # show verification status
python utils/verify.py update --result pass --report <path> <component>  # record verification
python utils/reporter.py                              # regenerate checkpoints/report.md
```

---

## Verification — YOU MUST follow this before any Phase 2 run

**IMPORTANT: The Phase 2 9.7 BPC plateau failure (March 2026) burned three 50k-step runs because a mandatory pre-flight check was skipped. The check was documented. It was not run. Never skip it.**

`run_experiments.sh` enforces two gates before any Phase 2 training:

1. **Smoke test** (`utils/smoke_test.py`): 1000-step outer_crl health check. Catches pipeline failures before wasting hours. Result stored in `checkpoints/smoke_test_result.json` and tied to a hash of the Phase 2 component files — re-runs automatically if model changes. **Hard exit on failure.**

2. **Staleness check** (`utils/verify.py check`): blocks training if any component file has changed since its last recorded verification. Tracks 8 named components (see `TRACKED_COMPONENTS` in `utils/verify.py`), not the facade model files.

Workflow when you change model code:
1. Change model code locally
2. Ask Claude Code: **"run full verification"** — triggers the Phase B' agent pipeline (`.claude/rules/run-full-verification.md`): agents cross-check every equation and code snippet in `references/components/` against the primary sources in `references/sources/`, and verify the implementation at cited line numbers. Reports written to `references/verification/reports/`.
3. Review reports — any FAIL verdict must be fixed before proceeding
4. `python utils/verify.py update --result pass --report <path> <component>` (component names: `attention_rope_norms`, `mla_attention`, `diff_attention`, `mhc`, `mol_ffn`, `causal_recurrence`, `zone_ed_pipeline`, `boundary_router`)
5. `git commit` (including updated `references/verification/last_verified.json` and reports)
6. Cloud: `run_experiments.sh` enforces both gates — smoke test + hash check

---

## Key invariants

### train.py ↔ run_experiments.sh argument contract
Every argument in `run_experiments.sh` must exist in `train.py`'s `argparse` block AND `train()` signature. **This was the source of the first crash.** Before adding any CLI arg, add it to both.

Verify: `grep -o '\-\-[a-z_]*' run_experiments.sh | sort` must be a subset of `grep "add_argument" phase1/train.py | grep -o '"\-\-[a-z_]*"' | tr -d '"' | sort`

### Component correctness — read the specs, do not rely on memory
Authoritative correctness criteria live in `references/components/`. Before touching any component, read its spec:

| Component | Spec | What to verify |
|-----------|------|----------------|
| mHC hyper-connections | `references/components/mhc.md` | H_res/H_pre/H_post init, go-mHC (Cayley transform + block Frobenius, not Sinkhorn), softmax not softplus |
| MoL routing | `references/components/mol_ffn.md` | sigmoid scores, unbiased weights, load-balance sign |
| CausalRecurrenceLayer | `references/components/causal_recurrence.md` | sqrt(1-a²) term, float32 sigmoid, parallel scan |
| BoundaryRouter | `references/components/boundary_router.md` | adjacent key k_{t-1}, p_0=1.0, top-M selection |
| Zone E / Zone D | `references/components/zone_ed_pipeline.md` | EMA, plug-back, gated residual, smoke test item 10 |
| MLA attention | `references/components/mla_attention.md` | KV shared latent (Eq. 9–11), Q separate latent (Eq. 12–13), no RMSNorm on latents (intentional deviation), d_c=d//4, d_c_q=d//2 |
| Diff Attn V2 / DiffMLA | `references/components/diff_attention.md` | Doubled Q heads, GQA pairing via repeat_interleave (not blocked split), sigmoid λ (not exp), no per-head RMSNorm, W_lambda bias=True |

---

## Architecture

```
phase1/components/  Component implementations: _shared, attention_rope_norms, mla_attention,
                    diff_attention, mhc, mol_ffn, transformer_block
phase1/model.py     Facade: re-exports components + ToyTransformer (CONFIGS, forward wiring)
phase1/train.py     Training loop: AMP, torch.compile, resume, eval
phase2/components/  Component implementations: causal_recurrence, zone_e, zone_d, boundary_router
phase2/model.py     Facade: re-exports components + OuterModel (CONFIGS, forward wiring, 10 configs)
phase2/train.py     H-Net ratio loss (Eq. 10), alpha warmup, boundary/chunk metrics
utils/data.py       Data loader: TokenDataset, get_dataloader (WikiText-103 default)
utils/metrics.py    TrainLogger (JSONL), ParamCounter, ce_to_bpc
utils/smoke_test.py Phase 2 pre-flight: 1000-step health check (encoder diversity, loss reduction)
utils/verify.py     Component-level staleness checker (schema v2: 8 named components)
utils/reporter.py   Auto-generates checkpoints/report.md every 30s during training
run_experiments.sh  Run all configs; enforces smoke test + verify before Phase 2
references/components/  Authoritative specs for each component with verification checklists
references/sources/     Reference papers and verbatim reference implementations
references/verification/ Verification reports and last_verified.json
```

Base config: `d=512, n_layers=8, n_heads=8, mol_rank=8, SwiGLU, RoPE, RMSNorm, weight-tying`
Dataset: WikiText-103-raw, BPE tokenizer (vocab=4096, sentencepiece), seq_len=256
`checkpoints/report.md` — primary training artifact, regenerate with `python utils/reporter.py`

---

## Phase 1 Results

All Phase 1 runs are being re-run from scratch with the current codebase (go-mHC).
Prior seed42 results are superseded. See `checkpoints/report.md` for live training state.

**`diff_attn` capacity note:** Doubled Q heads add ~25% more attention parameters vs. `baseline` (+2.1M). Use `diff_attn_matched` (d_ff=1240, ~27.86M) for the controlled Q4 comparison. `diff_attn` reports the combined mechanism + capacity effect and is an upper bound complement.

---

## Phase 2 Status and Key Decisions

**Goal:** Isolate and study the outer encoder — which architecture produces the best concept tokens? Composition with an inner network moves to Phase 3.

**Run outer_crl first, always.** It uses the cosine_rule router (no learned params) and validates the pipeline before adding learned routing. If it doesn't improve on a strided baseline, the pipeline is broken.

**Run order:** `outer_crl` → `outer_crl_learned` → `outer_crl_full` / `outer_crl_full_learned` → transformer variants → ablations

Phase 2 training: 50k steps, alpha=0.03 (H-Net ratio loss), alpha warmup over first 2k steps.

**10 configs (OuterModel.CONFIGS):**

| Config | Encoder | Router | Notes |
|--------|---------|--------|-------|
| `outer_crl` | CRL (282K, bottlenecked) | cosine_rule | Baseline — run first |
| `outer_crl_learned` | CRL | learned_e2e | Learned routing on bottlenecked CRL |
| `outer_crl_full` | CRL (1.57M, full-width) | cosine_rule | No bottleneck confound |
| `outer_crl_full_learned` | CRL full | learned_e2e | Full CRL + learned routing |
| `outer_transformer` | Transformer (12.8M) | learned_e2e | Standard attention |
| `outer_diff_attn` | Diff Attn V2 (13.9M) | learned_e2e | Noise-cancelling attention |
| `outer_mla` | MLA (11.5M) | learned_e2e | KV-compressed attention |
| `outer_strided` | Identity | fixed_stride | Hard lower bound (no encoder) |
| `outer_crl_learned_noste` | CRL | learned_e2e | STE ablation (use_ste=False) |
| `outer_crl_fixed_stride` | CRL (282K) | fixed_stride | Clean routing ablation: same encoder as outer_crl, fixed routing. Isolates routing quality from encoder quality for Q3. |

**Encoder capacity note:** These are architecture-vs-architecture comparisons at d=512, NOT param-matched. CRL (bottlenecked) is 45× smaller than transformer variants. `outer_crl_full` (1.57M) removes the bottleneck confound. Differences in BPC may reflect capacity, not architecture quality.

**Per-step metrics logged:** `compression_ratio`, `loss_ratio`, `loss_ratio_excess` (0.0=converged), `boundary_entropy`, `chunk_len_mean`, `chunk_len_var`
**Per-eval metrics logged:** `boundary_bpc`, `midchunk_bpc`, `residual_proj_norm`

Note: `boundary_bpc > midchunk_bpc` is geometrically expected (fresh concept token has less smoothed context). Diagnostic value is in the trend over training and cross-config comparison, not the absolute sign.

---

## Compact instructions

When compacting, preserve:
- Current phase, active task, and any open decisions
- Verification status of any component under discussion; any FAIL verdicts
- Any active mHC grad-norm observations from current runs
- Any in-progress code changes not yet committed

Discard:
- Raw grep/search output and full file reads of unchanged reference material
- Completed task history with no bearing on current work
