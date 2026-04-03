# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Phase 1 toy validation of the MoLE architecture (Mixture-of-LoRAs Encoder) at ~5M params on WikiText-103 (BPE, vocab=4096). Phase 2 (HDC) wraps the mol inner network with Zone E and Zone D for content-aware compression.

**Phase 1 configs (9 total):**
- Core: `baseline`, `mhc`, `mol`, `compose`
- Attention variants: `mla` (MLA KV compression, arXiv:2405.04434), `diff_attn` (Diff Attn V2, Jan 2026), `diff_mla` (novel Diff V2 + MLA composition)
- Ablations: `baseline_wide` (widened FFN, param-matched to mol for Q1 fair comparison), `mol_single` (capacity-matched single LoRA, no routing, for Q2)

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
- Output shape contracts for all 15 non-upcycle configs (9 Phase 1 + 6 Phase 2)
- No NaN/Inf in logits
- boundary_probs in [0, 1] for all Phase 2 configs

---

## Commands

```bash
bash run_experiments.sh                               # run all configs (Phase 1 + Phase 2)
bash run_experiments.sh phase1                        # Phase 1 only (9 configs)
bash run_experiments.sh phase2                        # Phase 2 only (8 HDC configs)
bash run_experiments.sh baseline                      # single Phase 1 config
bash run_experiments.sh hdc_gate                      # single Phase 2 config
python phase1/train.py --config mol --resume          # manual resume (Phase 1)
python phase2/train.py --config hdc_gate --resume     # manual resume (Phase 2)
python utils/smoke_test.py                            # Phase 2 pre-flight health check
python utils/smoke_test.py --check-only               # check stored smoke test result
python utils/verify.py status                         # show verification status
python utils/verify.py update --result pass --report <path>  # record verification
python utils/reporter.py                              # regenerate checkpoints/report.md
```

---

## Verification — YOU MUST follow this before any Phase 2 run

**IMPORTANT: The Phase 2 9.7 BPC plateau failure (March 2026) burned three 50k-step runs because a mandatory pre-flight check was skipped. The check was documented. It was not run. Never skip it.**

`run_experiments.sh` enforces two gates before any Phase 2 training:

1. **Smoke test** (`utils/smoke_test.py`): 1000-step hdc_rulebased health check. Catches pipeline failures before wasting hours. Result stored in `checkpoints/smoke_test_result.json` and tied to `phase2/model.py` hash — re-runs automatically if model changes. **Hard exit on failure.**

2. **Staleness check** (`utils/verify.py check`): blocks training if `phase1/model.py` or `phase2/model.py` have changed since the last recorded verification.

Workflow when you change model code:
1. Change model code locally
2. Ask Claude Code: **"run full verification"** — triggers the Phase B' agent pipeline (`.claude/rules/run-full-verification.md`): agents cross-check every equation and code snippet in `references/components/` against the primary sources in `references/sources/`, and verify the implementation at cited line numbers. Reports written to `references/verification/reports/`.
3. Review reports — any FAIL verdict must be fixed before proceeding
4. `python utils/verify.py update --result pass --report <path>`
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
| mHC hyper-connections | `references/components/mhc.md` | H_res/H_pre/H_post init, sinkhorn, softmax not softplus |
| MoL routing | `references/components/mol_ffn.md` | sigmoid scores, unbiased weights, load-balance sign |
| CausalRecurrenceLayer | `references/components/causal_recurrence.md` | sqrt(1-a²) term, float32 sigmoid, parallel scan |
| BoundaryRouter | `references/components/boundary_router.md` | adjacent key k_{t-1}, p_0=1.0, top-M selection |
| Zone E / Zone D | `references/components/zone_ed_pipeline.md` | EMA, plug-back, gated residual, smoke test item 10 |
| MLA attention | `references/components/mla_attention.md` | KV shared latent (Eq. 9–11), Q separate latent (Eq. 12–13), no RMSNorm on latents (intentional deviation), d_c=d//4, d_c_q=d//2 |
| Diff Attn V2 / DiffMLA | `references/components/diff_attention.md` | Doubled Q heads, GQA pairing via repeat_interleave (not blocked split), sigmoid λ (not exp), no per-head RMSNorm, W_lambda bias=True |

---

## Architecture

```
phase1/model.py     ToyTransformer: baseline / baseline_wide / mhc / mol / mol_single / compose / mla / diff_attn / diff_mla
phase1/train.py     Training loop: AMP, torch.compile, resume, eval
phase2/model.py     HDCModel: ZoneE + InnerTransformer + ZoneD
phase2/train.py     Dual-loss training loop (loss_ntp + λ_comp * loss_comp)
utils/data.py       Data loader: TokenDataset, get_dataloader (WikiText-103 default)
utils/metrics.py    TrainLogger (JSONL), ParamCounter, ce_to_bpc
utils/smoke_test.py Phase 2 pre-flight: 1000-step health check (encoder diversity, loss reduction)
utils/verify.py     Verification staleness checker (hash-based, runs on cloud)
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

### Original runs (100k steps, no seed suffix)

| Config | Best val BPC | Notes |
|--------|-------------|-------|
| compose | 3.5316 | Best overall — but mHC optimization issue unresolved |
| mol | 3.5702 | Inner network for all Phase 2 configs |
| mhc | 3.5736 | Rising grad norm (0.54→0.70); diagnose before composing |
| baseline | 3.5875 | Reference |

### Seed42 reruns + new attention configs (as of 2026-04-03)

| Config | Best val BPC | Steps | Notes |
|--------|-------------|-------|-------|
| compose_seed42 | 3.5416 | 100k complete | mHC + MoL composition; best seed42 result |
| baseline_wide_seed42 | 3.5389 | 100k complete | d_ff=1600 (param-matched to mol); Q1 ablation |
| mhc_seed42 | 3.5482 | 100k complete | Grad norm still rising (0.80→1.37); issue unresolved |
| mol_seed42 | 3.5497 | 100k complete | Confirms prior 3.5702 run; inner network for Phase 2 |
| mol_single_seed42 | 3.5558 | 100k complete | Q2: routing beats single-LoRA by ~0.006 BPC |
| baseline_seed42 | 3.5605 | 100k complete | Seed42 reproducibility run |
| mla_seed42 | 3.7175 (step 64999) | ~67k in progress | MLA KV compression; declining fast (−0.068 per 10k steps); LR still high |
| diff_attn_seed42 | — | not yet started | Differential Attention V2 (Jan 2026) |
| diff_mla_seed42 | — | not yet started | Diff V2 + MLA composition (novel) |

Do not compose mHC+MoL+HDC until mHC's grad norm issue is diagnosed. The rising grad norm is continuous — observed 0.80→1.37 over 100k steps with no stabilization. Diagnostic: `python phase1/train.py --config mhc --max_lr 1.5e-4 --total_steps 25000`.

---

## Phase 2 Status and Key Decisions

A1 gate ablation: does content-aware compression at R=4 improve BPC over mol alone (3.5702)?

**IMPORTANT: Run hdc_rulebased first, always. It validates the pipeline before learned routing. If it doesn't improve on mol, the pipeline is broken — fix before proceeding.**

Run order: `hdc_rulebased` → `hdc_gate` → `hdc_stride` → `hdc_r2/r8` (after A1 passes) → upcycle configs

Phase 2 training: 50k steps. Upcycle configs require `checkpoints/mol_best.pt`.

A1 success: `hdc_gate` val BPC < 3.5702, compression ratio stable near 0.25, boundary_entropy decreasing.

Per-step metrics logged: `compression_ratio`, `loss_comp`, `boundary_entropy`
Per-eval metrics logged: `boundary_bpc`, `mid_chunk_bpc`
