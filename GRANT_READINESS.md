# Grant Readiness — Google TRC Application

**Target:** Google TPU Research Cloud (sites.research.google/trc)
**Access type:** 30-day free TPU quota, renewable; no institutional affiliation required
**Commitment required:** Public output (paper, blog, or open-source code); feedback to Google
**Status of this document:** Pre-application checklist and self-audit guide

---

## 1. What TRC Evaluates

TRC does not publish a rubric, but stated criteria from their FAQ and application form map to
four axes:

| Axis | What they look for | Where we stand |
|------|--------------------|----------------|
| **Research quality** | Clear hypothesis, controlled comparisons, falsifiable claims | STUDY_DESIGN.md, PHASE3_STUDY_DESIGN.md — strong |
| **Feasibility** | Realistic scope for 30-day TPU block | Single A100-equivalent run; 120–150 hrs realistic — needs estimate fix |
| **Impact / novelty** | Contribution beyond reproducing existing results | Systematic ablation of mHC, MoL, HDC at 1B; KromHC at toy scale is new |
| **Public output** | Paper, blog post, open-source code | Code is open; blog or preprint required as deliverable |

Independent researchers without affiliation are accepted. The differentiator is evidence that
you've already done rigorous, systematic work — not just a pitch for future work.

**Our strongest asset:** This project has Phase 1 completed (single-seed, multi-seed reruns
pending) and Phase 2 designed and gated, with documented methodology, controlled ablations,
and verified implementations. That is unusual for an independent researcher and substantially
raises the application's credibility.

---

## 2. Eligibility — Hard Requirements

| Requirement | Status | Action needed |
|-------------|--------|---------------|
| Project must be ML research | ✅ | None |
| Research must be shareable publicly (paper, blog, OSS) | ✅ (code is open) | Commit to blog post or preprint as deliverable |
| Must comply with Google AI Principles | ✅ (NLP / LM research, no harmful applications) | None |
| Must provide feedback on TPU platform | ✅ (straightforward) | None |
| Approved country (US) | ✅ | None |
| PyTorch-XLA or JAX — TPUs don't run vanilla PyTorch CUDA | ❌ | **Porting cost — see Section 6** |

The PyTorch-XLA requirement is the only hard technical blocker. It is solvable but non-trivial.

---

## 3. What Makes a Serious, Grant-Worthy Experiment

### 3.1 Scientific rigor

A grant-worthy experiment answers a specific, falsifiable question with controlled comparisons.
Each question must have a clean null hypothesis.

| Phase 3 question | Null hypothesis | What "serious" looks like |
|-----------------|-----------------|--------------------------|
| Q4: Does mHC work at 1B with exact DS constraint? | KromHC/mHC-lite ≤ baseline_1b in val BPC | Both trained at equal steps/tokens, matched parameter budgets documented |
| Q5: Does MoL routing scale? | mol_1b ≤ baseline_wide_1b | Parameter-matched baseline (see STUDY_DESIGN.md §2) |
| Q6: Does mHC + MoL compose? | compose_1b ≤ max(mhc_lite_1b, mol_1b) | Not run unless Q4 AND Q5 pass |
| Q7: Does HDC improve at Chinchilla scale? | hdc_1b ≤ mol_1b | Not run unless A1 gate passes in Phase 2 |

The conditional structure (Q6 and Q7 only run if preconditions pass) is a mark of
scientific seriousness. It prevents running experiments just to generate results.

### 3.2 Reproducibility

For external reviewers and for publication:

- [ ] All hyperparameters documented in a single config file, not scattered across scripts
- [ ] Random seeds logged and fixed for baseline runs
- [ ] All results traceable to a specific git commit hash
- [ ] `requirements.txt` pins exact dependency versions
- [ ] Dataset download and preprocessing fully automated (no manual steps)
- [ ] Training script takes all config from CLI args (no hardcoded values)

### 3.3 Compute budget honesty

The application will ask how much compute you need. **Do not underestimate.** An honest,
justified estimate signals that you understand your project:

| Phase 3 config | Estimated A100 hours | Notes |
|---------------|---------------------|-------|
| baseline_1b | ~120–150 hrs | 500k steps × ~5 tokens/step; central estimate |
| mhc_lite_1b | ~130–160 hrs | KromHC overhead ~5–10% |
| mol_1b | ~130–160 hrs | MoL routing overhead ~5–10% |
| compose_1b (conditional) | ~140–170 hrs | Only if Q4+Q5 pass |
| hdc_1b (conditional) | ~150–180 hrs | Dual-loss overhead |
| **Minimum (Q4+Q5 only)** | **~500–640 hrs** | 3–4 runs |
| **Full study** | **~700–900 hrs** | 5 runs |

**Note:** The PHASE3_STUDY_DESIGN.md currently states ~28–30 hrs/run. This is incorrect.
The correct estimate is ~120–150 hrs/run based on OLMo-1B training benchmarks.
**This must be corrected before submitting any application.**

### 3.4 Ethical considerations

TRC requires compliance with Google AI Principles. For this project:

- **No harmful applications.** Language modeling research on FineWeb (English web text) for
  architecture validation. No use cases involving surveillance, manipulation, or misuse.
- **Dataset provenance documented.**
  - WikiText-103: derived from Wikipedia, licensed CC-BY-SA 3.0. Copyleft applies to
    the dataset itself; model weights trained on CC-BY-SA data are generally considered
    a derivative work in a different form and are not subject to SA requirements under
    most legal interpretations, but this remains an unsettled area. Training and research
    use is clearly permitted. Publishing weights should include an attribution note.
  - FineWeb (Phase 3): Common Crawl with quality filters, licensed CC-BY 4.0. Permissive.
- **No personally identifiable information.** WikiText-103 and FineWeb contain no PII.
- **Open weights and code.** All trained models will be published under a permissive license.
- **No deceptive framing.** Results reported as-is, including negative results (mHC Phase 1
  failure is documented openly in PHASE3_STUDY_DESIGN.md).

---

## 4. Code Quality Audit — What an Outside Reviewer Expects

Before applying, the codebase should be auditable by someone who has never seen it.
Walk through each item below and mark it complete.

### 4.1 Correctness — implementation matches claimed design

- [ ] **Full verification pass (Phase B')** — run the `references/verification/` pipeline for all
  6 components. All reports must be PASS before claiming the implementation is correct.
  Command: ask Claude Code "run full verification"
- [ ] **README.md is accurate** — currently references enwik8 and ~5M params; project is now
  WikiText-103 and ~28M params. README needs a rewrite to reflect current state.
- [ ] **PHASE3_STUDY_DESIGN.md compute estimates corrected** — 28–30 hrs/run → 120–150 hrs/run
- [ ] **Phase 1 final results documented** — compose 3.5316, mol 3.5702, mhc 3.5736, baseline
  3.5875 should be in a committed, stable results table (currently only in CLAUDE.md)

### 4.2 Transparency — negative results are visible

A grant reviewer should be able to see that you report failures honestly:

- [ ] mHC Phase 1 failure (3.5643 vs 3.5582 baseline) is documented with root cause analysis
  ✅ — already in `references/components/mhc.md` under "Scale limitations"
- [ ] Phase 2 9.7 BPC plateau failure documented with post-mortem
  ✅ — in `utils/smoke_test.py` comments and `CLAUDE.md`
- [ ] float16/bfloat16 detection bug (T4 falling back silently) documented and fixed
  ✅ — fixed in `phase1/train.py` and `phase2/train.py`

### 4.3 Reproducibility infrastructure

- [ ] `run_experiments.sh` produces fully reproducible results from a clean clone
- [ ] All random seeds are logged in checkpoints (verify `ckpt["seed"]` or equivalent)
- [ ] `requirements.txt` or `pyproject.toml` pins all dependency versions with hashes
- [ ] No hardcoded paths — all paths relative or configurable via CLI

### 4.4 Code legibility

An outside reviewer reading `phase1/model.py` and `phase2/model.py` should be able to
understand the implementation without reading every comment:

- [ ] Class and method names match paper terminology (e.g., `HyperConnection`, `MoLFFN`,
  `CausalRecurrenceLayer`, `BoundaryRouter`)
- [ ] Each non-obvious formula has a comment citing the source (equation number + arXiv ID)
- [ ] `references/components/` specs are cited in code comments where relevant
- [ ] No dead code (unused configs, commented-out experiments, TODO stubs)

### 4.5 Documentation completeness

- [ ] **README.md** — accurate description of all phases, configs, and results
- [ ] **STUDY_DESIGN.md** — covers Phase 1 and 2 questions; Phase 3 in PHASE3_STUDY_DESIGN.md
- [ ] **TRAINING_METHODOLOGY.md** — training procedure, AMP, optimizer, LR schedule
- [ ] **references/components/** — authoritative specs for every architectural component
- [ ] **references/verification/reports/** — verification reports for all components

---

## 5. Application Narrative — Draft Outline

The TRC application asks for a project description. The narrative should cover:

### 5.1 The research question (2–3 sentences)
> We are studying whether the efficiency gains claimed for mHC hyper-connections and
> Mixture-of-LoRAs FFN routing hold at 1B parameters, given that both techniques were
> validated only at ≥1B scale in their original papers. Our Phase 1 results (28M params)
> show MoL routing improves BPC while mHC underperforms — consistent with scale-dependent
> behavior. Phase 3 tests the null hypothesis that these results invert at proper scale
> using exact doubly-stochastic implementations (KromHC / mHC-lite).

### 5.2 Why this is novel (1–2 sentences)
> No published ablation exists comparing mHC-lite (exact Birkhoff-von Neumann) against
> standard mHC and baseline at 1B parameters. The KromHC paper (arXiv:2601.21579) validates
> at 12 layers; we extend to 24 layers with a full architectural composition study.

### 5.3 Prior work (links to code and methodology docs)
- GitHub repo URL (make public before applying)
- Link to Phase 1 / Phase 2 results
- Link to PHASE3_STUDY_DESIGN.md

### 5.4 Compute justification
> 5 runs × ~150 A100-equivalent hours = ~750 hours. TPU v3-8 runs at ~3× A100 throughput,
> so ~250 TPU v3-8 hours requested. One 30-day block (~720 TPU v3-8 hours) covers the
> full study with margin for debugging.

### 5.5 Public output commitment
> All code, trained model weights, and a technical blog post / arXiv preprint will be
> published under Apache 2.0 / CC-BY license within 90 days of completing training.

---

## 6. PyTorch-XLA Port — the Technical Blocker

TPUs require XLA compilation. Standard CUDA PyTorch ops do not run on TPUs.

### What requires changes

| Component | XLA compatibility | Notes |
|-----------|-------------------|-------|
| Core transformer (attention, RoPE, RMSNorm) | ✅ straightforward | Standard ops; `torch.compile` with XLA backend |
| `torch.cuda.amp.autocast` | ❌ needs change | Replace with `torch.amp.autocast("xla")` |
| `torch.cuda.GradScaler` | ❌ not needed on TPU | bfloat16 is native on TPU v3+; remove scaler |
| `einops.einsum` / `torch.einsum` | ✅ | XLA supports these |
| Sinkhorn log-space (custom loops) | ⚠️ test required | Dynamic control flow may require `torch.cond` or unrolling |
| `torch.topk` in MoL routing | ✅ | Standard XLA op |
| BoundaryRouter (variable-length segments) | ⚠️ test required | Dynamic shapes are problematic on TPU; may need padding |
| CausalRecurrenceLayer parallel scan | ⚠️ test required | Associative scan supported in JAX natively; PyTorch-XLA requires care |
| DataLoader with `pin_memory=True` | ❌ needs change | TPU uses `xm.MpDeviceLoader` instead |

### Port strategy

1. Start with `phase1/model.py` (baseline config only) — no dynamic shapes, no recurrence
2. Validate loss curve matches T4 baseline within ±0.001 BPC
3. Port MoL routing — validate `topk` and routing table are XLA-static
4. Port mHC / Sinkhorn — validate doubly stochastic output numerically on XLA
5. Port `phase2/model.py` last — BoundaryRouter dynamic shapes are the hardest part

**Estimated port effort:** 2–4 days for a developer familiar with both codebases.
Do this work BEFORE submitting the application — the ability to say "we have successfully
run a smoke test on TPU" will substantially strengthen the application.

---

## 7. Pre-Application Checklist

Complete all items before submitting:

**Documentation**
- [ ] Fix README.md (stale: enwik8, ~5M params, missing Phase 2)
- [ ] Fix PHASE3_STUDY_DESIGN.md compute estimate (28–30 hrs → 120–150 hrs/run)
- [ ] Add a stable RESULTS.md with final Phase 1 and Phase 2 outcomes
- [ ] Confirm public GitHub repo with clean commit history

**Code quality**
- [ ] Run full Phase B' verification (`references/verification/reports/` — all PASS)
- [ ] Audit requirements.txt — all versions pinned
- [ ] Confirm all random seeds logged in checkpoint files
- [ ] Remove any dead code or stale TODO comments

**Scientific integrity**
- [ ] All negative results (mHC Phase 1, HDC plateau) documented in public-facing docs
- [ ] Conditional experiment structure enforced (Q6/Q7 depend on Q4/Q5 passing)
- [ ] Dataset provenance documented (FineWeb CC-BY, WikiText-103 CC-BY-SA)

**Technical**
- [ ] Phase1 baseline smoke test on PyTorch-XLA (proves TPU viability)
- [ ] GradScaler logic removed from XLA version (bfloat16 native on TPU v3+)

**Application**
- [ ] Compute estimate calculated and justified (see Section 5.4)
- [ ] Public output commitment written (blog post or arXiv preprint)
- [ ] Apply at: https://sites.research.google/trc/

---

## 8. Alternative Grant Programs (fallback order)

If TRC is rejected or PyTorch-XLA port is too costly:

| Program | Barrier | Next action |
|---------|---------|-------------|
| **fal.ai** | Lowest — email only | grants@fal.ai — send project summary + GitHub link |
| **CloudRift** | Low — rolling application | $100–$1,000 GPU credits, no affiliation required |
| **LTFF** | Medium — needs safety framing | Add a "model interpretability / architecture transparency" angle |
| **EleutherAI** | Medium — community contribution first | Contribute to an open EleutherAI project to build standing |
