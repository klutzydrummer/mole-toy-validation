# Grant Readiness — Fix Tracker

Tracks every issue identified by the adversarial critic review (2026-03-23).
Source: critic verdict "FUND WITH CONDITIONS" based on review of README, STUDY_DESIGN,
PHASE3_STUDY_DESIGN, GRANT_READINESS, TRAINING_METHODOLOGY, model.py, train.py,
references/components/mhc.md, references/components/mol_ffn.md, and Phase B' reports.

Status legend: [ ] pending  [~] in progress  [x] complete

---

## Blockers — must fix before TRC application

- [x] **#1 — Seed control missing from training code** *(fixed 2026-03-23)*
  Add `--seed` CLI arg to `phase1/train.py` and `phase2/train.py`.
  Call `torch.manual_seed`, `torch.cuda.manual_seed_all`, `random.seed`, `numpy.random.seed`
  before model construction. Save seed in every checkpoint. Load on resume.
  Without this, no result in this repo is strictly reproducible.

- [x] **#2 — mol_rank default mismatch** *(fixed 2026-03-23)*
  `phase1/model.py` TransformerBlock and ToyTransformer default `mol_rank=4`.
  All docs (CLAUDE.md, README, STUDY_DESIGN) lock `mol_rank=8`.
  Fix: change model.py defaults to `mol_rank=8`.

- [ ] **#7 — requirements.txt not pinned**
  Uses `>=` floors; `litlogger` has no version at all.
  Fix: run `pip freeze` on Lightning.ai and replace with `==` pinned versions.
  **Requires user action on remote environment.**

- [x] **#8 — No committed results file** *(fixed 2026-03-23: RESULTS.md created)*
  Results live only in CLAUDE.md (private AI context). README claims results but
  checkpoints/ is empty.
  Fix: create RESULTS.md with Phase 1 final numbers, clearly marked as single-seed
  preliminary results pending multi-seed reruns.

- [ ] **#10 — PyTorch-XLA port not started**
  TPUs require XLA. Cannot run TRC hardware without it.
  Fix: port phase1/train.py baseline config to XLA; run 1000-step smoke test on TPU;
  verify loss matches T4. Do this before submitting application.
  **This is the longest-lead item — start early.**

---

## Significant concerns — fix before applying

- [x] **#3 — Stale float16 factual error in TRAINING_METHODOLOGY.md §6** *(fixed 2026-03-23)*
  Says "sigmoid(7.5) rounds to 1.0 in float16 (max < 1 is 0.9990)".
  Correct: float16 max < 1.0 is 0.99951; sigmoid(7.5) = 0.99944 → 0.99951, NOT 1.0.
  This was corrected in causal_recurrence.md and CLAUDE.md but not propagated here.

- [x] **#6 — README results table misleads on mol vs baseline comparison** *(fixed 2026-03-23: added disclaimer note)*
  Table presents mol 3.5702 vs baseline 3.5875 as if it's a clean win.
  STUDY_DESIGN.md §1 explicitly says this comparison is uncontrolled (different param counts).
  Fix: add a note that these are preliminary single-seed uncontrolled comparisons;
  the controlled comparison is mol vs baseline_wide (Q1).

---

## Minor issues — fix for completeness

- [x] **#4 — mhc.md prose says softmax(dim=0), code says dim=-1** *(verified 2026-03-23 — not an error: dim=0 and dim=-1 are equivalent on 1D stream_collapse_logits; Deviation 1 prose at line 168 already correctly says dim=-1)*
  Phase B' WARN. Fix: change prose to `dim=-1` consistently.

- [x] **#5 — mol_ffn.md checklist item 11 wrong method name** *(fixed 2026-03-23: get_load_stats → get_mol_stats)*
  Says `model.get_load_stats()`, should be `model.get_mol_stats()`.
  Phase B' WARN. Fix: correct the method name in the checklist.

- [x] **#9 — WikiText-103 license not documented** *(fixed 2026-03-23: CC-BY-SA 3.0 noted in GRANT_READINESS.md §3.4 and README.md)*
  CC-BY-SA (copyleft, derived from Wikipedia). Matters for publishing weights.
  Fix: add license info to GRANT_READINESS.md §3.4 and README.md.

---

---

## Round 2 critic findings (2026-03-23)

- [x] **N7 — PHASE3_STUDY_DESIGN.md had wrong mHC numbers (3.5643/3.5582)** *(fixed: corrected to 3.5736/3.5875)*
- [x] **N6 — LR sweep compute estimate ~12 hrs was ~2× too low** *(fixed: corrected to ~24 hrs; totals updated)*
- [x] **N5 — phase2/train.py total_steps default 100k ≠ run_experiments.sh 50k** *(fixed: default changed to 50000)*
- [x] **N3 — Multi-seed runs overwrite each other's checkpoints** *(fixed: checkpoint names now include seed, e.g. baseline_seed42_latest.pt; run_experiments.sh passes --seed and checks per-seed checkpoint)*
- [x] **N9 — Best-val checkpoint missing seed** *(fixed: seed now saved in both best and latest checkpoints)*
- [x] **N2 — MoLFFN class default rank=4** *(fixed: changed to rank=8)*
- [x] **N1 — model.py docstring says "rank 4" for mol config** *(fixed: updated to "rank 8")*
- [x] **C6 — GRANT_READINESS.md overstated Phase 2 as "completed or in progress"** *(fixed: corrected to "Phase 1 completed, Phase 2 designed and gated")*
- [ ] **N10 (C7) — run_experiments.sh sudo shutdown fires on any error** *(known operational risk; not a scientific issue — left for user decision)*

---

## Out of scope for code fixes (execution-dependent)

- Multi-seed Phase 1 reruns (baseline ×3, mol ×3, baseline_wide ×3, mol_single ×3)
  Cannot be done locally. Required before primary claims (Q1, Q2) can be made.
  Depends on #1 (seed control) being merged first.

---

## Completion gate

All blockers (#1, #2, #7, #8, #10) must be marked [x] before submitting TRC application.
Significant concerns (#3, #6) should be [x] before applying.
Minor issues (#4, #5, #9) should be [x] before making repo public-facing.
