# References

This directory is the authoritative source library for the MoLE toy-validation project.
It exists to ground every architectural decision and implementation in primary literature,
preventing hallucination from propagating into model code.

## How agents must use this directory

**Never modify `sources/`.** It contains raw fetched material — paper content and author
code. It is append-only. New research runs add new files; nothing is overwritten.

**`components/` is the working surface.** Each file covers one model component and
contains everything needed to verify that component in one read: the authoritative
equations (cited to paper + equation number), reference code from the authors, a pointer
to our implementation, known intentional deviations, and a verification checklist.

**Before making any change to model code**, read the relevant `components/*.md` file.
If the component file does not exist or seems incomplete, run the research pipeline
(Phase A → B → B') before proceeding.

## Directory layout

```
sources/
  papers/          Raw paper content fetched from arXiv (markdown/text, one file per paper)
  code/            Reference implementations fetched from author repositories
components/        Per-component authoritative reference (schema below)
verification/
  checklist.md     Master checklist across all components
  last_verified.json  {file: {git_hash, verified_at, result, report}}
  reports/         Timestamped verification reports (output of Phase C)
```

## Component file schema

Every `components/*.md` file must contain these sections in order:

1. **Component** — name and one-line description
2. **Sources** — list of papers and code files in `sources/` that this is derived from
3. **Authoritative equations** — verbatim from paper, with paper name + equation number
4. **Reference implementation** — exact code snippet from author repo, with attribution
5. **Our implementation** — `file:line` pointer, description of any intentional deviations
6. **Verification checklist** — numbered list of concrete things to check

## Verification workflow (Phase C — runs locally, never in cloud)

`utils/verify.py --check-staleness` compares git hashes of model files against
`verification/last_verified.json`. If stale, run `utils/verify.py --full` to re-verify.

`run_experiments.sh` calls `--check-staleness` before any training. If stale, it
exits with an error — push a fresh verification before running on the cloud instance.

## Research pipeline (Phases A → B → B')

Run locally on demand, including when new papers are published:

- **Phase A** (research agents): fetch papers and code into `sources/`
- **Phase B** (extraction agents): derive `components/*.md` from `sources/`
- **Phase B'** (validation agents): every claim in `components/*.md` must be traceable
  to a verbatim quote or code snippet in `sources/`. Flags anything that is not.

Phase B' output must be reviewed and approved before `last_verified.json` is updated.

## Components covered

| Component | Papers | Code |
|-----------|--------|------|
| `causal_recurrence` | Griffin (arXiv:2402.19427) | Google DeepMind / authors |
| `boundary_router` | H-Net (arXiv:2507.07955), DLCM | H-Net authors |
| `zone_ed_pipeline` | H-Net (arXiv:2507.07955) | H-Net authors |
| `mol_ffn` | DeepSeek-V3 (arXiv:2412.19437) | deepseek-ai/DeepSeek-V3 |
| `mhc` | mHC (arXiv:2512.24880) | tokenbender/mHC |
| `attention_rope_norms` | RoPE (2104.09864), RMSNorm (1910.07467), SwiGLU (2002.05202) | various |
