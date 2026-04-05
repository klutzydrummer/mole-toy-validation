# Nix shell for local development tools — shape validation, linting, smoke test.
# Does NOT include the training stack (CUDA torch, litlogger).
# Those live in requirements.txt and run on Lightning.ai only.
#
# Usage:
#   nix-shell                                             # enter shell
#   nix-shell --run "python utils/shape_check.py"        # shape validation
#   nix-shell --run "ruff check ."                       # linter
#   nix-shell --run "python utils/smoke_test.py --batch_size 4"  # pre-flight smoke test (CPU)

{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    # CPU-only PyTorch — sufficient for shape check and smoke test.
    # No CUDA. Smoke test runs ~5-10 min on CPU at batch_size=4.
    (pkgs.python3.withPackages (ps: with ps; [
      torch
      torchinfo
      numpy
      sentencepiece  # BPE tokenizer for WikiText-103 data prep
      datasets       # HuggingFace datasets — downloads WikiText-103 on first run
    ]))

    # Linter — same binary used in CI and by CLAUDE.md
    pkgs.ruff
  ];

  shellHook = ''
    echo "dev shell ready"
    echo "  python utils/shape_check.py                          — validate tensor shapes (Phase 1 + Phase 2)"
    echo "  python utils/smoke_test.py --batch_size 4            — Phase 2 pre-flight smoke test (CPU, ~5-10 min)"
    echo "  ruff check .                                         — lint"
  '';
}
