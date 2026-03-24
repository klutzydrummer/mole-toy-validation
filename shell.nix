# Nix shell for local development tools — shape validation and linting.
# Does NOT include the training stack (CUDA torch, sentencepiece, litlogger).
# Those live in requirements.txt and run on Lightning.ai only.
#
# Usage:
#   nix-shell                                    # enter shell
#   nix-shell --run "python utils/shape_check.py"  # run shape check directly
#   nix-shell --run "ruff check ."               # run linter

{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    # CPU-only PyTorch — sufficient for a single forward pass shape check.
    # No CUDA, no training. Fast to install, low memory.
    (pkgs.python3.withPackages (ps: with ps; [
      torch
      torchinfo
      numpy
    ]))

    # Linter — same binary used in CI and by CLAUDE.md
    pkgs.ruff
  ];

  shellHook = ''
    echo "dev shell ready"
    echo "  python utils/shape_check.py   — validate tensor shapes (Phase 1 + Phase 2)"
    echo "  ruff check .                  — lint"
  '';
}
