"""
Shape validation — single forward pass of every model config on CPU.

Checks:
  - Phase 1 (9 configs) at d=512, seq=256, batch=1
  - Phase 1 scaling (10 configs: baseline/mla/diff_attn/diff_mla/mol at d=256 and d=768)
  - Phase 2 (6 non-upcycle configs) at d=512

Skips upcycle configs (they require a pre-trained mol checkpoint).

Run locally before pushing any phase1/model.py or phase2/model.py change:
    nix-shell shell.nix --run "python utils/shape_check.py"
"""

import os
import sys

import torch
import torchinfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1.model import ToyTransformer
from phase2.model import HDCModel

# ── Full production dims ───────────────────────────────────────────────────────
VOCAB     = 4096
D         = 512
N_LAYERS  = 8
N_HEADS   = 8
SEQ_LEN   = 256
BATCH     = 1


def _summary(model, input_size, dtypes):
    """Run torchinfo at depth=3 — shows ZoneE/ZoneD/InnerTransformer structure
    without expanding every attention sub-sublayer into noise."""
    return torchinfo.summary(
        model,
        input_size=input_size,
        dtypes=dtypes,
        col_names=["input_size", "output_size", "num_params"],
        depth=3,
        verbose=1,
        device="cpu",
    )


def check_phase1():
    configs = [
        # (config_name, extra kwargs for ToyTransformer)
        ("baseline",      {}),
        ("baseline_wide", {"d_ff": 1600}),  # d_ff scaled to match mol params
        ("mhc",           {}),
        ("mol",           {}),
        ("mol_single",    {}),
        ("compose",       {}),
        ("mla",           {}),
        ("diff_attn",     {}),
        ("diff_mla",      {}),
    ]

    results = []
    for name, kwargs in configs:
        print(f"\n{'='*60}")
        print(f"Phase 1 — {name}")
        print("="*60)
        try:
            model = ToyTransformer(
                config=name, d=D, n_layers=N_LAYERS, n_heads=N_HEADS,
                vocab_size=VOCAB, max_len=SEQ_LEN + 64,
                mol_rank=8,
                **kwargs,
            ).eval()

            _summary(model, input_size=(BATCH, SEQ_LEN), dtypes=[torch.long])

            x = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
            with torch.no_grad():
                out = model(x)

            # Shape contract: [B, L, vocab]
            assert out.shape == (BATCH, SEQ_LEN, VOCAB), (
                f"Expected ({BATCH}, {SEQ_LEN}, {VOCAB}), got {out.shape}"
            )
            assert not torch.isnan(out).any(), "NaN in output logits"
            assert not torch.isinf(out).any(), "Inf in output logits"

            print(f"\n[PASS] {name}: output {tuple(out.shape)}")
            results.append((name, True, None))

        except Exception as e:
            import traceback
            print(f"\n[FAIL] {name}: {e}")
            traceback.print_exc()
            results.append((name, False, str(e)))

    return results


def check_phase1_scaling():
    """Validate Phase 1 scaling configs at d=256 (n_heads=4) and d=768 (n_heads=12)."""
    scale_configs = [
        # (config_name, d, n_heads)
        ("baseline",  256, 4),
        ("mla",       256, 4),
        ("diff_attn", 256, 4),
        ("diff_mla",  256, 4),
        ("mol",       256, 4),
        ("baseline",  768, 12),
        ("mla",       768, 12),
        ("diff_attn", 768, 12),
        ("diff_mla",  768, 12),
        ("mol",       768, 12),
    ]

    results = []
    for name, d, n_heads in scale_configs:
        label = f"{name}_d{d}"
        print(f"\n{'='*60}")
        print(f"Phase 1 Scaling — {label}  (d={d}, n_heads={n_heads})")
        print("="*60)
        try:
            model = ToyTransformer(
                config=name, d=d, n_layers=N_LAYERS, n_heads=n_heads,
                vocab_size=VOCAB, max_len=SEQ_LEN + 64,
                mol_rank=8,
            ).eval()

            x = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
            with torch.no_grad():
                out = model(x)

            assert out.shape == (BATCH, SEQ_LEN, VOCAB), (
                f"Expected ({BATCH}, {SEQ_LEN}, {VOCAB}), got {out.shape}"
            )
            assert not torch.isnan(out).any(), "NaN in output logits"
            assert not torch.isinf(out).any(), "Inf in output logits"

            n_params = sum(p.numel() for p in model.parameters())
            print(f"\n[PASS] {label}: output {tuple(out.shape)}  params={n_params:,}")
            results.append((label, True, None))

        except Exception as e:
            import traceback
            print(f"\n[FAIL] {label}: {e}")
            traceback.print_exc()
            results.append((label, False, str(e)))

    return results


def check_phase2():
    # Upcycle configs (hdc_upcycle_*) require a pre-trained mol checkpoint — skipped.
    configs = [
        "hdc_rulebased",
        "hdc_gate",
        "hdc_stride",
        "hdc_r2",
        "hdc_r8",
        "hdc_e2e_isolated",
    ]

    results = []
    for name in configs:
        print(f"\n{'='*60}")
        print(f"Phase 2 — {name}")
        print("="*60)
        try:
            model = HDCModel(
                config=name, d=D, n_layers=N_LAYERS, n_heads=N_HEADS,
                vocab_size=VOCAB, seq_len=SEQ_LEN,
                mol_rank=4,
            ).eval()

            R = model.R
            M = SEQ_LEN // R

            # torchinfo with tuple-output models: wrap to return only logits
            # so summary can infer output shape; we assert both outputs manually.
            class _LogitsOnly(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self._m = m
                def forward(self, x):
                    return self._m(x)[0]

            _summary(_LogitsOnly(model), input_size=(BATCH, SEQ_LEN), dtypes=[torch.long])

            x = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
            with torch.no_grad():
                logits, boundary_probs = model(x)

            # Shape contracts
            assert logits.shape == (BATCH, SEQ_LEN, VOCAB), (
                f"logits: expected ({BATCH}, {SEQ_LEN}, {VOCAB}), got {logits.shape}"
            )
            assert boundary_probs.shape == (BATCH, SEQ_LEN), (
                f"boundary_probs: expected ({BATCH}, {SEQ_LEN}), got {boundary_probs.shape}"
            )

            # Value contracts
            assert not torch.isnan(logits).any(), "NaN in logits"
            assert not torch.isinf(logits).any(), "Inf in logits"
            bp_min, bp_max = boundary_probs.min().item(), boundary_probs.max().item()
            assert bp_min >= 0.0 and bp_max <= 1.0, (
                f"boundary_probs out of [0,1]: [{bp_min:.4f}, {bp_max:.4f}]"
            )

            # Compression: exactly M = SEQ_LEN // R positions selected
            # At init, mean boundary_prob won't equal 1/R — just report it.
            mean_bp = boundary_probs.mean().item()
            print(
                f"\n  R={R}  M={M}  boundary_probs mean={mean_bp:.4f}"
                f"  (target at convergence: {1.0/R:.4f})"
            )

            print(
                f"[PASS] {name}: logits {tuple(logits.shape)},"
                f" boundary_probs {tuple(boundary_probs.shape)}"
            )
            results.append((name, True, None))

        except Exception as e:
            import traceback
            print(f"\n[FAIL] {name}: {e}")
            traceback.print_exc()
            results.append((name, False, str(e)))

    return results


def main():
    print("Shape check — Phase 1 + Phase 2 at full production dims")
    print(f"  d={D}, n_layers={N_LAYERS}, n_heads={N_HEADS}, seq={SEQ_LEN}, batch={BATCH}, vocab={VOCAB}")
    print(f"  device: CPU  |  torch: {torch.__version__}")

    p1 = check_phase1()
    p1s = check_phase1_scaling()
    p2 = check_phase2()

    all_results = p1 + p1s + p2
    passed = [r for r in all_results if r[1]]
    failed = [r for r in all_results if not r[1]]

    print(f"\n{'='*60}")
    print(f"Results: {len(passed)}/{len(all_results)} passed")
    if failed:
        print("\nFailed configs:")
        for name, _, err in failed:
            print(f"  {name}: {err}")
        sys.exit(1)
    else:
        print("All shape contracts verified.")
        sys.exit(0)


if __name__ == "__main__":
    main()
