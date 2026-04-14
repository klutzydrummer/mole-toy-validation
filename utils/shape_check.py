"""
Shape validation — single forward pass of every model config on CPU.

Checks:
  - Phase 1 (9 configs) at d=512, seq=256, batch=1
  - Phase 1 scaling (10 configs: baseline/mla/diff_attn/diff_mla/mol at d=256 and d=768)
  - Phase 2 (10 configs) at d=512

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
from phase2.model import OuterModel

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
        ("mol_single",    {"mol_rank": 72}),  # rank=72 exact capacity match to mol (9×8)
        ("compose",       {}),
        ("mla",           {}),
        ("diff_attn",     {}),
        ("diff_mla",      {}),
        # go-mHC compositions
        ("diff_mhc",      {}),
        ("mla_mhc",       {}),
        ("diff_mla_mhc",  {}),
        # nGPT hypersphere experiments
        ("ngpt",           {}),
        ("ngpt_mla",       {}),
        ("ngpt_diff_attn", {}),
    ]

    results = []
    param_counts = {}
    for name, kwargs in configs:
        print(f"\n{'='*60}")
        print(f"Phase 1 — {name}")
        print("="*60)
        try:
            # mol_rank default is 8; mol_single overrides to 72 via kwargs (exact capacity match)
            mol_rank = kwargs.pop("mol_rank", 8)
            model = ToyTransformer(
                config=name, d=D, n_layers=N_LAYERS, n_heads=N_HEADS,
                vocab_size=VOCAB, max_len=SEQ_LEN + 64,
                mol_rank=mol_rank,
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

            param_counts[name] = sum(p.numel() for p in model.parameters())
            print(f"\n[PASS] {name}: output {tuple(out.shape)}  params={param_counts[name]:,}")
            results.append((name, True, None))

        except Exception as e:
            import traceback
            print(f"\n[FAIL] {name}: {e}")
            traceback.print_exc()
            results.append((name, False, str(e)))

    # Q1 fairness check: baseline_wide must be within 5% of mol total param count.
    # d_ff=1600 is an approximation — STUDY_DESIGN.md documents the derivation as
    # "approximately d_ff ≈ 1600". Actual gap at d=512 is ~3.2% (acceptable per study design).
    # This check catches gross misconfigurations (e.g., accidentally using default d_ff=1408).
    if "mol" in param_counts and "baseline_wide" in param_counts:
        mol_p = param_counts["mol"]
        bw_p  = param_counts["baseline_wide"]
        ratio = abs(bw_p - mol_p) / mol_p
        if ratio >= 0.05:
            print(f"\n[FAIL] Q1 fairness: baseline_wide ({bw_p:,}) and mol ({mol_p:,}) "
                  f"differ by {ratio:.1%} (>5%). Adjust d_ff.")
            results.append(("Q1_param_match", False, f"baseline_wide/mol ratio {ratio:.3f}"))
        else:
            print(f"\n[PASS] Q1 param match: baseline_wide={bw_p:,}  mol={mol_p:,}  "
                  f"diff={ratio:.1%} (within 5% tolerance)")

    # Q2 fairness check: mol_single (rank=72) must be within 2% of mol total param count.
    if "mol" in param_counts and "mol_single" in param_counts:
        mol_p = param_counts["mol"]
        ms_p  = param_counts["mol_single"]
        ratio = abs(ms_p - mol_p) / mol_p
        if ratio >= 0.02:
            print(f"\n[FAIL] Q2 fairness: mol_single ({ms_p:,}) and mol ({mol_p:,}) "
                  f"differ by {ratio:.1%} (>2%). Adjust mol_single rank.")
            results.append(("Q2_param_match", False, f"mol_single/mol ratio {ratio:.3f}"))
        else:
            print(f"\n[PASS] Q2 param match: mol_single={ms_p:,}  mol={mol_p:,}  "
                  f"diff={ratio:.1%}")

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
    configs = list(OuterModel.CONFIGS.keys())

    results = []
    for name in configs:
        print(f"\n{'='*60}")
        print(f"Phase 2 — {name}")
        print("="*60)
        try:
            model = OuterModel(
                config=name, d=D, n_layers=N_LAYERS, n_heads=N_HEADS,
                vocab_size=VOCAB, seq_len=SEQ_LEN,
            ).eval()

            # torchinfo with tuple-output models: wrap to return only logits
            class _LogitsOnly(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self._m = m
                def forward(self, x):
                    return self._m(x)[0]

            _summary(_LogitsOnly(model), input_size=(BATCH, SEQ_LEN), dtypes=[torch.long])

            x = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
            with torch.no_grad():
                logits, boundary_probs, compression_ratio = model(x)

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
            cr = compression_ratio.item()
            assert 0.0 < cr <= 1.0, f"compression_ratio out of (0, 1]: {cr:.4f}"

            mean_bp = boundary_probs.mean().item()
            print(
                f"\n  boundary_probs mean={mean_bp:.4f}"
                f"  compression_ratio={cr:.4f}"
                f"  (target_rate={model.target_rate:.2f})"
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
