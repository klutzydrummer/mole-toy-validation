#!/usr/bin/env python3
"""
Phase 2 smoke test — MUST PASS before any Phase 2 training run is allowed.

Runs 1000 training steps of hdc_rulebased (no learned router, simplest config)
then checks the health metrics from zone_ed_pipeline.md checklist item 10.
Takes ~3 minutes on a T4 GPU. Catches pipeline failures before they waste hours.

CHECKS (all must pass):
  1. Loss reduction: loss(steps 800-1000) < 0.90 * loss(steps 0-200)
     Catches: broken gradient flow, NaN propagation.

  2. Encoder diversity: encoder_out.var(dim=1).mean() > 1e-4
     Catches: CRL over-smoothing — Zone E producing near-identical representations
     across all L positions, making the inner transformer blind (root cause of the
     9.7 BPC plateau in the initial Phase 2 runs).

  3. Concept diversity: concept_tokens.var(dim=1).mean() > 1e-4
     Catches: inner transformer receiving near-identical concept token inputs,
     which causes attention to degenerate and concept_out to be uniform.

  4. No NaN or inf in any metric at any step.

Result is written to checkpoints/smoke_test_result.json with the current
git blob hash of phase2/model.py. run_experiments.sh reads this file and
blocks Phase 2 training if result != "pass" or the model hash has changed.

Usage:
  python utils/smoke_test.py                # run smoke test (1000 steps)
  python utils/smoke_test.py --check-only   # check stored result only, no training
  python utils/smoke_test.py --steps 500    # override step count (min recommended: 500)
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_FILE = REPO_ROOT / "checkpoints" / "smoke_test_result.json"

# Thresholds from zone_ed_pipeline.md checklist item 10
ENCODER_DIVERSITY_MIN   = 1e-4   # encoder_out.var(dim=1).mean()
CONCEPT_DIVERSITY_MIN   = 1e-4   # concept_tokens.var(dim=1).mean()
LOSS_REDUCTION_REQUIRED = 0.90   # loss at end must be < 0.90 * loss at start


def _git_blob_hash(rel_path: str) -> str | None:
    try:
        r = subprocess.run(
            ["git", "hash-object", rel_path],
            capture_output=True, text=True, cwd=REPO_ROOT, check=True,
        )
        h = r.stdout.strip()
        return h if h else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _unwrap(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _check_finite(val: float, name: str) -> bool:
    if not math.isfinite(val):
        print(f"  FAIL: {name} = {val} (NaN or inf)")
        return False
    return True


@torch.no_grad()
def _diversity_metrics(model, val_loader, device) -> dict:
    """
    Forward pass to measure encoder_out and concept_token diversity.
    Returns dict with encoder_diversity and concept_diversity.
    Both are variance across the sequence dimension (L or M), averaged over
    batch and feature dimensions.
    """
    raw = _unwrap(model)
    raw.eval()

    x, _ = next(iter(val_loader))
    x = x.to(device)

    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=(device.type == "cuda")):
        h = raw.embed(x)                                       # [B, L, d]
        concept_tokens, encoder_out, _, _, _ = raw.zone_e(h)  # [B, M, d], [B, L, d]

    # Variance across sequence positions, averaged over batch and d
    # .var(dim=1): [B, d] — per-channel variance across positions
    enc_div = encoder_out.float().var(dim=1).mean().item()     # scalar
    con_div = concept_tokens.float().var(dim=1).mean().item()  # scalar

    raw.train()
    return {"encoder_diversity": enc_div, "concept_diversity": con_div}


def run_smoke_test(steps: int = 1000, batch_size: int = 32) -> dict:
    """
    Train hdc_rulebased for `steps` steps and evaluate health metrics.
    Returns a result dict: {"pass": bool, "checks": {...}, "metrics": {...}}.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from phase2.model import HDCModel
    from utils.data import get_dataloader, set_dataset, set_tokenizer, get_vocab_size

    set_dataset("wikitext103")
    set_tokenizer("bpe")
    vocab_size = get_vocab_size()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print(f"  device: {device}  vocab: {vocab_size}  steps: {steps}  batch: {batch_size}")

    model = HDCModel(
        config="hdc_rulebased", d=512, n_layers=8, n_heads=8,
        vocab_size=vocab_size, seq_len=256,
        n_experts=8, mol_rank=8, mol_top_k=2,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95),
        weight_decay=0.1, fused=(device.type == "cuda"),
    )
    scaler = torch.amp.GradScaler(enabled=use_amp)

    train_loader = get_dataloader("train", seq_len=256, batch_size=batch_size)
    val_loader   = get_dataloader("val",   seq_len=256, batch_size=batch_size)
    train_iter   = iter(train_loader)

    early_losses  = []   # steps 0 .. steps//5
    late_losses   = []   # steps 4*steps//5 .. steps
    any_nonfinite = False

    model.train()
    t0 = time.time()
    early_cutoff = steps // 5
    late_cutoff  = 4 * steps // 5

    for step in range(steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits, bp = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        if not math.isfinite(loss.item()):
            any_nonfinite = True
            print(f"  NaN/inf loss at step {step}")
            break

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step < early_cutoff:
            early_losses.append(loss.item())
        if step >= late_cutoff:
            late_losses.append(loss.item())

        if (step + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (step + 1) * (steps - step - 1)
            print(f"  step {step+1:4d}/{steps}  loss={loss.item():.4f}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"  Training complete in {elapsed:.0f}s")

    # --- Health metric checks ---
    diversity = _diversity_metrics(model, val_loader, device)
    loss_early = sum(early_losses) / max(1, len(early_losses))
    loss_late  = sum(late_losses)  / max(1, len(late_losses))
    loss_ratio = loss_late / max(loss_early, 1e-9)

    checks = {}
    metrics = {
        "loss_early":         loss_early,
        "loss_late":          loss_late,
        "loss_ratio":         loss_ratio,
        "encoder_diversity":  diversity["encoder_diversity"],
        "concept_diversity":  diversity["concept_diversity"],
        "any_nonfinite":      any_nonfinite,
        "steps":              steps,
        "elapsed_seconds":    elapsed,
    }

    print()
    print("  ── Check results ──────────────────────────────────────")

    # Check 1: no NaN/inf
    checks["no_nonfinite"] = not any_nonfinite
    _print_check("no NaN/inf during training", checks["no_nonfinite"])

    # Check 2: loss is finite
    checks["loss_finite"] = (
        _check_finite(loss_early, "loss_early") and
        _check_finite(loss_late,  "loss_late")
    )
    _print_check("loss values are finite", checks["loss_finite"])

    # Check 3: loss reduction
    checks["loss_reduction"] = (
        checks["loss_finite"] and loss_ratio < LOSS_REDUCTION_REQUIRED
    )
    _print_check(
        f"loss reduced by ≥10%  ({loss_ratio:.3f} < {LOSS_REDUCTION_REQUIRED:.2f})",
        checks["loss_reduction"],
        f"early={loss_early:.4f}  late={loss_late:.4f}  ratio={loss_ratio:.3f}",
    )

    # Check 4: encoder diversity (root cause of 9.7 BPC plateau)
    checks["encoder_diversity"] = (
        math.isfinite(diversity["encoder_diversity"]) and
        diversity["encoder_diversity"] > ENCODER_DIVERSITY_MIN
    )
    _print_check(
        f"Zone E encoder_out is diverse  (var={diversity['encoder_diversity']:.2e} > {ENCODER_DIVERSITY_MIN:.0e})",
        checks["encoder_diversity"],
        "CRITICAL: Zone E CRL over-smoothing — all encoder positions look the same. "
        "Inner transformer cannot learn. See causal_recurrence.md and zone_ed_pipeline.md item 10.",
    )

    # Check 5: concept diversity
    checks["concept_diversity"] = (
        math.isfinite(diversity["concept_diversity"]) and
        diversity["concept_diversity"] > CONCEPT_DIVERSITY_MIN
    )
    _print_check(
        f"Concept tokens are diverse  (var={diversity['concept_diversity']:.2e} > {CONCEPT_DIVERSITY_MIN:.0e})",
        checks["concept_diversity"],
        "CRITICAL: Concept tokens are near-identical — inner transformer attention "
        "will degenerate. Fix Zone E before running full Phase 2.",
    )

    all_pass = all(checks.values())
    print()
    if all_pass:
        print("  SMOKE TEST PASSED — Phase 2 training is unblocked.")
    else:
        failed = [k for k, v in checks.items() if not v]
        print(f"  SMOKE TEST FAILED — {len(failed)} check(s) failed: {', '.join(failed)}")
        print("  Phase 2 training is BLOCKED until this passes.")
        print("  Fix the root cause, update verification, then re-run this test.")

    return {"pass": all_pass, "checks": checks, "metrics": metrics}


def _print_check(label: str, passed: bool, detail: str = "") -> None:
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {label}")
    if not passed and detail:
        # Wrap detail at 80 chars with indent
        words = detail.split()
        line = "         "
        for word in words:
            if len(line) + len(word) + 1 > 80:
                print(line)
                line = "         " + word
            else:
                line += (" " if line.strip() else "") + word
        if line.strip():
            print(line)


def cmd_run(args) -> int:
    print("=" * 60)
    print("  Phase 2 Smoke Test")
    print("=" * 60)
    print()

    result = run_smoke_test(steps=args.steps, batch_size=args.batch_size)
    _save_result(result)
    return 0 if result["pass"] else 1


def cmd_check_only(args) -> int:
    """Check stored result without running training."""
    if not RESULT_FILE.exists():
        print("SMOKE TEST: no result on file — run `python utils/smoke_test.py` first.")
        return 1

    with open(RESULT_FILE) as f:
        stored = json.load(f)

    current_hash = _git_blob_hash("phase2/model.py")
    stored_hash  = stored.get("model_hash")

    if current_hash != stored_hash:
        print("SMOKE TEST: phase2/model.py has changed since last smoke test.")
        print(f"  stored hash:  {stored_hash}")
        print(f"  current hash: {current_hash}")
        print("  Re-run: python utils/smoke_test.py")
        return 1

    if stored.get("result") != "pass":
        failed = [k for k, v in stored.get("checks", {}).items() if not v]
        print(f"SMOKE TEST: last run FAILED — {', '.join(failed)}")
        print(f"  run at: {stored.get('run_at', 'unknown')}")
        print("  Re-run after fixing the root cause: python utils/smoke_test.py")
        return 1

    print(f"SMOKE TEST: pass (run at {stored.get('run_at', 'unknown')}, "
          f"model hash {stored_hash[:12]})")
    return 0


def _save_result(result: dict) -> None:
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    now          = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    model_hash   = _git_blob_hash("phase2/model.py")
    payload = {
        "result":     "pass" if result["pass"] else "fail",
        "run_at":     now,
        "model_hash": model_hash,
        "checks":     result["checks"],
        "metrics":    result["metrics"],
    }
    with open(RESULT_FILE, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"\n  Result saved to {RESULT_FILE.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 smoke test")
    parser.add_argument("--check-only",  action="store_true",
                        help="Check stored result only — do not run training")
    parser.add_argument("--steps",       type=int, default=1000,
                        help="Training steps (default: 1000, minimum recommended: 500)")
    parser.add_argument("--batch_size",  type=int, default=32,
                        help="Batch size (default: 32, matches run_experiments.sh)")
    args = parser.parse_args()

    if args.check_only:
        sys.exit(cmd_check_only(args))
    else:
        sys.exit(cmd_run(args))
