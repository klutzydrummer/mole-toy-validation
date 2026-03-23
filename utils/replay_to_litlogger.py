"""
Replay existing JSONL training logs into LightningLogger for configs that
ran before LightningLogger was integrated.

Usage:
  python utils/replay_to_litlogger.py --config baseline
  python utils/replay_to_litlogger.py --config baseline mhc
  python utils/replay_to_litlogger.py --all
  python utils/replay_to_litlogger.py --all --teamspace my-teamspace
"""

import argparse
import json
import os
import sys

try:
    from litlogger import LightningLogger
except ImportError:
    print("ERROR: litlogger not installed. Run: pip install litlogger")
    sys.exit(1)

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
CONFIGS = ["baseline", "mhc", "mol", "compose"]


def load_jsonl(config):
    path = os.path.join(CKPT_DIR, f"{config}.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def load_summary(config):
    path = os.path.join(CKPT_DIR, f"{config}_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def replay(config, teamspace):
    records = load_jsonl(config)
    if not records:
        print(f"  [{config}] No JSONL data found — skipping.")
        return

    summary = load_summary(config)

    train_records = [r for r in records if "type" not in r]
    eval_records  = [r for r in records if r.get("type") == "eval"]
    mol_records   = [r for r in records if r.get("type") == "mol"]

    print(f"  [{config}] {len(train_records)} train records, "
          f"{len(eval_records)} eval records, {len(mol_records)} mol records")

    lit = LightningLogger(name=f"phase1-{config}", teamspace=teamspace)

    # Log hyperparams from summary if available
    if summary:
        lit.log_hyperparams({
            "config": config,
            "d": summary.get("d", 256),
            "n_layers": summary.get("n_layers", 8),
            "n_heads": summary.get("n_heads", 8),
            "seq_len": summary.get("seq_len", 512),
            "batch_size": summary.get("batch_size", 32),
            "total_steps": summary.get("total_steps", 50000),
            "n_params": summary.get("n_params"),
        })

    # Deduplicate and sort train records by step
    seen_steps = {}
    for r in train_records:
        seen_steps[r["step"]] = r
    train_sorted = sorted(seen_steps.values(), key=lambda r: r["step"])

    # Replay train steps
    n = len(train_sorted)
    for i, r in enumerate(train_sorted):
        step = r["step"]
        metrics = {"train/loss": r["loss"], "train/bpc": r["bpc"], "train/lr": r["lr"]}
        if "grad_norm" in r:
            metrics["train/grad_norm"] = r["grad_norm"]
        lit.log_metrics(metrics, step=step)

        if (i + 1) % 100 == 0 or i == n - 1:
            print(f"    train {i+1}/{n} (step {step})", end="\r")

    print()

    # Replay eval records
    seen_eval = {}
    for r in eval_records:
        seen_eval[r["step"]] = r
    for step, r in sorted(seen_eval.items()):
        lit.log_metrics({"val/loss": r["val_loss"], "val/bpc": r["val_bpc"]}, step=step)

    # Replay mol records (avg balance per eval checkpoint)
    seen_mol = {}
    for r in mol_records:
        seen_mol[r["step"]] = r
    for step, r in sorted(seen_mol.items()):
        stats = r.get("mol_stats", [])
        if stats:
            avg_bal = sum(s["expert_balance"] for s in stats) / len(stats)
            lit.log_metrics({"mol/avg_balance": avg_bal}, step=step)

    # Final summary metrics
    if summary:
        lit.log_metrics({
            "final/best_val_bpc": summary["best_val_bpc"],
            "final/val_bpc": summary["final_val_bpc"],
        })

    lit.finalize()
    print(f"  [{config}] Done.")


def main():
    parser = argparse.ArgumentParser(description="Replay JSONL logs to LightningLogger")
    parser.add_argument("--config", nargs="+", choices=CONFIGS,
                        help="One or more configs to replay")
    parser.add_argument("--all", action="store_true",
                        help="Replay all configs that have JSONL data")
    parser.add_argument("--teamspace", type=str, default="mole-toy-validation-project")
    args = parser.parse_args()

    if not args.config and not args.all:
        parser.error("Specify --config <name> or --all")

    configs = CONFIGS if args.all else args.config

    # Filter to only configs with data
    to_replay = [c for c in configs
                 if os.path.exists(os.path.join(CKPT_DIR, f"{c}.jsonl"))]

    if not to_replay:
        print("No JSONL files found in checkpoints/. Nothing to replay.")
        sys.exit(0)

    print(f"Replaying {len(to_replay)} config(s) to teamspace '{args.teamspace}':")
    for config in to_replay:
        print(f"\nReplaying: {config}")
        replay(config, args.teamspace)

    print("\nAll done.")


if __name__ == "__main__":
    main()
