"""
Training reporter — runs as a separate process, reads JSONL logs written by
train.py, and generates checkpoints/report.md for Claude Code and humans.

Usage:
  python utils/reporter.py                  # one-shot report
  python utils/reporter.py --watch          # poll every 30s until killed
  python utils/reporter.py --watch --interval 60
"""

import argparse
import glob
import json
import math
import os
import sys
import time
from datetime import datetime

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
REPORT_PATH = os.path.join(CKPT_DIR, "report.md")
CONFIGS = ["baseline", "mhc", "mol", "compose"]
TOTAL_STEPS = 50000


# ── data loading ─────────────────────────────────────────────────────────────

def load_jsonl(config):
    path = os.path.join(CKPT_DIR, f"{config}.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
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


# ── analysis helpers ──────────────────────────────────────────────────────────

def smooth(values, window=10):
    """Simple moving average."""
    if len(values) < 2:
        return values
    out = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def convergence_status(eval_bpcs):
    """
    Returns (status_str, delta_per_1k_steps) based on last 3 eval points.
    status: 'improving' | 'plateaued' | 'diverging' | 'insufficient data'
    """
    if len(eval_bpcs) < 2:
        return "insufficient data", None
    recent = eval_bpcs[-3:]
    # delta between first and last of recent window
    delta = recent[-1][1] - recent[0][1]   # positive = getting worse
    steps_span = recent[-1][0] - recent[0][0]
    rate = (delta / steps_span * 1000) if steps_span > 0 else 0.0  # BPC change per 1k steps

    if len(eval_bpcs) < 3:
        if delta < -0.002:
            return "improving", rate
        return "insufficient data", rate

    if delta > 0.005:
        return "diverging", rate
    elif delta > -0.002:
        return "plateaued", rate
    else:
        return "improving", rate


def detect_loss_spikes(train_records, threshold=1.5):
    """Return list of (step, bpc) where loss spiked > threshold × previous."""
    spikes = []
    bpcs = [(r["step"], r["bpc"]) for r in train_records if "bpc" in r and "type" not in r]
    for i in range(1, len(bpcs)):
        if bpcs[i][1] > bpcs[i-1][1] * threshold and bpcs[i][1] > 2.0:
            spikes.append(bpcs[i])
    return spikes


def sample_curve(points, n=20):
    """Downsample a list of (step, val) to at most n points."""
    if len(points) <= n:
        return points
    step = len(points) // n
    return points[::step]


# ── per-config analysis ───────────────────────────────────────────────────────

def analyze_config(config):
    records = load_jsonl(config)
    summary = load_summary(config)

    train_records = [r for r in records if "type" not in r]
    eval_records  = [r for r in records if r.get("type") == "eval"]
    mol_records   = [r for r in records if r.get("type") == "mol"]

    if not records:
        return None

    result = {
        "config": config,
        "n_train": len(train_records),
        "n_eval": len(eval_records),
        "current_step": train_records[-1]["step"] if train_records else 0,
        "current_bpc": train_records[-1]["bpc"] if train_records else None,
        "current_lr": train_records[-1]["lr"] if train_records else None,
        "eval_bpcs": [(r["step"], r["val_bpc"]) for r in eval_records],
        "best_val_bpc": min((r["val_bpc"] for r in eval_records), default=None),
        "spikes": detect_loss_spikes(train_records),
        "summary": summary,
        "mol_records": mol_records,
        "train_curve": sample_curve(
            [(r["step"], r["bpc"]) for r in train_records if "bpc" in r], n=25
        ),
    }

    result["status"], result["bpc_rate"] = convergence_status(result["eval_bpcs"])
    result["progress_pct"] = 100.0 * result["current_step"] / TOTAL_STEPS

    # MoL: latest avg balance per layer
    if mol_records:
        latest_mol = mol_records[-1]["mol_stats"]
        result["mol_balance"] = {
            s["layer"]: s["expert_balance"] for s in latest_mol
        }
        result["mol_counts"] = {
            s["layer"]: s["expert_counts"] for s in latest_mol
        }
    else:
        result["mol_balance"] = {}
        result["mol_counts"] = {}

    return result


# ── report generation ─────────────────────────────────────────────────────────

STATUS_ICON = {
    "improving": "↓",          # BPC going down = good
    "plateaued": "→",
    "diverging": "↑ ⚠",
    "insufficient data": "…",
}


def render_report(analyses):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = []

    lines += [
        f"# Phase 1 Training Report",
        f"",
        f"_Generated {now} — updated every 30s by `utils/reporter.py`_",
        f"",
        f"## Summary",
        f"",
    ]

    # Summary table
    header = f"| Config     | Progress | Current BPC | Best Val BPC | Status |"
    sep    = f"|------------|----------|-------------|--------------|--------|"
    lines += [header, sep]
    for a in analyses:
        if a is None:
            continue
        prog = f"{a['progress_pct']:.0f}% ({a['current_step']:,}/{TOTAL_STEPS:,})"
        cur  = f"{a['current_bpc']:.4f}" if a['current_bpc'] is not None else "—"
        best = f"{a['best_val_bpc']:.4f}" if a['best_val_bpc'] is not None else "—"
        icon = STATUS_ICON.get(a["status"], "?")
        lines.append(f"| {a['config']:<10} | {prog:<8} | {cur:<11} | {best:<12} | {icon} {a['status']:<18} |")

    lines += ["", "---", ""]

    # Per-config detail
    for a in analyses:
        if a is None:
            continue

        lines += [f"## `{a['config']}`", ""]

        # Convergence
        if a["bpc_rate"] is not None:
            direction = "improvement" if a["bpc_rate"] < 0 else "degradation"
            lines.append(f"**Convergence**: {a['status']} — {abs(a['bpc_rate']):.4f} BPC {direction}/1k steps")
        else:
            lines.append(f"**Convergence**: {a['status']}")
        lines.append("")

        # Val BPC progression
        if a["eval_bpcs"]:
            lines.append("**Validation BPC at eval checkpoints:**")
            lines.append("")
            lines.append("| Step | Val BPC | Δ from prev |")
            lines.append("|------|---------|-------------|")
            prev = None
            for step, bpc in a["eval_bpcs"]:
                delta = f"{bpc - prev:+.4f}" if prev is not None else "—"
                marker = " ★" if bpc == a["best_val_bpc"] else ""
                lines.append(f"| {step:>6,} | {bpc:.4f}{marker} | {delta} |")
                prev = bpc
            lines.append("")

        # Smoothed train loss curve (sampled)
        if a["train_curve"]:
            lines.append("**Training BPC (sampled, ~25 points):**")
            lines.append("")
            lines.append("```")
            for step, bpc in a["train_curve"]:
                bar_len = max(0, int((bpc - 1.0) * 40))
                bar = "█" * min(bar_len, 60)
                lines.append(f"  {step:>6,} | {bpc:.4f} | {bar}")
            lines.append("```")
            lines.append("")

        # Spikes
        if a["spikes"]:
            lines.append(f"**⚠ Loss spikes detected** ({len(a['spikes'])} events):")
            for step, bpc in a["spikes"][:5]:
                lines.append(f"  - step {step:,}: BPC {bpc:.4f}")
            lines.append("")

        # MoL expert balance
        if a["mol_balance"]:
            lines.append("**Expert utilization (latest eval window):**")
            lines.append("")
            lines.append("| Layer | Balance (1.0=ideal) | Expert counts (%) |")
            lines.append("|-------|---------------------|-------------------|")
            for layer, bal in sorted(a["mol_balance"].items()):
                counts = a["mol_counts"].get(layer, [])
                total = sum(counts)
                pcts = " ".join(f"{100*c/total:.0f}%" for c in counts) if total > 0 else "—"
                flag = " ⚠ collapsed" if bal < 0.5 else ""
                lines.append(f"| {layer:>5} | {bal:.3f}{flag:<14} | {pcts} |")
            lines.append("")

        # Final summary if training complete
        if a["summary"]:
            s = a["summary"]
            lines += [
                f"**Training complete** — {s['elapsed_seconds']/60:.1f} min, "
                f"{s['n_params']:,} params, best val BPC **{s['best_val_bpc']:.4f}**",
                "",
            ]

        lines += ["---", ""]

    # Cross-config comparison (only when ≥2 have eval data)
    complete = [a for a in analyses if a and a["best_val_bpc"] is not None]
    if len(complete) >= 2:
        lines += ["## Cross-config comparison", ""]
        ranked = sorted(complete, key=lambda a: a["best_val_bpc"])
        baseline_bpc = next((a["best_val_bpc"] for a in ranked if a["config"] == "baseline"), None)
        lines.append("| Rank | Config | Best Val BPC | vs baseline |")
        lines.append("|------|--------|--------------|-------------|")
        for i, a in enumerate(ranked, 1):
            if baseline_bpc:
                delta = a["best_val_bpc"] - baseline_bpc
                vs = f"{delta:+.4f}"
            else:
                vs = "—"
            lines.append(f"| {i} | `{a['config']}` | {a['best_val_bpc']:.4f} | {vs} |")
        lines += ["", "---", ""]

    # Claude Code guidance section
    lines += [
        "## Notes for Claude Code",
        "",
        "This report is auto-generated from `checkpoints/*.jsonl`. To interpret:",
        "",
        "- **BPC (bits per character)**: lower is better. Baseline target ~1.2–1.4 BPC.",
        "- **Balance score**: MoL expert entropy / max_entropy. <0.5 = expert collapse (bad).",
        "- **`improving`**: val BPC dropped >0.002 over last 3 evals. `plateaued`: <0.002 change.",
        "- **Loss spikes**: single-step BPC >1.5× previous AND >2.0 BPC. Usually transient.",
        "- To re-run analysis: `python utils/reporter.py`",
        "- To watch live: `python utils/reporter.py --watch`",
        "",
    ]

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def run_once():
    os.makedirs(CKPT_DIR, exist_ok=True)
    analyses = [analyze_config(c) for c in CONFIGS]
    # Only report configs that have data
    analyses = [a for a in analyses if a is not None]
    if not analyses:
        print("No training data found in checkpoints/. Run training first.")
        return
    report = render_report(analyses)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"Report written to {REPORT_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Training reporter")
    parser.add_argument("--watch", action="store_true",
                        help="Poll continuously until killed (Ctrl+C)")
    parser.add_argument("--interval", type=int, default=30,
                        help="Poll interval in seconds (default: 30)")
    args = parser.parse_args()

    if args.watch:
        print(f"Watching checkpoints/ — updating report every {args.interval}s. Ctrl+C to stop.")
        try:
            while True:
                run_once()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nReporter stopped.")
            run_once()  # final report on exit
    else:
        run_once()


if __name__ == "__main__":
    main()
