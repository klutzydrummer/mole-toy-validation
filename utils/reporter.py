"""
Training reporter — runs as a separate process, reads JSONL logs written by
train.py, and generates checkpoints/report.md for Claude Code and humans.

Usage:
  python utils/reporter.py                  # one-shot report
  python utils/reporter.py --watch          # poll every 30s until killed
  python utils/reporter.py --watch --interval 60
"""

import argparse
import json
import math
import os
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

def smooth(values, window=5):
    """Simple moving average over a list of scalars."""
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
    Returns (status_str, rate) where rate = BPC change per 1k steps (negative = improving).
    status: 'improving' | 'plateaued' | 'diverging' | 'insufficient data'
    """
    if len(eval_bpcs) < 2:
        return "insufficient data", None
    recent = eval_bpcs[-3:]
    delta = recent[-1][1] - recent[0][1]   # positive = worse
    steps_span = recent[-1][0] - recent[0][0]
    rate = (delta / steps_span * 1000) if steps_span > 0 else 0.0

    if len(eval_bpcs) < 3:
        return ("improving" if delta < -0.002 else "insufficient data"), rate

    if delta > 0.005:
        return "diverging", rate
    elif delta > -0.002:
        return "plateaued", rate
    else:
        return "improving", rate


def improvement_deceleration(eval_bpcs):
    """
    Measure whether the *rate* of BPC improvement is itself slowing down.
    Returns (is_decelerating: bool, description: str).
    Needs ≥4 eval points to be meaningful.
    """
    if len(eval_bpcs) < 4:
        return False, None

    # Compute per-interval improvement rates (negative = improving)
    rates = []
    for i in range(1, len(eval_bpcs)):
        delta = eval_bpcs[i][1] - eval_bpcs[i-1][1]
        steps = eval_bpcs[i][0] - eval_bpcs[i-1][0]
        rates.append(delta / steps * 1000 if steps > 0 else 0.0)

    # Compare first half vs second half of rate history
    mid = len(rates) // 2
    early_rate = sum(rates[:mid]) / mid        # more negative = faster early improvement
    recent_rate = sum(rates[mid:]) / (len(rates) - mid)

    # Deceleration: early improvement was faster (more negative) than recent
    decel = recent_rate - early_rate           # positive = slowing down
    if decel > 0.005 and early_rate < -0.002:
        pct = abs(decel / early_rate) * 100
        return True, f"improvement rate slowed {pct:.0f}% (early: {early_rate:.4f} → recent: {recent_rate:.4f} BPC/1k steps)"
    return False, None


def grad_norm_trend(grad_norm_points):
    """
    Detect a sustained upward trend in grad norm over the last N points.
    Returns (is_rising: bool, description: str | None).
    A gradual rise is concerning; a spike is handled separately.
    """
    if len(grad_norm_points) < 20:
        return False, None
    recent = [g for _, g in grad_norm_points[-20:]]
    early  = [g for _, g in grad_norm_points[:20]]
    avg_recent = sum(recent) / len(recent)
    avg_early  = sum(early)  / len(early)
    ratio = avg_recent / avg_early if avg_early > 0 else 1.0
    if ratio > 1.2 and avg_recent > 0.3:
        return True, f"grad norm rising: early avg {avg_early:.3f} → recent avg {avg_recent:.3f} ({ratio:.2f}×)"
    return False, None


def detect_loss_spikes(train_records, threshold=1.5):
    """Return list of (step, bpc) where loss spiked > threshold × previous."""
    spikes = []
    bpcs = [(r["step"], r["bpc"]) for r in train_records if "bpc" in r]
    for i in range(1, len(bpcs)):
        if bpcs[i][1] > bpcs[i-1][1] * threshold and bpcs[i][1] > 2.0:
            spikes.append(bpcs[i])
    return spikes


def sample_curve(points, n=25):
    """Deduplicate by step (last write wins), sort, then downsample to n points."""
    deduped = dict(points)  # last value for each step wins
    ordered = sorted(deduped.items())  # sort by step
    if len(ordered) <= n:
        return ordered
    stride = len(ordered) // n
    return ordered[::stride]


def train_bpc_at_step(train_records, target_step, window=3):
    """Return the smoothed train BPC near a given eval step."""
    bpcs = [(r["step"], r["bpc"]) for r in train_records if "bpc" in r]
    nearby = [b for s, b in bpcs if abs(s - target_step) <= window * 100]
    if not nearby:
        # Fall back to closest
        closest = min(bpcs, key=lambda x: abs(x[0] - target_step), default=None)
        return closest[1] if closest else None
    return sum(nearby) / len(nearby)


# ── per-config analysis ───────────────────────────────────────────────────────

def analyze_config(config):
    records = load_jsonl(config)
    summary = load_summary(config)

    train_records = [r for r in records if "type" not in r]
    eval_records  = [r for r in records if r.get("type") == "eval"]
    mol_records   = [r for r in records if r.get("type") == "mol"]

    if not records:
        return None

    eval_bpcs = [(r["step"], r["val_bpc"]) for r in eval_records]

    # Grad norm: smooth over all logged steps
    grad_norms = [(r["step"], r["grad_norm"]) for r in train_records if "grad_norm" in r]
    recent_gnorm = None
    if grad_norms:
        recent = [g for _, g in grad_norms[-10:]]
        recent_gnorm = sum(recent) / len(recent)

    # Train/val gap at each eval point
    train_val_gaps = []
    for step, val_bpc in eval_bpcs:
        t_bpc = train_bpc_at_step(train_records, step)
        if t_bpc is not None:
            train_val_gaps.append((step, t_bpc, val_bpc, val_bpc - t_bpc))

    status, bpc_rate = convergence_status(eval_bpcs)
    is_decelerating, decel_desc = improvement_deceleration(eval_bpcs)
    gnorm_rising, gnorm_trend_desc = grad_norm_trend(grad_norms)

    result = {
        "config": config,
        "current_step": train_records[-1]["step"] if train_records else 0,
        "current_bpc": train_records[-1]["bpc"] if train_records else None,
        "current_lr": train_records[-1]["lr"] if train_records else None,
        "eval_bpcs": eval_bpcs,
        # Prefer the 200-batch final eval from summary.json when available;
        # fall back to best in-training eval (50-batch) for in-progress runs.
        "best_val_bpc": (summary["final_val_bpc"] if summary else
                         min((b for _, b in eval_bpcs), default=None)),
        "spikes": detect_loss_spikes(train_records),
        "summary": summary,
        "mol_records": mol_records,
        "train_curve": sample_curve(
            [(r["step"], r["bpc"]) for r in train_records if "bpc" in r], n=25
        ),
        "grad_norm_curve": sample_curve(grad_norms, n=25),
        "recent_gnorm": recent_gnorm,
        "train_val_gaps": train_val_gaps,
        "status": status,
        "bpc_rate": bpc_rate,
        "is_decelerating": is_decelerating,
        "decel_desc": decel_desc,
        "gnorm_rising": gnorm_rising,
        "gnorm_trend_desc": gnorm_trend_desc,
        "progress_pct": 100.0 * (train_records[-1]["step"] if train_records else 0) / TOTAL_STEPS,
    }

    if mol_records:
        latest_mol = mol_records[-1]["mol_stats"]
        result["mol_balance"] = {s["layer"]: s["expert_balance"] for s in latest_mol}
        result["mol_counts"]  = {s["layer"]: s["expert_counts"]  for s in latest_mol}
    else:
        result["mol_balance"] = {}
        result["mol_counts"]  = {}

    return result


# ── report generation ─────────────────────────────────────────────────────────

STATUS_ICON = {
    "improving":        "↓",
    "plateaued":        "→",
    "diverging":        "↑ ⚠",
    "insufficient data": "…",
}

# Expected outcome per config: (beat_baseline, description)
_CONFIG_HYPOTHESIS = {
    "baseline": (None,  "Control. Vanilla transformer. Target ~1.2–1.4 BPC at 50k steps."),
    "mhc":      (True,  "Should beat baseline: mHC multi-stream residual adds representational diversity."),
    "mol":      (True,  "Should beat baseline: MoL sparse experts add capacity without proportional cost."),
    "compose":  (True,  "Should beat both mhc and mol: additive gains expected if components are orthogonal."),
}


def key_findings(analyses):
    """Return list of finding strings derived from cross-config results."""
    findings = []
    complete = [a for a in analyses if a["best_val_bpc"] is not None]
    if not complete:
        return findings

    ranked = sorted(complete, key=lambda a: a["best_val_bpc"])
    baseline = next((a for a in analyses if a["config"] == "baseline"), None)
    mol     = next((a for a in analyses if a["config"] == "mol"), None)
    mhc     = next((a for a in analyses if a["config"] == "mhc"), None)
    compose = next((a for a in analyses if a["config"] == "compose"), None)

    # Winner
    winner = ranked[0]
    if baseline and winner["config"] != "baseline":
        delta = baseline["best_val_bpc"] - winner["best_val_bpc"]
        findings.append(
            f"**Best config: `{winner['config']}`** at {winner['best_val_bpc']:.4f} BPC "
            f"(+{delta:.4f} over baseline)."
        )
    elif baseline:
        findings.append(
            f"**No config beat baseline** ({baseline['best_val_bpc']:.4f} BPC). "
            f"All modifications regressed or were neutral."
        )

    # mHC vs baseline
    if mhc and baseline and mhc["best_val_bpc"] and baseline["best_val_bpc"]:
        delta = mhc["best_val_bpc"] - baseline["best_val_bpc"]
        if delta > 0.001:
            findings.append(
                f"**mHC regressed vs baseline** ({delta:+.4f} BPC). "
                f"Architecture may need LR tuning — see grad norm trend."
            )
        elif delta < -0.001:
            findings.append(
                f"**mHC improved vs baseline** ({delta:+.4f} BPC). "
                f"Multi-stream residual provides measurable benefit at this scale."
            )
        else:
            findings.append(f"**mHC neutral vs baseline** ({delta:+.4f} BPC, within noise).")

    # Composition penalty/benefit
    if mol and compose and mol["best_val_bpc"] and compose["best_val_bpc"]:
        penalty = compose["best_val_bpc"] - mol["best_val_bpc"]
        if penalty > 0.001:
            findings.append(
                f"**Composition penalty: compose is {penalty:.4f} BPC worse than mol alone.** "
                f"mHC interferes with MoL when combined — investigate before scaling."
            )
        elif penalty < -0.001:
            findings.append(
                f"**Composition benefit**: compose is {abs(penalty):.4f} BPC better than mol alone — "
                f"components are complementary."
            )
        else:
            findings.append(
                f"**Composition neutral** (compose vs mol: {penalty:+.4f} BPC). "
                f"mHC adds no harm but no gain on top of MoL."
            )

    # Hypothesis validation
    for a in complete:
        expected_better, _ = _CONFIG_HYPOTHESIS.get(a["config"], (None, ""))
        if expected_better is None or not baseline:
            continue
        actual_better = (a["best_val_bpc"] < baseline["best_val_bpc"] - 0.001)
        if expected_better and not actual_better:
            findings.append(
                f"**`{a['config']}` hypothesis not confirmed**: expected to beat baseline, did not."
            )

    # Alerts
    for a in analyses:
        if a["gnorm_rising"] and a["gnorm_trend_desc"]:
            findings.append(f"**⚠ `{a['config']}` grad norm trending up**: {a['gnorm_trend_desc']}.")
        if a["spikes"]:
            findings.append(f"**⚠ `{a['config']}` had {len(a['spikes'])} loss spike(s)**.")

    return findings


def next_steps(analyses):
    """Return list of actionable next-step strings derived from current results."""
    steps = []
    complete  = [a for a in analyses if a["summary"] is not None]
    running   = [a for a in analyses if a["summary"] is None and a["current_step"] > 0]

    if running:
        for a in running:
            steps.append(f"Wait for `{a['config']}` to finish ({a['progress_pct']:.0f}% complete).")
        return steps

    if len(complete) < len(CONFIGS):
        steps.append("Start remaining configs: `bash run_experiments.sh`.")
        return steps

    # All done — phase 1 complete
    mhc     = next((a for a in analyses if a["config"] == "mhc"), None)
    mol     = next((a for a in analyses if a["config"] == "mol"), None)
    compose = next((a for a in analyses if a["config"] == "compose"), None)

    if mhc and mhc["gnorm_rising"]:
        steps.append(
            "**Diagnose mHC optimization**: rerun `mhc` with `--max_lr 1.5e-4` and `--max_lr 7.5e-5`. "
            "Rising grad norm (1.29×) suggests Sinkhorn constraint is interfering. "
            "A clean mHC result is needed before trusting compose."
        )

    if mol and compose and mol["best_val_bpc"] and compose["best_val_bpc"]:
        penalty = compose["best_val_bpc"] - mol["best_val_bpc"]
        if penalty > 0.001:
            steps.append(
                f"**Investigate composition penalty** ({penalty:.4f} BPC): "
                "try `compose` with a per-component LR (mHC params at half rate). "
                "Or check whether mHC's Sinkhorn normalization is disrupting MoL routing."
            )

    steps.append(
        "**Plan Phase 2**: HDC toy integration — implement Zone E (encoder + router) and "
        "Zone D (EMA de-chunker) around the existing mol inner network at R=4, 5M params. "
        "This is the A1 gate ablation at toy scale."
    )

    return steps


def render_report(analyses):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = []

    lines += [
        "# Phase 1 Training Report",
        "",
        f"_Generated {now} — updated every 30s by `utils/reporter.py`_",
        "",
    ]

    # ── Cross-config comparison (top — highest signal) ────────────────────────
    complete = [a for a in analyses if a["best_val_bpc"] is not None]
    ranked   = sorted(complete, key=lambda a: a["best_val_bpc"])
    baseline_bpc = next((a["best_val_bpc"] for a in ranked if a["config"] == "baseline"), None)

    if complete:
        lines += ["## Results", ""]
        lines.append("| Rank | Config | Best Val BPC | vs baseline | Params | Time |")
        lines.append("|------|--------|--------------|-------------|--------|------|")
        for i, a in enumerate(ranked, 1):
            vs     = f"{a['best_val_bpc'] - baseline_bpc:+.4f}" if baseline_bpc else "—"
            params = f"{a['summary']['n_params']:,}" if a["summary"] else "—"
            mins   = f"{a['summary']['elapsed_seconds']/60:.0f}m" if a["summary"] else "—"
            lines.append(f"| {i} | `{a['config']}` | {a['best_val_bpc']:.4f} | {vs} | {params} | {mins} |")
        lines += [""]

    # ── Progress summary (for in-progress runs) ───────────────────────────────
    running = [a for a in analyses if a["summary"] is None]
    if running:
        lines += ["## In Progress", ""]
        lines.append("| Config | Progress | Train BPC | Best Val BPC | Grad Norm | Status |")
        lines.append("|--------|----------|-----------|--------------|-----------|--------|")
        for a in running:
            prog  = f"{a['progress_pct']:.0f}% ({a['current_step']:,}/{TOTAL_STEPS:,})"
            cur   = f"{a['current_bpc']:.4f}" if a['current_bpc'] is not None else "—"
            best  = f"{a['best_val_bpc']:.4f}" if a['best_val_bpc'] is not None else "—"
            gnorm = f"{a['recent_gnorm']:.3f}" if a['recent_gnorm'] is not None else "—"
            icon  = STATUS_ICON.get(a["status"], "?")
            lines.append(f"| `{a['config']}` | {prog} | {cur} | {best} | {gnorm} | {icon} {a['status']} |")
        lines += [""]

    # ── Key findings ──────────────────────────────────────────────────────────
    findings = key_findings(analyses)
    if findings:
        lines += ["## Key Findings", ""]
        for f in findings:
            lines.append(f"- {f}")
        lines += [""]

    # ── Recommended next steps ────────────────────────────────────────────────
    steps = next_steps(analyses)
    if steps:
        lines += ["## Next Steps", ""]
        for i, s in enumerate(steps, 1):
            lines.append(f"{i}. {s}")
        lines += [""]

    lines += ["---", ""]

    # ── Per-config detail sections ────────────────────────────────────────────
    for a in analyses:
        is_complete = a["summary"] is not None
        lines += [f"## `{a['config']}`", ""]

        # Hypothesis
        _, hyp_desc = _CONFIG_HYPOTHESIS.get(a["config"], (None, ""))
        lines.append(f"_Hypothesis: {hyp_desc}_")
        lines.append("")

        # Convergence line
        if a["bpc_rate"] is not None:
            direction = "improvement" if a["bpc_rate"] < 0 else "degradation"
            lines.append(f"**Convergence**: {a['status']} — {abs(a['bpc_rate']):.4f} BPC {direction}/1k steps")
        else:
            lines.append(f"**Convergence**: {a['status']}")

        if a["is_decelerating"] and a["decel_desc"]:
            lines.append(f"**⚠ Asymptotic approach detected**: {a['decel_desc']}")
        if a["gnorm_rising"] and a["gnorm_trend_desc"]:
            lines.append(f"**⚠ Grad norm trending up**: {a['gnorm_trend_desc']}")

        lines.append("")

        # Val BPC + train/val gap table
        if a["eval_bpcs"]:
            has_gap = bool(a["train_val_gaps"])
            lines.append("**Validation BPC progression:**")
            lines.append("")
            if has_gap:
                lines.append("| Step | Train BPC | Val BPC | Gap (val−train) | Δ Val BPC |")
                lines.append("|------|-----------|---------|-----------------|-----------|")
            else:
                lines.append("| Step | Val BPC | Δ from prev |")
                lines.append("|------|---------|-------------|")

            prev_val = None
            gap_map = {g[0]: g for g in a["train_val_gaps"]}
            for step, val_bpc in a["eval_bpcs"]:
                delta = f"{val_bpc - prev_val:+.4f}" if prev_val is not None else "—"
                marker = " ★" if val_bpc == a["best_val_bpc"] else ""
                if has_gap and step in gap_map:
                    _, t_bpc, v_bpc, gap = gap_map[step]
                    gap_flag = " ⚠" if gap < -0.05 else ""
                    lines.append(f"| {step:>6,} | {t_bpc:.4f} | {v_bpc:.4f}{marker} | {gap:+.4f}{gap_flag} | {delta} |")
                else:
                    lines.append(f"| {step:>6,} | {val_bpc:.4f}{marker} | {delta} |")
                prev_val = val_bpc
            lines.append("")

        # Training BPC curve — compact summary for completed runs, full chart for in-progress
        if a["train_curve"]:
            if is_complete:
                pts = a["train_curve"]
                n   = len(pts)
                q   = [pts[0], pts[n//4], pts[n//2], pts[3*n//4], pts[-1]]
                segments = " → ".join(f"{bpc:.4f} (step {step:,})" for step, bpc in q)
                lines.append(f"**Training BPC**: {segments}")
                lines.append("")
            else:
                lines.append("**Training BPC (sampled):**")
                lines.append("```")
                for step, bpc in a["train_curve"]:
                    bar = "█" * min(max(0, int((bpc - 1.0) * 40)), 60)
                    lines.append(f"  {step:>6,} | {bpc:.4f} | {bar}")
                lines.append("```")
                lines.append("")

        # Grad norm curve — always show if data exists (key diagnostic)
        if a["grad_norm_curve"]:
            lines.append("**Gradient norm (sampled):**")
            lines.append("```")
            max_gnorm = max(g for _, g in a["grad_norm_curve"])
            for step, gnorm in a["grad_norm_curve"]:
                bar = "█" * min(int(gnorm / max(max_gnorm, 1e-8) * 40), 40)
                lines.append(f"  {step:>6,} | {gnorm:.4f} | {bar}")
            lines.append("```")
            lines.append("")

        # Spikes
        if a["spikes"]:
            lines.append(f"**⚠ Loss spikes** ({len(a['spikes'])} events):")
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
                lines.append(f"| {layer:>5} | {bal:.3f}{flag} | {pcts} |")
            lines.append("")

        if a["summary"]:
            s = a["summary"]
            lines += [
                f"**Training complete** — {s['elapsed_seconds']/60:.1f} min, "
                f"{s['n_params']:,} params, best val BPC **{s['best_val_bpc']:.4f}**",
                "",
            ]

        lines += ["---", ""]

    # ── Notes for Claude Code (reference, bottom) ─────────────────────────────
    lines += [
        "## Notes for Claude Code",
        "",
        "- **BPC**: lower is better. Results table uses `final_val_bpc` from summary.json "
          "(200-batch eval) for completed runs; in-progress shows best in-training val BPC (50-batch).",
        "- **Grad norm**: decrease or plateau is healthy. Sustained rise >1.2× early avg = ⚠ optimization issue.",
        "- **Train/val gap**: val−train BPC. Positive is normal. Negative (train > val) = data or eval bug.",
        "- **Composition penalty**: if compose > mol, mHC is hurting MoL — diagnose before scaling.",
        "- **Balance score**: MoL expert utilization uniformity. <0.5 = expert collapse.",
        "- To regenerate: `python utils/reporter.py`",
        "- To watch live: `python utils/reporter.py --watch`",
        "",
    ]

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def run_once():
    os.makedirs(CKPT_DIR, exist_ok=True)
    analyses = [analyze_config(c) for c in CONFIGS]
    analyses = [a for a in analyses if a is not None]
    if not analyses:
        print("No training data found in checkpoints/. Run training first.")
        return
    report = render_report(analyses)
    # Write to a temp file then rename — atomic on POSIX, prevents partial reads
    tmp = REPORT_PATH + ".tmp"
    with open(tmp, "w") as f:
        f.write(report)
    os.replace(tmp, REPORT_PATH)
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
            run_once()
    else:
        run_once()


if __name__ == "__main__":
    main()
