"""
Training reporter — runs as a separate process, reads JSONL logs written by
train.py, and generates checkpoints/report.md for Claude Code and humans.

Usage:
  python utils/reporter.py                  # one-shot report
  python utils/reporter.py --watch          # poll every 30s until killed
  python utils/reporter.py --watch --interval 60
"""

import argparse
import glob as _glob
import json
import math
import os
import time
from datetime import datetime, timezone

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
REPORT_PATH = os.path.join(CKPT_DIR, "report.md")

PHASE1_CONFIGS  = ["baseline", "mhc", "mol", "compose"]
PHASE2_CONFIGS  = [
    "hdc_rulebased", "hdc_gate", "hdc_stride", "hdc_r2", "hdc_r8",
    "hdc_e2e_isolated", "hdc_upcycle_stride", "hdc_upcycle_gate",
]
CONFIGS = PHASE1_CONFIGS + PHASE2_CONFIGS

# Total training steps per config
_TOTAL_STEPS = {c: 100000 for c in PHASE1_CONFIGS + PHASE2_CONFIGS}
_TOTAL_STEPS["hdc_upcycle_stride"] = 50000
_TOTAL_STEPS["hdc_upcycle_gate"]   = 50000
TOTAL_STEPS = 100000  # kept for backwards compat

# Configs that use a frozen mol inner (upcycle)
_UPCYCLE_CONFIGS = {"hdc_upcycle_stride", "hdc_upcycle_gate"}


# ── data loading ─────────────────────────────────────────────────────────────

def find_jsonl(config):
    """Return (path, seed) for the JSONL file for this config, or (None, None)."""
    matches = _glob.glob(os.path.join(CKPT_DIR, f"{config}_seed*.jsonl"))
    if not matches:
        return None, None
    # If multiple seeds exist, sort and take the first (lowest seed number)
    matches.sort()
    path = matches[0]
    name = os.path.splitext(os.path.basename(path))[0]  # e.g. "baseline_seed42"
    try:
        seed = int(name.split("_seed")[-1])
    except ValueError:
        seed = None
    return path, seed


def load_jsonl(config):
    path, _ = find_jsonl(config)
    if path is None:
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
    _, seed = find_jsonl(config)
    records = load_jsonl(config)
    summary = load_summary(config)

    train_records = [r for r in records if "type" not in r]
    eval_records  = [r for r in records if r.get("type") == "eval"]
    mol_records   = [r for r in records if r.get("type") == "mol"]
    hdc_records   = [r for r in records if r.get("type") == "hdc"]
    pos_records   = [r for r in records if r.get("type") == "pos_loss"]

    if not records:
        return None

    eval_bpcs = [(r["step"], r["val_bpc"]) for r in eval_records]

    grad_norms = [(r["step"], r["grad_norm"]) for r in train_records if "grad_norm" in r]
    recent_gnorm = None
    if grad_norms:
        recent = [g for _, g in grad_norms[-10:]]
        recent_gnorm = sum(recent) / len(recent)

    train_val_gaps = []
    for step, val_bpc in eval_bpcs:
        t_bpc = train_bpc_at_step(train_records, step)
        if t_bpc is not None:
            train_val_gaps.append((step, t_bpc, val_bpc, val_bpc - t_bpc))

    status, bpc_rate = convergence_status(eval_bpcs)
    is_decelerating, decel_desc = improvement_deceleration(eval_bpcs)
    gnorm_rising, gnorm_trend_desc = grad_norm_trend(grad_norms)

    # HDC-specific metrics (Phase 2 only)
    hdc_health = {}
    if hdc_records:
        recent_hdc = hdc_records[-10:]
        hdc_health["compression_ratio"] = sum(r.get("compression_ratio", 0) for r in recent_hdc) / len(recent_hdc)
        hdc_health["loss_comp"]         = sum(r.get("loss_comp", 0) for r in recent_hdc) / len(recent_hdc)
        hdc_health["boundary_entropy"]  = sum(r.get("boundary_entropy", 0) for r in recent_hdc) / len(recent_hdc)
        hdc_health["ratio_curve"]       = sample_curve(
            [(r["step"], r["compression_ratio"]) for r in hdc_records if "compression_ratio" in r], n=15
        )
        hdc_health["entropy_curve"]     = sample_curve(
            [(r["step"], r["boundary_entropy"]) for r in hdc_records if "boundary_entropy" in r], n=15
        )
        # Compression ratio stability: std of recent ratios
        recent_ratios = [r.get("compression_ratio", 0) for r in recent_hdc]
        mean_r = sum(recent_ratios) / len(recent_ratios)
        hdc_health["ratio_std"] = math.sqrt(sum((r - mean_r)**2 for r in recent_ratios) / len(recent_ratios))
        # Target ratio from config name
        if "r2" in config:
            hdc_health["target_ratio"] = 0.5
        elif "r8" in config:
            hdc_health["target_ratio"] = 0.125
        else:
            hdc_health["target_ratio"] = 0.25

    pos_health = {}
    if pos_records:
        latest = pos_records[-1]
        pos_health["boundary_bpc"] = latest.get("boundary_bpc")
        pos_health["midchunk_bpc"] = latest.get("midchunk_bpc")
        pos_health["step"]         = latest.get("step")
        pos_health["curve"]        = [
            (r["step"], r.get("boundary_bpc"), r.get("midchunk_bpc"))
            for r in pos_records if "boundary_bpc" in r
        ]

    phase = 1 if config in PHASE1_CONFIGS else 2
    total_steps = _TOTAL_STEPS.get(config, TOTAL_STEPS)

    result = {
        "config": config,
        "seed": seed if seed is not None else (summary.get("seed") if summary else None),
        "phase": phase,
        "current_step": train_records[-1]["step"] if train_records else 0,
        "current_bpc":  train_records[-1]["bpc"]  if train_records else None,
        "current_lr":   train_records[-1]["lr"]   if train_records else None,
        "eval_bpcs": eval_bpcs,
        "best_val_bpc": (summary["final_val_bpc"] if summary else
                         min((b for _, b in eval_bpcs), default=None)),
        "spikes": detect_loss_spikes(train_records),
        "summary": summary,
        "mol_records": mol_records,
        "train_curve": sample_curve(
            [(r["step"], r["bpc"]) for r in train_records if "bpc" in r], n=25
        ),
        "grad_norm_curve": sample_curve(grad_norms, n=25),
        "recent_gnorm":  recent_gnorm,
        "train_val_gaps": train_val_gaps,
        "status":         status,
        "bpc_rate":       bpc_rate,
        "is_decelerating": is_decelerating,
        "decel_desc":      decel_desc,
        "gnorm_rising":    gnorm_rising,
        "gnorm_trend_desc": gnorm_trend_desc,
        "progress_pct": 100.0 * (train_records[-1]["step"] if train_records else 0) / total_steps,
        "hdc_health": hdc_health,
        "pos_health":  pos_health,
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

# Expected outcome per config: (beat_baseline_config, description)
# beat_baseline_config = name of the config this should beat, or None
_CONFIG_HYPOTHESIS = {
    # Phase 1
    "baseline":         (None,           "Control. Vanilla transformer. Target ~3.5–3.6 BPC at 100k steps."),
    "mhc":              ("baseline",     "Should beat baseline: mHC multi-stream residual adds representational diversity."),
    "mol":              ("baseline",     "Should beat baseline: MoL sparse experts add capacity without proportional cost."),
    "compose":          ("mol",          "Should beat mol: additive gains if mHC and MoL are orthogonal."),
    # Phase 2 — all compared against mol (best Phase 1 config, inner network for HDC)
    "hdc_rulebased":    ("mol",          "Pipeline validation. Rule-based cosine routing. Should match or beat mol if HDC helps at all."),
    "hdc_gate":         ("hdc_rulebased","Learned e2e routing (H-Net style). Should beat hdc_rulebased if content-aware boundaries help."),
    "hdc_stride":       (None,           "Fixed-stride lower bound. Expected to lose to hdc_rulebased. Shows value of any routing signal."),
    "hdc_r2":           ("hdc_gate",     "R=2 compression sweep. Less compression — may trade throughput for quality."),
    "hdc_r8":           ("hdc_gate",     "R=8 compression sweep. More compression — tests limits of concept formation."),
    "hdc_e2e_isolated":   (None,           "A5 comparison: gradient isolation. Expected near-random boundaries. Should match hdc_stride."),
    # Upcycle — frozen mol inner, 50k steps
    "hdc_upcycle_stride": (None,           "Upcycle lower bound: fixed-stride + frozen mol inner. Validates Zone E/D can learn around frozen weights."),
    "hdc_upcycle_gate":   ("hdc_upcycle_stride", "Upcycle with learned routing. Should beat upcycle_stride if content-aware boundaries help the frozen inner."),
}


def key_findings(analyses):
    """Return list of finding strings derived from cross-config results."""
    findings = []
    complete = [a for a in analyses if a["best_val_bpc"] is not None]
    if not complete:
        return findings

    ranked   = sorted(complete, key=lambda a: a["best_val_bpc"])
    baseline = next((a for a in analyses if a["config"] == "baseline"), None)
    mol      = next((a for a in analyses if a["config"] == "mol"),      None)
    mhc      = next((a for a in analyses if a["config"] == "mhc"),      None)
    compose  = next((a for a in analyses if a["config"] == "compose"),  None)

    p1_complete = [a for a in complete if a["phase"] == 1]

    # Overall winner
    winner = ranked[0]
    if baseline and winner["config"] != "baseline":
        delta = baseline["best_val_bpc"] - winner["best_val_bpc"]
        findings.append(
            f"**Best overall: `{winner['config']}`** at {winner['best_val_bpc']:.4f} BPC "
            f"(+{delta:.4f} over baseline)."
        )
    elif baseline and len(complete) > 1:
        findings.append(
            f"**No config beat baseline** ({baseline['best_val_bpc']:.4f} BPC)."
        )

    # Phase 1: mHC vs baseline
    if mhc and baseline and mhc["best_val_bpc"] and baseline["best_val_bpc"]:
        delta = mhc["best_val_bpc"] - baseline["best_val_bpc"]
        if delta > 0.001:
            findings.append(
                f"**mHC regressed vs baseline** ({delta:+.4f} BPC) — see grad norm trend."
            )
        elif delta < -0.001:
            findings.append(f"**mHC improved vs baseline** ({delta:+.4f} BPC).")
        else:
            findings.append(f"**mHC neutral vs baseline** ({delta:+.4f} BPC).")

    # Phase 1: composition
    if mol and compose and mol["best_val_bpc"] and compose["best_val_bpc"]:
        penalty = compose["best_val_bpc"] - mol["best_val_bpc"]
        if penalty > 0.001:
            findings.append(
                f"**Composition penalty** ({penalty:+.4f} BPC): compose worse than mol alone — "
                f"mHC interferes with MoL."
            )
        elif penalty < -0.001:
            findings.append(f"**Composition benefit** ({penalty:+.4f} BPC): mHC+MoL better than MoL alone.")
        else:
            findings.append(f"**Composition neutral** ({penalty:+.4f} BPC).")

    # Phase 2: A1 gate result
    hdc_ruled = next((a for a in analyses if a["config"] == "hdc_rulebased"), None)
    hdc_gate  = next((a for a in analyses if a["config"] == "hdc_gate"),      None)
    if mol and hdc_ruled and hdc_ruled["best_val_bpc"] and mol["best_val_bpc"]:
        delta = mol["best_val_bpc"] - hdc_ruled["best_val_bpc"]
        if delta > 0.001:
            findings.append(
                f"**A1 gate: HDC helps** — hdc_rulebased beats mol by {delta:.4f} BPC. "
                f"Chunking adds value even with rule-based boundaries."
            )
        elif delta < -0.001:
            findings.append(
                f"**A1 gate: HDC regressed** ({-delta:.4f} BPC worse than mol). "
                f"Pipeline or compression ratio may need tuning — see Section 14.1 diagnostics."
            )
        else:
            findings.append(f"**A1 gate: HDC neutral vs mol** ({delta:+.4f} BPC).")

    if hdc_gate and hdc_ruled and hdc_gate["best_val_bpc"] and hdc_ruled["best_val_bpc"]:
        delta = hdc_ruled["best_val_bpc"] - hdc_gate["best_val_bpc"]
        if delta > 0.001:
            findings.append(
                f"**Learned routing helps** — hdc_gate beats hdc_rulebased by {delta:.4f} BPC. "
                f"Content-aware boundaries outperform cosine threshold."
            )
        elif delta < -0.001:
            findings.append(
                f"**Learned routing regressed** ({-delta:.4f} BPC worse than rule-based). "
                f"Router may not be learning content-aware boundaries."
            )

    # Phase 2: HDC health alerts
    for a in [a for a in analyses if a["phase"] == 2]:
        h = a.get("hdc_health", {})
        if h:
            target = h.get("target_ratio", 0.25)
            ratio  = h.get("compression_ratio", target)
            if abs(ratio - target) > 0.3 * target:
                findings.append(
                    f"**⚠ `{a['config']}` compression ratio unstable**: "
                    f"{ratio:.3f} vs target {target:.3f} — raise λ_comp."
                )
        p = a.get("pos_health", {})
        if p and p.get("boundary_bpc") and p.get("midchunk_bpc"):
            u_gap = p["midchunk_bpc"] - p["boundary_bpc"]
            if u_gap > 0.15:
                findings.append(
                    f"**⚠ `{a['config']}` U-shaped loss**: mid-chunk BPC {u_gap:+.4f} above "
                    f"boundary BPC — gated residual may not be effective."
                )

    # Phase 1 hypothesis validation
    for a in p1_complete:
        ref_cfg, _ = _CONFIG_HYPOTHESIS.get(a["config"], (None, ""))
        if ref_cfg is None:
            continue
        ref = next((x for x in analyses if x["config"] == ref_cfg), None)
        if ref and ref["best_val_bpc"] and a["best_val_bpc"]:
            if a["best_val_bpc"] > ref["best_val_bpc"] + 0.001:
                findings.append(
                    f"**`{a['config']}` hypothesis not confirmed**: "
                    f"expected to beat `{ref_cfg}`, did not ({a['best_val_bpc']:.4f} vs {ref['best_val_bpc']:.4f})."
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

    p1_analyses  = [a for a in analyses if a["phase"] == 1]
    p2_analyses  = [a for a in analyses if a["phase"] == 2]
    p1_running   = [a for a in p1_analyses if a["summary"] is None and a["current_step"] > 0]
    p2_running   = [a for a in p2_analyses if a["summary"] is None and a["current_step"] > 0]
    p1_complete  = [a for a in p1_analyses if a["summary"] is not None]
    p2_complete  = [a for a in p2_analyses if a["summary"] is not None]

    # Still running
    for a in p1_running + p2_running:
        steps.append(f"Wait for `{a['config']}` to complete ({a['progress_pct']:.0f}%).")
    if p1_running or p2_running:
        return steps

    # Phase 1 not started
    if not p1_complete and not p2_complete:
        steps.append("Run all experiments: `bash run_experiments.sh`")
        return steps

    # Phase 1 incomplete
    p1_done_names = {a["config"] for a in p1_complete}
    p1_remaining  = [c for c in PHASE1_CONFIGS if c not in p1_done_names]
    if p1_remaining:
        steps.append(f"Complete Phase 1: `bash run_experiments.sh phase1` "
                     f"(remaining: {', '.join(p1_remaining)})")
        return steps

    # Phase 1 complete — check mHC diagnostic
    mhc = next((a for a in analyses if a["config"] == "mhc"), None)
    if mhc and mhc["gnorm_rising"]:
        steps.append(
            "**Diagnose mHC**: rerun with `--max_lr 1.5e-4 --total_steps 25000` and "
            "`--max_lr 7.5e-5`. Rising grad norm suggests Sinkhorn interference."
        )

    # Phase 2 not started
    if not p2_complete:
        steps.append("Start Phase 2: `bash run_experiments.sh phase2`")
        return steps

    # Phase 2 incomplete
    p2_done_names = {a["config"] for a in p2_complete}
    p2_remaining  = [c for c in PHASE2_CONFIGS if c not in p2_done_names]
    if p2_remaining:
        steps.append(f"Continue Phase 2: `bash run_experiments.sh phase2` "
                     f"(remaining: {', '.join(p2_remaining)})")
        return steps

    # Everything done
    mol      = next((a for a in analyses if a["config"] == "mol"),              None)
    hdc_gate = next((a for a in analyses if a["config"] == "hdc_gate"),         None)
    upc_gate = next((a for a in analyses if a["config"] == "hdc_upcycle_gate"), None)

    if hdc_gate and mol and hdc_gate["best_val_bpc"] and mol["best_val_bpc"]:
        if hdc_gate["best_val_bpc"] < mol["best_val_bpc"] - 0.001:
            steps.append(
                "**Phase 2 succeeded** — HDC improves over mol. "
                "Next: scale to 1–3B params for A1 validation at production scale."
            )
        else:
            steps.append(
                "**HDC did not improve over mol at toy scale.** "
                "Check upcycle configs (hdc_upcycle_stride, hdc_upcycle_gate) for comparison."
            )

    if upc_gate and mol and upc_gate["best_val_bpc"] and mol["best_val_bpc"]:
        if upc_gate["best_val_bpc"] < mol["best_val_bpc"] - 0.001:
            steps.append(
                f"**Upcycle succeeded** — hdc_upcycle_gate ({upc_gate['best_val_bpc']:.4f}) "
                f"beats mol ({mol['best_val_bpc']:.4f}). HDC pipeline is viable with warm-started inner."
            )
        else:
            steps.append(
                f"**Upcycle did not beat mol** ({upc_gate['best_val_bpc']:.4f} vs {mol['best_val_bpc']:.4f}). "
                "HDC compression overhead exceeds representational benefit at this scale."
            )

    return steps


def render_report(analyses):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = []

    lines += [
        "# Phase 1 Training Report",
        "",
        f"_Generated {now} — updated every 30s by `utils/reporter.py`_",
        "",
    ]

    # ── Results tables (top — highest signal) ────────────────────────────────
    p1_complete  = [a for a in analyses if a["phase"] == 1 and a["best_val_bpc"] is not None]
    p2_complete  = [a for a in analyses if a["phase"] == 2 and a["best_val_bpc"] is not None]
    baseline_bpc = next((a["best_val_bpc"] for a in p1_complete if a["config"] == "baseline"), None)
    mol_bpc      = next((a["best_val_bpc"] for a in p1_complete if a["config"] == "mol"),      None)

    if p1_complete:
        lines += ["## Phase 1 Results", ""]
        lines.append("| Rank | Config | Seed | Best Val BPC | vs baseline | Params | Time |")
        lines.append("|------|--------|------|--------------|-------------|--------|------|")
        for i, a in enumerate(sorted(p1_complete, key=lambda x: x["best_val_bpc"]), 1):
            vs     = f"{a['best_val_bpc'] - baseline_bpc:+.4f}" if baseline_bpc else "—"
            params = f"{a['summary']['n_params']:,}" if a["summary"] else "—"
            mins   = f"{a['summary']['elapsed_seconds']/60:.0f}m" if a["summary"] else "—"
            seed   = str(a["seed"]) if a["seed"] is not None else "—"
            lines.append(f"| {i} | `{a['config']}` | {seed} | {a['best_val_bpc']:.4f} | {vs} | {params} | {mins} |")
        lines += [""]

    if p2_complete:
        lines += ["## Phase 2 Results", ""]
        lines.append("| Rank | Config | Seed | Type | Best Val BPC | vs mol | R | Params | Time |")
        lines.append("|------|--------|------|------|--------------|--------|---|--------|------|")
        for i, a in enumerate(sorted(p2_complete, key=lambda x: x["best_val_bpc"]), 1):
            vs     = f"{a['best_val_bpc'] - mol_bpc:+.4f}" if mol_bpc else "—"
            R      = a["summary"].get("R", "?") if a["summary"] else "?"
            params = f"{a['summary']['n_params']:,}" if a["summary"] else "—"
            mins   = f"{a['summary']['elapsed_seconds']/60:.0f}m" if a["summary"] else "—"
            typ    = "upcycle" if a["config"] in _UPCYCLE_CONFIGS else "e2e"
            seed   = str(a["seed"]) if a["seed"] is not None else "—"
            lines.append(f"| {i} | `{a['config']}` | {seed} | {typ} | {a['best_val_bpc']:.4f} | {vs} | {R} | {params} | {mins} |")
        lines += [""]

    # ── Progress summary (for in-progress runs) ───────────────────────────────
    running = [a for a in analyses if a["summary"] is None and a["current_step"] > 0]
    if running:
        lines += ["## In Progress", ""]
        lines.append("| Config | Phase | Progress | Train BPC | Best Val BPC | Grad Norm | Status |")
        lines.append("|--------|-------|----------|-----------|--------------|-----------|--------|")
        for a in running:
            tot   = _TOTAL_STEPS.get(a["config"], TOTAL_STEPS)
            prog  = f"{a['progress_pct']:.0f}% ({a['current_step']:,}/{tot:,})"
            cur   = f"{a['current_bpc']:.4f}" if a['current_bpc'] is not None else "—"
            best  = f"{a['best_val_bpc']:.4f}" if a['best_val_bpc'] is not None else "—"
            gnorm = f"{a['recent_gnorm']:.3f}" if a['recent_gnorm'] is not None else "—"
            icon  = STATUS_ICON.get(a["status"], "?")
            lines.append(f"| `{a['config']}` | {a['phase']} | {prog} | {cur} | {best} | {gnorm} | {icon} {a['status']} |")
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
                    if math.isfinite(bpc):
                        bar = "█" * min(max(0, int((bpc - 1.0) * 40)), 60)
                    else:
                        bar = "nan"
                    lines.append(f"  {step:>6,} | {bpc:.4f} | {bar}")
                lines.append("```")
                lines.append("")

        # Grad norm curve — always show if data exists (key diagnostic)
        if a["grad_norm_curve"]:
            lines.append("**Gradient norm (sampled):**")
            lines.append("```")
            finite = [g for _, g in a["grad_norm_curve"] if math.isfinite(g)]
            max_gnorm = max(finite, default=1.0)
            for step, gnorm in a["grad_norm_curve"]:
                if math.isfinite(gnorm):
                    bar = "█" * min(int(gnorm / max(max_gnorm, 1e-8) * 40), 40)
                else:
                    bar = "nan"
                lines.append(f"  {step:>6,} | {gnorm:.4f} | {bar}")
            lines.append("```")
            lines.append("")

        # Spikes
        if a["spikes"]:
            lines.append(f"**⚠ Loss spikes** ({len(a['spikes'])} events):")
            for step, bpc in a["spikes"][:5]:
                lines.append(f"  - step {step:,}: BPC {bpc:.4f}")
            lines.append("")

        # HDC health metrics (Phase 2 only)
        h = a.get("hdc_health", {})
        if h:
            target = h.get("target_ratio", 0.25)
            ratio  = h.get("compression_ratio", 0)
            std    = h.get("ratio_std", 0)
            ent    = h.get("boundary_entropy", 0)
            comp_l = h.get("loss_comp", 0)
            ratio_flag = " ⚠ UNSTABLE" if abs(ratio - target) > 0.3 * target else ""
            lines.append("**HDC Health:**")
            lines.append(f"- Compression ratio: {ratio:.3f} (target {target:.3f}, σ={std:.3f}){ratio_flag}")
            lines.append(f"- Boundary entropy: {ent:.3f} (lower = more decisive routing)")
            lines.append(f"- Compression loss: {comp_l:.5f}")
            lines.append("")

            if h.get("ratio_curve"):
                lines.append("**Compression ratio over training:**")
                lines.append("```")
                for step, r in h["ratio_curve"]:
                    bar = ("█" * min(int(r * 80), 40)) if math.isfinite(r) else "nan"
                    lines.append(f"  {step:>6,} | {r:.3f} | {bar}")
                lines.append("```")
                lines.append("")

        # Per-position loss (Phase 2 only)
        p = a.get("pos_health", {})
        if p and p.get("boundary_bpc") and p.get("midchunk_bpc"):
            b_bpc = p["boundary_bpc"]
            m_bpc = p["midchunk_bpc"]
            u_gap = m_bpc - b_bpc
            flag  = " ⚠ U-SHAPED LOSS" if u_gap > 0.15 else (" ✓ healthy" if u_gap < 0.05 else "")
            lines.append("**Per-position BPC (latest eval):**")
            lines.append(f"- Boundary tokens:  {b_bpc:.4f} BPC")
            lines.append(f"- Mid-chunk tokens: {m_bpc:.4f} BPC  (delta: {u_gap:+.4f}){flag}")
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

    render_agent_report(analyses)
    print(f"Agent report written to {AGENT_REPORT_PATH}")


AGENT_REPORT_PATH = os.path.join(CKPT_DIR, "report_agent.json")

def render_agent_report(analyses):
    """
    Compact JSON report for Claude Code consumption.
    No prose, no charts — only structured signals needed for analysis.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")

    # Reference BPCs for delta calculations
    ref = {}
    for cfg in ("baseline", "mol"):
        a = next((x for x in analyses if x["config"] == cfg), None)
        if a and a["best_val_bpc"] is not None:
            ref[cfg] = round(a["best_val_bpc"], 4)

    configs_out = []
    for a in analyses:
        flags = []
        if a["gnorm_rising"]:
            flags.append("grad_norm_rising")
        if a["is_decelerating"]:
            flags.append("improvement_decelerating")
        if a["spikes"]:
            flags.append(f"loss_spikes:{len(a['spikes'])}")
        h = a.get("hdc_health", {})
        if h:
            target = h.get("target_ratio", 0.25)
            ratio  = h.get("compression_ratio", target)
            if abs(ratio - target) > 0.3 * target:
                flags.append(f"compression_unstable:{ratio:.3f}_vs_{target:.3f}")
        p = a.get("pos_health", {})
        if p and p.get("boundary_bpc") and p.get("midchunk_bpc"):
            if p["midchunk_bpc"] - p["boundary_bpc"] > 0.15:
                flags.append("u_shaped_loss")

        entry = {
            "config":        a["config"],
            "seed":          a["seed"],
            "phase":         a["phase"],
            "type":          "upcycle" if a["config"] in _UPCYCLE_CONFIGS else ("hdc" if a["phase"] == 2 else "dense"),
            "status":        "complete" if a["summary"] else ("running" if a["current_step"] > 0 else "pending"),
            "progress":      round(a["progress_pct"] / 100, 3),
            "step":          a["current_step"],
            "total_steps":   _TOTAL_STEPS.get(a["config"], TOTAL_STEPS),
            "best_val_bpc":  round(a["best_val_bpc"], 4) if a["best_val_bpc"] else None,
            "train_bpc":     round(a["current_bpc"], 4) if a["current_bpc"] else None,
            "grad_norm":     round(a["recent_gnorm"], 3) if a["recent_gnorm"] else None,
            "convergence":   a["status"],
            "bpc_rate_per_1k": round(a["bpc_rate"], 5) if a["bpc_rate"] is not None else None,
            "n_params":      a["summary"]["n_params"] if a["summary"] else None,
            "elapsed_min":   round(a["summary"]["elapsed_seconds"] / 60, 1) if a["summary"] else None,
            "flags":         flags,
        }

        # Deltas vs reference configs
        if a["best_val_bpc"] is not None:
            if "baseline" in ref:
                entry["vs_baseline"] = round(a["best_val_bpc"] - ref["baseline"], 4)
            if "mol" in ref and a["phase"] == 2:
                entry["vs_mol"] = round(a["best_val_bpc"] - ref["mol"], 4)

        # HDC-specific signals
        if h:
            entry["hdc"] = {
                "compression_ratio": round(h.get("compression_ratio", 0), 3),
                "target_ratio":      round(h.get("target_ratio", 0.25), 3),
                "ratio_std":         round(h.get("ratio_std", 0), 4),
                "boundary_entropy":  round(h.get("boundary_entropy", 0), 3),
            }

        # Eval curve (step, val_bpc) — full precision, all evals
        if a["eval_bpcs"]:
            entry["eval_curve"] = [[s, round(b, 4)] for s, b in a["eval_bpcs"]]

        # Sampled train curve (step, bpc) — 10 points
        if a["train_curve"]:
            tc = a["train_curve"]
            stride = max(1, len(tc) // 10)
            entry["train_curve_sampled"] = [[s, round(b, 4)] for s, b in tc[::stride]]

        configs_out.append(entry)

    # Cross-config rankings (completed only)
    complete = [c for c in configs_out if c["status"] == "complete" and c["best_val_bpc"] is not None]
    rankings = [{"rank": i+1, "config": c["config"], "best_val_bpc": c["best_val_bpc"]}
                for i, c in enumerate(sorted(complete, key=lambda x: x["best_val_bpc"]))]

    # All active flags aggregated
    all_flags = [{"config": c["config"], "flags": c["flags"]}
                 for c in configs_out if c["flags"]]

    report = {
        "generated_utc": now,
        "reference_bpcs": ref,
        "summary": {
            "phase1": {
                "complete": sum(1 for c in configs_out if c["phase"] == 1 and c["status"] == "complete"),
                "running":  sum(1 for c in configs_out if c["phase"] == 1 and c["status"] == "running"),
                "pending":  sum(1 for c in configs_out if c["phase"] == 1 and c["status"] == "pending"),
            },
            "phase2": {
                "complete": sum(1 for c in configs_out if c["phase"] == 2 and c["status"] == "complete"),
                "running":  sum(1 for c in configs_out if c["phase"] == 2 and c["status"] == "running"),
                "pending":  sum(1 for c in configs_out if c["phase"] == 2 and c["status"] == "pending"),
            },
        },
        "rankings": rankings,
        "alerts": all_flags,
        "configs": configs_out,
    }

    tmp = AGENT_REPORT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, separators=(",", ":"))
    os.replace(tmp, AGENT_REPORT_PATH)


def main():
    parser = argparse.ArgumentParser(description="Training reporter")
    parser.add_argument("--watch", action="store_true",
                        help="Poll continuously until killed (Ctrl+C)")
    parser.add_argument("--interval", type=int, default=30,
                        help="Poll interval in seconds (default: 30)")
    parser.add_argument("--parent-pid", type=int, default=None,
                        help="Exit when this PID is no longer running")
    args = parser.parse_args()

    if args.watch:
        parent = args.parent_pid
        print(f"Watching checkpoints/ — updating report every {args.interval}s. Ctrl+C to stop."
              + (f" (watching PID {parent})" if parent else ""))
        try:
            while True:
                run_once()
                time.sleep(args.interval)
                if parent:
                    try:
                        os.kill(parent, 0)
                    except ProcessLookupError:
                        print(f"\nParent process {parent} exited — reporter shutting down.")
                        run_once()
                        break
        except KeyboardInterrupt:
            print("\nReporter stopped.")
            run_once()
    else:
        run_once()


if __name__ == "__main__":
    main()
