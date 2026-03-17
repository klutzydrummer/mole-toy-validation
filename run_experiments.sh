#!/bin/bash
# Run all MoLE toy validation experiments in sequence.
# Covers Phase 1 (4 configs) and Phase 2 (6 HDC configs).
# Every config auto-resumes from checkpoint if interrupted.
#
# Usage:
#   bash run_experiments.sh              # run everything (default)
#   bash run_experiments.sh phase1       # Phase 1 only
#   bash run_experiments.sh phase2       # Phase 2 only
#   bash run_experiments.sh baseline     # single Phase 1 config
#   bash run_experiments.sh hdc_gate     # single Phase 2 config
#
# Phase 2 run order is enforced: hdc_rulebased always runs before hdc_gate.
# This validates the pipeline (Zone E → inner → Zone D) before adding learned routing.
#
# On Lightning.ai free tier (T4):
#   Phase 1: ~4-5 hours total (4 configs × ~45-75 min)
#   Phase 2: ~6-8 hours total (6 configs × ~60-90 min)
#   Session interrupted? Re-run the same command — all configs auto-resume.

set -e

TARGET="${1:-all}"

# ── Sanity checks ──────────────────────────────────────────────────────────────
if [ ! -f "data/enwik8_train.npy" ] || [ ! -f "data/enwik8_val.npy" ]; then
    echo "ERROR: data/enwik8_train.npy or data/enwik8_val.npy not found."
    echo "       Run: bash setup.sh"
    exit 1
fi

# ── Reporter ───────────────────────────────────────────────────────────────────
mkdir -p checkpoints
bash utils/kill_reporter.sh

REPORTER_PIDFILE="checkpoints/.reporter.pid"
python utils/reporter.py --watch --interval 30 &
REPORTER_PID=$!
echo "$REPORTER_PID" > "$REPORTER_PIDFILE"
trap "kill $REPORTER_PID 2>/dev/null; rm -f '$REPORTER_PIDFILE'; python utils/reporter.py; echo ''; echo 'Shutting down instance...'; sudo shutdown -h now" EXIT

# ── Phase 1 runner ─────────────────────────────────────────────────────────────
run_phase1() {
    local cfg="$1"
    echo ""
    echo "========================================"
    echo "  Phase 1: $cfg"
    echo "========================================"

    RESUME_FLAG=""
    if [ -f "checkpoints/${cfg}_latest.pt" ]; then
        echo "  Found checkpoint — resuming."
        RESUME_FLAG="--resume"
    fi

    python phase1/train.py \
        --config    "$cfg" \
        --total_steps  50000 \
        --batch_size   32 \
        --seq_len      512 \
        --eval_interval 2500 \
        --log_interval  100 \
        $RESUME_FLAG
}

# ── Phase 2 runner ─────────────────────────────────────────────────────────────
run_phase2() {
    local cfg="$1"
    echo ""
    echo "========================================"
    echo "  Phase 2: $cfg"
    echo "========================================"

    RESUME_FLAG=""
    if [ -f "checkpoints/${cfg}_latest.pt" ]; then
        echo "  Found checkpoint — resuming."
        RESUME_FLAG="--resume"
    fi

    python phase2/train.py \
        --config    "$cfg" \
        --total_steps  50000 \
        --batch_size   32 \
        --seq_len      512 \
        --eval_interval 2500 \
        --log_interval  100 \
        $RESUME_FLAG
}

# ── Dispatch ───────────────────────────────────────────────────────────────────
case "$TARGET" in

  all | phase1_then_phase2)
    # Phase 1
    run_phase1 baseline
    run_phase1 mhc
    run_phase1 mol
    run_phase1 compose

    # Phase 2 — hdc_rulebased always first (pipeline validation before learned routing)
    run_phase2 hdc_rulebased
    run_phase2 hdc_gate
    run_phase2 hdc_stride
    run_phase2 hdc_r2
    run_phase2 hdc_r8
    run_phase2 hdc_e2e_isolated
    ;;

  phase1)
    run_phase1 baseline
    run_phase1 mhc
    run_phase1 mol
    run_phase1 compose
    ;;

  phase2)
    # hdc_rulebased always first
    run_phase2 hdc_rulebased
    run_phase2 hdc_gate
    run_phase2 hdc_stride
    run_phase2 hdc_r2
    run_phase2 hdc_r8
    run_phase2 hdc_e2e_isolated
    ;;

  # Individual Phase 1 configs
  baseline | mhc | mol | compose)
    run_phase1 "$TARGET"
    ;;

  # Individual Phase 2 configs
  hdc_rulebased | hdc_gate | hdc_stride | hdc_r2 | hdc_r8 | hdc_e2e_isolated)
    # Safety: warn if running hdc_gate before hdc_rulebased
    if [ "$TARGET" = "hdc_gate" ] && [ ! -f "checkpoints/hdc_rulebased_summary.json" ]; then
        echo "WARNING: hdc_rulebased has not completed yet."
        echo "         hdc_rulebased validates the HDC pipeline before learned routing."
        echo "         It is strongly recommended to run it first."
        echo "         Continuing in 5 seconds — Ctrl+C to cancel."
        sleep 5
    fi
    run_phase2 "$TARGET"
    ;;

  *)
    echo "ERROR: Unknown target '$TARGET'"
    echo "Usage: bash run_experiments.sh [all|phase1|phase2|<config>]"
    echo "Phase 1 configs: baseline mhc mol compose"
    echo "Phase 2 configs: hdc_rulebased hdc_gate hdc_stride hdc_r2 hdc_r8 hdc_e2e_isolated"
    exit 1
    ;;
esac

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  All done. Results in checkpoints/"
echo "========================================"
python - <<'EOF'
import json, os, glob

PHASE1 = ["baseline", "mhc", "mol", "compose"]
PHASE2 = ["hdc_rulebased", "hdc_gate", "hdc_stride", "hdc_r2", "hdc_r8", "hdc_e2e_isolated"]

def print_phase(label, configs):
    rows = []
    for cfg in configs:
        path = f"checkpoints/{cfg}_summary.json"
        if os.path.exists(path):
            with open(path) as f:
                s = json.load(f)
            rows.append((s["config"], s["best_val_bpc"], s["n_params"]))
    if not rows:
        return
    print(f"\n{label}")
    print(f"{'Config':<22} {'Best BPC':>10} {'Params':>12}")
    print("-" * 46)
    for cfg, bpc, params in rows:
        print(f"{cfg:<22} {bpc:>10.4f} {params:>12,}")

print_phase("Phase 1", PHASE1)
print_phase("Phase 2", PHASE2)

# Cross-phase comparison: best Phase 2 vs mol baseline
import os
mol_path = "checkpoints/mol_summary.json"
p2_paths = [f"checkpoints/{c}_summary.json" for c in PHASE2]
if os.path.exists(mol_path) and any(os.path.exists(p) for p in p2_paths):
    with open(mol_path) as f:
        mol_bpc = json.load(f)["best_val_bpc"]
    print(f"\nPhase 2 vs mol baseline ({mol_bpc:.4f}):")
    for cfg in PHASE2:
        path = f"checkpoints/{cfg}_summary.json"
        if os.path.exists(path):
            with open(path) as f:
                s = json.load(f)
            delta = s["best_val_bpc"] - mol_bpc
            print(f"  {cfg:<22} {s['best_val_bpc']:.4f}  ({delta:+.4f})")
EOF
