#!/bin/bash
# Run all MoLE toy validation experiments in sequence.
# Covers Phase 1 (4 configs) and Phase 2 (8 HDC configs, including upcycle).
# Every config auto-resumes from checkpoint if interrupted.
#
# Usage:
#   bash run_experiments.sh                      # run everything (default)
#   bash run_experiments.sh phase1               # Phase 1 only
#   bash run_experiments.sh phase2               # Phase 2 only (all 8 configs)
#   bash run_experiments.sh baseline             # single Phase 1 config
#   bash run_experiments.sh hdc_gate             # single Phase 2 config
#   bash run_experiments.sh phase2 --shutdown    # shutdown instance when done
#
# Phase 2 run order is enforced: hdc_rulebased always runs before hdc_gate.
# Upcycle configs require checkpoints/mol_best.pt (run mol first).
#
# On Lightning.ai free tier (T4):
#   Phase 1: ~4-5 hours total (4 configs × ~45-75 min)
#   Phase 2: ~8-10 hours total (8 configs × ~60-90 min)
#   Session interrupted? Re-run the same command — all configs auto-resume.

set -e

TARGET="all"
SHUTDOWN=0
for arg in "$@"; do
  if [ "$arg" = "--shutdown" ]; then
    SHUTDOWN=1
  elif [[ "$arg" != --* ]]; then
    TARGET="$arg"
  fi
done

# ── Sanity checks ──────────────────────────────────────────────────────────────
if [ ! -f "data/wikitext103_bpe_train.npy" ] || [ ! -f "data/wikitext103_bpe_val.npy" ]; then
    echo "ERROR: WikiText-103 BPE splits not found."
    echo "       Run: bash setup.sh"
    exit 1
fi

# ── Verification check ─────────────────────────────────────────────────────────
# Verifies model code was checked against references/ before this run.
# Full verification runs locally (Claude Code). This is hash-only, no agents.
if ! python utils/verify.py check; then
    echo ""
    echo "Push a passing verification from your local machine before training."
    exit 1
fi

# ── Reporter ───────────────────────────────────────────────────────────────────
mkdir -p checkpoints
bash utils/kill_reporter.sh

REPORTER_PIDFILE="checkpoints/.reporter.pid"
python utils/reporter.py --watch --interval 30 &
REPORTER_PID=$!
echo "$REPORTER_PID" > "$REPORTER_PIDFILE"

# Track whether the script was interrupted by Ctrl+C.
# SIGINT sets this flag then re-raises so the EXIT trap fires normally,
# but shutdown is skipped — giving you a chance to git pull and restart.
_INTERRUPTED=0
trap '_INTERRUPTED=1; trap - INT; kill -INT $$' INT

if [ "$SHUTDOWN" = "1" ]; then
    trap "kill $REPORTER_PID 2>/dev/null; rm -f '$REPORTER_PIDFILE'; python utils/reporter.py; if [ \$_INTERRUPTED -eq 0 ]; then echo ''; echo 'Shutting down instance...'; sudo shutdown -h now; fi" EXIT
else
    trap "kill $REPORTER_PID 2>/dev/null; rm -f '$REPORTER_PIDFILE'; python utils/reporter.py" EXIT
fi

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
        --d            512 \
        --n_heads      8 \
        --mol_rank     8 \
        --total_steps  50000 \
        --batch_size   32 \
        --seq_len      256 \
        --eval_interval 2500 \
        --log_interval  100 \
        $RESUME_FLAG
}

# ── Phase 2 runner ─────────────────────────────────────────────────────────────
# args: <config> [total_steps] [mol_ckpt]
run_phase2() {
    local cfg="$1"
    local steps="${2:-50000}"
    local mol_ckpt="${3:-}"
    echo ""
    echo "========================================"
    echo "  Phase 2: $cfg"
    echo "========================================"

    RESUME_FLAG=""
    if [ -f "checkpoints/${cfg}_latest.pt" ]; then
        echo "  Found checkpoint — resuming."
        RESUME_FLAG="--resume"
    fi

    MOL_FLAG=""
    if [ -n "$mol_ckpt" ]; then
        if [ ! -f "$mol_ckpt" ]; then
            echo "ERROR: mol_ckpt not found: $mol_ckpt"
            echo "       Run phase1 mol first: bash run_experiments.sh mol"
            exit 1
        fi
        MOL_FLAG="--mol_ckpt $mol_ckpt"
    fi

    python phase2/train.py \
        --config    "$cfg" \
        --d            512 \
        --n_heads      8 \
        --mol_rank     8 \
        --total_steps  "$steps" \
        --batch_size   32 \
        --seq_len      256 \
        --eval_interval 2500 \
        --log_interval  100 \
        $MOL_FLAG \
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
    run_phase2 hdc_upcycle_stride 25000 "checkpoints/mol_best.pt"
    run_phase2 hdc_upcycle_gate   25000 "checkpoints/mol_best.pt"
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
    run_phase2 hdc_upcycle_stride 25000 "checkpoints/mol_best.pt"
    run_phase2 hdc_upcycle_gate   25000 "checkpoints/mol_best.pt"
    ;;

  # Individual Phase 1 configs
  baseline | mhc | mol | compose)
    run_phase1 "$TARGET"
    ;;

  # Individual Phase 2 configs (standard, 50k steps)
  hdc_rulebased | hdc_gate | hdc_stride | hdc_r2 | hdc_r8 | hdc_e2e_isolated)
    if [ "$TARGET" = "hdc_gate" ] && [ ! -f "checkpoints/hdc_rulebased_summary.json" ]; then
        echo "WARNING: hdc_rulebased has not completed yet."
        echo "         hdc_rulebased validates the HDC pipeline before learned routing."
        echo "         It is strongly recommended to run it first."
        echo "         Continuing in 5 seconds — Ctrl+C to cancel."
        sleep 5
    fi
    run_phase2 "$TARGET"
    ;;

  # Individual upcycle configs (25k steps, requires mol_best.pt)
  hdc_upcycle_stride)
    run_phase2 hdc_upcycle_stride 25000 "checkpoints/mol_best.pt"
    ;;
  hdc_upcycle_gate)
    run_phase2 hdc_upcycle_gate 25000 "checkpoints/mol_best.pt"
    ;;

  *)
    echo "ERROR: Unknown target '$TARGET'"
    echo "Usage: bash run_experiments.sh [all|phase1|phase2|<config>]"
    echo "Phase 1 configs: baseline mhc mol compose"
    echo "Phase 2 configs: hdc_rulebased hdc_gate hdc_stride hdc_r2 hdc_r8 hdc_e2e_isolated"
    echo "                 hdc_upcycle_stride hdc_upcycle_gate  (require mol_best.pt)"
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
PHASE2 = [
    "hdc_rulebased", "hdc_gate", "hdc_stride", "hdc_r2", "hdc_r8",
    "hdc_e2e_isolated", "hdc_upcycle_stride", "hdc_upcycle_gate",
]

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
    print(f"{'Config':<26} {'Best BPC':>10} {'Params':>12}")
    print("-" * 50)
    for cfg, bpc, params in rows:
        print(f"{cfg:<26} {bpc:>10.4f} {params:>12,}")

print_phase("Phase 1", PHASE1)
print_phase("Phase 2", PHASE2)

# Cross-phase comparison: best Phase 2 vs mol baseline
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
            print(f"  {cfg:<26} {s['best_val_bpc']:.4f}  ({delta:+.4f})")
EOF
