#!/bin/bash
# Run all 4 Phase 1 configs in sequence.
# Each config resumes automatically if a checkpoint already exists.
# Usage: bash run_experiments.sh [baseline|mhc|mol|compose|all]
#
# On Lightning.ai free tier (T4):
#   Each config takes ~45-75 min at 50k steps. All 4 = ~4-5 hours.
#   The GPU session can be paused mid-run — re-run with --resume to continue.

set -e

TARGET="${1:-all}"

# Start reporter as a background process — polls checkpoints/*.jsonl every 30s
# and writes checkpoints/report.md. Killed automatically when this script exits.
python utils/reporter.py --watch --interval 30 &
REPORTER_PID=$!
trap "kill $REPORTER_PID 2>/dev/null; python utils/reporter.py" EXIT

run_config() {
    local cfg="$1"
    echo ""
    echo "========================================"
    echo "  Running: $cfg"
    echo "========================================"

    # Auto-resume if checkpoint exists
    RESUME_FLAG=""
    if [ -f "checkpoints/${cfg}_latest.pt" ]; then
        echo "  Found existing checkpoint — resuming."
        RESUME_FLAG="--resume"
    fi

    python phase1/train.py \
        --config "$cfg" \
        --total_steps 50000 \
        --batch_size 32 \
        --seq_len 512 \
        --eval_interval 2500 \
        --log_interval 100 \
        $RESUME_FLAG
}

if [ "$TARGET" = "all" ]; then
    run_config baseline
    run_config mhc
    run_config mol
    run_config compose
else
    run_config "$TARGET"
fi

echo ""
echo "========================================"
echo "  All done. Results in checkpoints/"
echo "========================================"
python - <<'EOF'
import json, os, glob
rows = []
for path in sorted(glob.glob("checkpoints/*_summary.json")):
    with open(path) as f:
        s = json.load(f)
    rows.append((s["config"], s["best_val_bpc"], s["n_params"]))
if rows:
    print(f"\n{'Config':<12} {'Best BPC':>10} {'Params':>12}")
    print("-" * 36)
    for cfg, bpc, params in rows:
        print(f"{cfg:<12} {bpc:>10.4f} {params:>12,}")
EOF
