#!/bin/bash
# Run MoLE toy validation experiments.
# 17 Phase 1 configs (5 study groups) + Phase 1 scaling study + 10 Phase 2 configs.
# Every config auto-resumes from checkpoint if interrupted.
#
# Study groups (see STUDY_DESIGN.md for research questions):
#   Study A — MoLE core     (Q1, Q2):  baseline baseline_wide mol mol_single mhc compose
#   Study B — Attention     (Q4, Q5):  mla diff_attn diff_mla
#   Study C — mHC compose   (Q6):      diff_mhc mla_mhc diff_mla_mhc
#   Study D — nGPT          (Q7):      ngpt ngpt_mla ngpt_diff_attn
#   Study E — Multi-sphere  (Q8):      ngpt_mhc_a ngpt_mhc_c
#
# Usage:
#   bash run_experiments.sh                      # run everything (default)
#   bash run_experiments.sh phase1               # all Phase 1 (17 configs, d=512)
#   bash run_experiments.sh phase1_scaling       # scaling study: 5 configs × d=256,768
#   bash run_experiments.sh phase2               # Phase 2 only (10 outer encoder configs)
#   bash run_experiments.sh study_mole           # Study A: MoLE core (Q1, Q2)
#   bash run_experiments.sh study_attention      # Study B: attention variants (Q4, Q5)
#   bash run_experiments.sh study_mhc_compose    # Study C: go-mHC compositions (Q6)
#   bash run_experiments.sh study_ngpt           # Study D: nGPT hyperspherical (Q7)
#   bash run_experiments.sh study_sphere         # Study E: multi-sphere compositions (Q8)
#   bash run_experiments.sh baseline             # single Phase 1 config
#   bash run_experiments.sh outer_crl            # single Phase 2 config
#   bash run_experiments.sh phase2 --shutdown    # shutdown instance when done
#
# Phase 2 run order: outer_crl first (validates pipeline, cosine rule),
# then learned routing, then full-width CRL, then transformer variants, then ablations.
#
# On Lightning.ai free tier (T4):
#   Phase 1 (all 17): ~14-17 hours (~45-75 min per config)
#   Phase 2: ~10-15 hours total (10 configs × ~60-90 min)
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
python utils/reporter.py --watch --interval 30 --parent-pid $$ &
REPORTER_PID=$!
echo "$REPORTER_PID" > "$REPORTER_PIDFILE"

# ── Training API server ─────────────────────────────────────────────────────────
# Exposes live metrics at https://training.onbrandsolutions.org via Cloudflare Tunnel.
# Requires TRAINING_API_TOKEN and TUNNEL_TOKEN env vars to be set on the instance.
# Skips silently if either is missing — training is never blocked by the API.
API_PID=""
CLOUDFLARED_PID=""
API_PIDFILE="checkpoints/.api.pid"
CLOUDFLARED_PIDFILE="checkpoints/.cloudflared.pid"

if [ -n "${TRAINING_API_TOKEN:-}" ] && [ -n "${TUNNEL_TOKEN:-}" ]; then
    echo "Starting training API server on port 8787..."
    uvicorn utils.api_server:app --host 0.0.0.0 --port 8787 \
        --log-level warning --no-access-log \
        >> checkpoints/api_server.log 2>&1 &
    API_PID=$!
    echo "$API_PID" > "$API_PIDFILE"

    echo "Starting Cloudflare Tunnel..."
    cloudflared tunnel run --token "$TUNNEL_TOKEN" \
        >> checkpoints/cloudflared.log 2>&1 &
    CLOUDFLARED_PID=$!
    echo "$CLOUDFLARED_PID" > "$CLOUDFLARED_PIDFILE"

    echo "  API: https://training.onbrandsolutions.org"
else
    echo "  [API] Skipping: TRAINING_API_TOKEN or TUNNEL_TOKEN not set."
fi

# Track whether the script was interrupted by Ctrl+C.
# SIGINT sets this flag then re-raises so the EXIT trap fires normally,
# but shutdown is skipped — giving you a chance to git pull and restart.
_INTERRUPTED=0
trap '_INTERRUPTED=1; trap - INT; kill -INT $$' INT

_cleanup() {
    kill "$REPORTER_PID" 2>/dev/null
    rm -f "$REPORTER_PIDFILE"
    [ -n "$API_PID" ]         && kill "$API_PID"         2>/dev/null
    [ -n "$CLOUDFLARED_PID" ] && kill "$CLOUDFLARED_PID" 2>/dev/null
    rm -f "$API_PIDFILE" "$CLOUDFLARED_PIDFILE"
    python utils/reporter.py
}

if [ "$SHUTDOWN" = "1" ]; then
    trap "_cleanup; if [ \$_INTERRUPTED -eq 0 ]; then echo ''; echo 'Shutting down instance...'; sudo shutdown -h now; fi" EXIT
else
    trap "_cleanup" EXIT
fi

# ── Phase 1 runner ─────────────────────────────────────────────────────────────
run_phase1() {
    local cfg="$1"
    local seed="${2:-42}"
    local steps="${3:-100000}"
    echo ""
    echo "========================================"
    echo "  Phase 1: $cfg (seed=$seed)"
    echo "========================================"

    # Skip if already trained to the target step count.
    # Reads the last step from the JSONL rather than relying on summary.json,
    # so this remains correct if total_steps is later increased.
    local jsonl="checkpoints/${cfg}_seed${seed}.jsonl"
    if [ -f "$jsonl" ]; then
        local last_step
        last_step=$(grep -oP '"step":\s*\K[0-9]+' "$jsonl" | tail -1)
        if [ -n "$last_step" ] && [ "$last_step" -ge $(( steps - 1 )) ]; then
            echo "  Already complete at step $last_step — skipping."
            return 0
        fi
    fi

    RESUME_FLAG=""
    if [ -f "checkpoints/${cfg}_seed${seed}_latest.pt" ]; then
        echo "  Found checkpoint — resuming."
        RESUME_FLAG="--resume"
    fi

    # Config-specific overrides
    # baseline_wide: d_ff=1600 to match mol total param count (~31.1M) for Q1 fair comparison
    # mol_single: mol_rank=72 (= 1 shared + 8 experts × rank 8 = rank-72 exact capacity match)
    # diff_attn_matched: d_ff=1240 to compensate for doubled Q projection (~27.83M ≈ baseline ~27.80M)
    EXTRA_FLAGS=""
    if [ "$cfg" = "baseline_wide" ]; then
        EXTRA_FLAGS="--d_ff 1600"
    elif [ "$cfg" = "mol_single" ]; then
        EXTRA_FLAGS="--mol_rank 72"
    elif [ "$cfg" = "diff_attn_matched" ]; then
        EXTRA_FLAGS="--d_ff 1240"
    fi

    python phase1/train.py \
        --config    "$cfg" \
        --d            512 \
        --n_heads      8 \
        --total_steps  "$steps" \
        --batch_size   32 \
        --seq_len      256 \
        --eval_interval 2500 \
        --log_interval  100 \
        --seed         "$seed" \
        $EXTRA_FLAGS \
        $RESUME_FLAG
}

# ── Phase 1 scaling runner ─────────────────────────────────────────────────────
# Runs a single Phase 1 config at a non-default scale (d≠512).
# Checkpoint prefix encodes scale: e.g. baseline_d768_seed42.
# n_heads is always d//64 to keep head_dim=64 constant across scales.
run_phase1_scale() {
    local cfg="$1"
    local d="$2"
    local n_heads="$3"
    local seed="${4:-42}"
    local steps="${5:-100000}"
    local prefix="${cfg}_d${d}_seed${seed}"

    echo ""
    echo "========================================"
    echo "  Phase 1 Scale: $cfg  d=$d  n_heads=$n_heads  (seed=$seed)"
    echo "========================================"

    local jsonl="checkpoints/${prefix}.jsonl"
    if [ -f "$jsonl" ]; then
        local last_step
        last_step=$(grep -oP '"step":\s*\K[0-9]+' "$jsonl" | tail -1)
        if [ -n "$last_step" ] && [ "$last_step" -ge $(( steps - 1 )) ]; then
            echo "  Already complete at step $last_step — skipping."
            return 0
        fi
    fi

    RESUME_FLAG=""
    if [ -f "checkpoints/${prefix}_latest.pt" ]; then
        echo "  Found checkpoint — resuming."
        RESUME_FLAG="--resume"
    fi

    python phase1/train.py \
        --config      "$cfg" \
        --d           "$d" \
        --n_heads     "$n_heads" \
        --ckpt_prefix "$prefix" \
        --total_steps "$steps" \
        --batch_size  32 \
        --seq_len     256 \
        --eval_interval 2500 \
        --log_interval  100 \
        --seed        "$seed" \
        $RESUME_FLAG
}

# ── Phase 2 smoke test ─────────────────────────────────────────────────────────
# Runs 1000 steps of outer_crl and checks health metrics defined in
# references/components/zone_ed_pipeline.md checklist item 10.
# BLOCKS all Phase 2 training if it fails or if phase2/model.py has changed
# since the last passing run.
run_smoke_test() {
    echo ""
    echo "========================================"
    echo "  Phase 2 Smoke Test (pre-flight check)"
    echo "========================================"

    # First check if a current passing result is already on file.
    if python utils/smoke_test.py --check-only 2>/dev/null; then
        echo "  Smoke test current — skipping re-run."
        return 0
    fi

    echo "  Running 1000-step smoke test (outer_crl)..."
    if ! python utils/smoke_test.py; then
        echo ""
        echo "========================================"
        echo "  SMOKE TEST FAILED — Phase 2 BLOCKED"
        echo "========================================"
        echo ""
        echo "  The Phase 2 pipeline has a health problem that will cause all"
        echo "  configs to waste compute and plateau at ~9.7 BPC (unigram entropy)."
        echo "  Check checkpoints/smoke_test_result.json for which checks failed."
        echo ""
        echo "  Required before re-running:"
        echo "    1. Fix the root cause (see failed checks above)"
        echo "    2. Run full verification: tell Claude 'run full verification'"
        echo "    3. python utils/verify.py update --result pass --report <path>"
        echo "    4. git commit"
        echo "    5. Re-run this script"
        echo ""
        exit 1
    fi
}

# ── Phase 2 runner ─────────────────────────────────────────────────────────────
# args: <config> [total_steps] [seed]
run_phase2() {
    local cfg="$1"
    local steps="${2:-50000}"
    local seed="${3:-42}"
    echo ""
    echo "========================================"
    echo "  Phase 2: $cfg (seed=$seed)"
    echo "========================================"

    # Skip if already trained to the target step count.
    local jsonl="checkpoints/${cfg}_seed${seed}.jsonl"
    if [ -f "$jsonl" ]; then
        local last_step
        last_step=$(grep -oP '"step":\s*\K[0-9]+' "$jsonl" | tail -1)
        if [ -n "$last_step" ] && [ "$last_step" -ge $(( steps - 1 )) ]; then
            echo "  Already complete at step $last_step — skipping."
            return 0
        fi
    fi

    RESUME_FLAG=""
    if [ -f "checkpoints/${cfg}_seed${seed}_latest.pt" ]; then
        echo "  Found checkpoint — resuming."
        RESUME_FLAG="--resume"
    fi

    python phase2/train.py \
        --config    "$cfg" \
        --d            512 \
        --n_heads      8 \
        --total_steps  "${steps}" \
        --batch_size   32 \
        --seq_len      256 \
        --eval_interval 2500 \
        --log_interval  100 \
        --seed         "$seed" \
        $RESUME_FLAG
}

# ── Dispatch ───────────────────────────────────────────────────────────────────
case "$TARGET" in

  all | phase1_then_phase2)
    # Phase 1
    run_phase1 baseline
    run_phase1 baseline_wide
    run_phase1 mhc
    run_phase1 mol
    run_phase1 mol_single
    run_phase1 compose
    run_phase1 mla
    run_phase1 diff_attn
    run_phase1 diff_attn_matched
    run_phase1 diff_mla
    # go-mHC compositions (April 2026)
    run_phase1 diff_mhc
    run_phase1 mla_mhc
    run_phase1 diff_mla_mhc
    # nGPT hypersphere experiments (April 2026)
    run_phase1 ngpt
    run_phase1 ngpt_mla
    run_phase1 ngpt_diff_attn
    # nGPT + mHC compositions (April 2026)
    run_phase1 ngpt_mhc_a
    run_phase1 ngpt_mhc_c

    # Phase 2 — smoke test gates ALL configs, no exceptions
    run_smoke_test
    run_phase2 outer_crl             # pipeline validation first (cosine rule, no learned params)
    run_phase2 outer_crl_learned     # learned routing on bottlenecked CRL
    run_phase2 outer_crl_full        # full-width CRL (no bottleneck confound)
    run_phase2 outer_crl_full_learned
    run_phase2 outer_transformer
    run_phase2 outer_diff_attn
    run_phase2 outer_mla
    run_phase2 outer_strided         # hard lower bound (no encoder)
    run_phase2 outer_crl_learned_noste  # STE ablation
    run_phase2 outer_crl_fixed_stride   # clean routing ablation (CRL encoder + fixed stride)
    ;;

  phase1)
    run_phase1 baseline
    run_phase1 baseline_wide
    run_phase1 mhc
    run_phase1 mol
    run_phase1 mol_single
    run_phase1 compose
    run_phase1 mla
    run_phase1 diff_attn
    run_phase1 diff_attn_matched
    run_phase1 diff_mla
    # go-mHC compositions (April 2026)
    run_phase1 diff_mhc
    run_phase1 mla_mhc
    run_phase1 diff_mla_mhc
    # nGPT hypersphere experiments (April 2026)
    run_phase1 ngpt
    run_phase1 ngpt_mla
    run_phase1 ngpt_diff_attn
    # nGPT + mHC compositions (April 2026)
    run_phase1 ngpt_mhc_a
    run_phase1 ngpt_mhc_c
    ;;

  phase2)
    # smoke test gates ALL Phase 2 configs, no exceptions
    run_smoke_test
    run_phase2 outer_crl
    run_phase2 outer_crl_learned
    run_phase2 outer_crl_full
    run_phase2 outer_crl_full_learned
    run_phase2 outer_transformer
    run_phase2 outer_diff_attn
    run_phase2 outer_mla
    run_phase2 outer_strided
    run_phase2 outer_crl_learned_noste
    run_phase2 outer_crl_fixed_stride
    ;;

  phase1_scaling)
    # Scaling study: 5 key configs at d=256 (n_heads=4) and d=768 (n_heads=12).
    # Checkpoint prefix: {cfg}_d{d}_seed42  (avoids collision with d=512 runs).
    # Goal: measure how each architecture's BPC deficit scales with model size.
    # d_c/d=25% is held constant, so ratio and absolute dimension change proportionally.
    # d=256: head_dim=64, d_c=64  (MLA bottleneck ratio same as d=512: 25%)
    # d=768: head_dim=64, d_c=192 (MLA bottleneck ratio same as d=512: 25%)
    run_phase1_scale baseline   256 4
    run_phase1_scale mla        256 4
    run_phase1_scale diff_attn  256 4
    run_phase1_scale diff_mla   256 4
    run_phase1_scale mol        256 4
    run_phase1_scale baseline   768 12
    run_phase1_scale mla        768 12
    run_phase1_scale diff_attn  768 12
    run_phase1_scale diff_mla   768 12
    run_phase1_scale mol        768 12
    ;;

  # Study A — MoLE core (Q1, Q2)
  study_mole)
    run_phase1 baseline
    run_phase1 baseline_wide
    run_phase1 mol
    run_phase1 mol_single
    run_phase1 mhc
    run_phase1 compose
    ;;

  # Study B — Attention variants (Q4, Q5)
  study_attention)
    run_phase1 mla
    run_phase1 diff_attn
    run_phase1 diff_attn_matched
    run_phase1 diff_mla
    ;;

  # Study C — go-mHC compositions (Q6)
  study_mhc_compose)
    run_phase1 diff_mhc
    run_phase1 mla_mhc
    run_phase1 diff_mla_mhc
    ;;

  # Study D — nGPT hyperspherical (Q7)
  study_ngpt)
    run_phase1 ngpt
    run_phase1 ngpt_mla
    run_phase1 ngpt_diff_attn
    ;;

  # Study E — multi-sphere compositions (Q8)
  study_sphere)
    run_phase1 ngpt_mhc_a
    run_phase1 ngpt_mhc_c
    ;;

  # Individual Phase 1 configs
  baseline | mhc | mol | compose | mla | diff_attn | diff_attn_matched | diff_mla | \
  diff_mhc | mla_mhc | diff_mla_mhc | \
  ngpt | ngpt_mla | ngpt_diff_attn | \
  ngpt_mhc_a | ngpt_mhc_c)
    run_phase1 "$TARGET"
    ;;

  baseline_wide)
    run_phase1 baseline_wide
    ;;

  mol_single)
    run_phase1 mol_single
    ;;

  # Individual Phase 2 configs (standard, 50k steps)
  outer_crl | outer_crl_learned | outer_crl_full | outer_crl_full_learned | \
  outer_transformer | outer_diff_attn | outer_mla | outer_strided | \
  outer_crl_learned_noste | outer_crl_fixed_stride)
    run_smoke_test
    run_phase2 "$TARGET"
    ;;

  *)
    echo "ERROR: Unknown target '$TARGET'"
    echo "Usage: bash run_experiments.sh [TARGET] [--shutdown]"
    echo ""
    echo "Bulk targets:"
    echo "  all                run all Phase 1 + Phase 2"
    echo "  phase1             all 17 Phase 1 configs (d=512)"
    echo "  phase1_scaling     5 configs × d=256,768"
    echo "  phase2             all 10 Phase 2 configs"
    echo ""
    echo "Study group targets (see STUDY_DESIGN.md):"
    echo "  study_mole         Study A: baseline baseline_wide mol mol_single mhc compose"
    echo "  study_attention    Study B: mla diff_attn diff_mla"
    echo "  study_mhc_compose  Study C: diff_mhc mla_mhc diff_mla_mhc"
    echo "  study_ngpt         Study D: ngpt ngpt_mla ngpt_diff_attn"
    echo "  study_sphere       Study E: ngpt_mhc_a ngpt_mhc_c"
    echo ""
    echo "Individual Phase 1 configs:"
    echo "  baseline baseline_wide mhc mol mol_single compose"
    echo "  mla diff_attn diff_mla"
    echo "  diff_mhc mla_mhc diff_mla_mhc"
    echo "  ngpt ngpt_mla ngpt_diff_attn"
    echo "  ngpt_mhc_a ngpt_mhc_c"
    echo ""
    echo "Individual Phase 2 configs:"
    echo "  outer_crl outer_crl_learned outer_crl_full outer_crl_full_learned"
    echo "  outer_transformer outer_diff_attn outer_mla"
    echo "  outer_strided outer_crl_learned_noste outer_crl_fixed_stride"
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

PHASE1 = ["baseline", "baseline_wide", "mhc", "mol", "mol_single", "compose",
          "mla", "diff_attn", "diff_mla",
          "diff_mhc", "mla_mhc", "diff_mla_mhc",
          "ngpt", "ngpt_mla", "ngpt_diff_attn",
          "ngpt_mhc_a", "ngpt_mhc_c"]
PHASE2 = [
    "outer_crl", "outer_crl_learned",
    "outer_crl_full", "outer_crl_full_learned",
    "outer_transformer", "outer_diff_attn", "outer_mla",
    "outer_strided", "outer_crl_learned_noste", "outer_crl_fixed_stride",
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

# Cross-phase comparison: best Phase 2 vs mol_seed42 baseline
mol_path = "checkpoints/mol_seed42_summary.json"
if not os.path.exists(mol_path):
    mol_path = "checkpoints/mol_summary.json"  # fallback to unseeded run
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
