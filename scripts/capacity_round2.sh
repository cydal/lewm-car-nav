#!/bin/bash
# Chains after the already-running capacity_run.sh (big1 on full-vector,
# PID below). Two more capacity points, so we get an actual
# size-vs-margin curve instead of one new data point, and one of them
# crosses with Stage 2: does more capacity recover any of the margin
# restricted observations lost, or is that information genuinely gone
# regardless of model size?
#
#   B: big1's config (4.84M params) on the RESTRICTED (59D) 900-episode
#      set. Same architecture that's currently training on full-vector,
#      just pointed at the other dataset -- isolates "does capacity help
#      restricted obs" from "does capacity help at all".
#   C: big2 (10.2M params, ~5.3x baseline) on the FULL-vector 900-episode
#      set -- a second, larger point on the capacity curve, to see if
#      bigger keeps paying off or saturates the way 2700 episodes did.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY=/home/ubuntu/miniconda3/envs/t3d/bin/python
LOGDIR=~/lewm_runs/overnight_logs
DATADIR=~/.stable-wm/datasets
mkdir -p "$LOGDIR"

export WANDB_API_KEY=$("$PY" -c "
for line in open('/home/ubuntu/world_models/le-wm/.env'):
    line = line.strip()
    if line.startswith('WAND_API='):
        print(line.split('=', 1)[1])
")

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGDIR/_timeline3.log"; }

run_step() {
    local name="$1"; shift
    log "START $name"
    if "$@" > "$LOGDIR/$name.log" 2>&1; then
        log "OK    $name"
    else
        log "FAIL  $name (see $LOGDIR/$name.log)"
    fi
}

# Wait for the currently-running big1-on-full-vector job (launched by
# capacity_run.sh) to finish, without being its child -- poll for the PID.
BIG1_PID=783883
log "waiting on big1 (pid $BIG1_PID) if still running"
while kill -0 "$BIG1_PID" 2>/dev/null; do sleep 60; done
log "big1 (full-vector) finished"

# --- Step B: big1's architecture, restricted (59D) observations.
run_step "17_capacity_restricted_big1_200ep" \
    "$PY" stage1_vector/train.py \
        --train-h5 "$DATADIR/stage1_vector_900_restricted_train.h5" \
        --val-h5 "$DATADIR/stage1_vector_900_restricted_val.h5" \
        --epochs 200 --batch-size 1024 --lr 5e-4 \
        --embed-dim 192 --encoder-hidden 384 --encoder-depth 2 \
        --predictor-depth 5 --predictor-heads 8 --predictor-dim-head 40 --predictor-mlp-dim 768 \
        --run-name stage_capacity_restricted_big1 --log-every 20 --ckpt-every 20

# --- Step C: big2, full-vector. A second, larger capacity point.
run_step "18_capacity_900_big2_150ep" \
    "$PY" stage1_vector/train.py \
        --train-h5 "$DATADIR/stage1_vector_900_train.h5" \
        --val-h5 "$DATADIR/stage1_vector_900_val.h5" \
        --epochs 150 --batch-size 1024 --lr 5e-4 \
        --embed-dim 256 --encoder-hidden 512 --encoder-depth 3 \
        --predictor-depth 6 --predictor-heads 8 --predictor-dim-head 48 --predictor-mlp-dim 1024 \
        --run-name stage_capacity_900_big2 --log-every 20 --ckpt-every 15

# --- Final summary across every capacity run: baseline size (already
# have numbers), big1-full, big1-restricted, big2-full.
run_step "19_capacity_full_summary" "$PY" - <<'PYEOF'
import glob, os, re, subprocess

PY = "/home/ubuntu/miniconda3/envs/t3d/bin/python"
DATADIR = os.path.expanduser("~/.stable-wm/datasets")
REPO = os.path.expanduser("~/world_models/lewm-car-nav")

RUNS = {
    "stage_capacity_900_big1": (
        f"{DATADIR}/stage1_vector_900_val.h5",
        ["--embed-dim", "192", "--encoder-hidden", "384", "--encoder-depth", "2",
         "--predictor-depth", "5", "--predictor-heads", "8",
         "--predictor-dim-head", "40", "--predictor-mlp-dim", "768"],
    ),
    "stage_capacity_restricted_big1": (
        f"{DATADIR}/stage1_vector_900_restricted_val.h5",
        ["--embed-dim", "192", "--encoder-hidden", "384", "--encoder-depth", "2",
         "--predictor-depth", "5", "--predictor-heads", "8",
         "--predictor-dim-head", "40", "--predictor-mlp-dim", "768"],
    ),
    "stage_capacity_900_big2": (
        f"{DATADIR}/stage1_vector_900_val.h5",
        ["--embed-dim", "256", "--encoder-hidden", "512", "--encoder-depth", "3",
         "--predictor-depth", "6", "--predictor-heads", "8",
         "--predictor-dim-head", "48", "--predictor-mlp-dim", "1024"],
    ),
}

for run_name, (val_h5, model_args) in RUNS.items():
    ckpt_dir = os.path.expanduser(f"~/lewm_runs/stage1_checkpoints/{run_name}")
    if not os.path.isdir(ckpt_dir):
        print(f"=== {run_name}: no checkpoint dir, skipped ===")
        continue
    ckpts = sorted(
        glob.glob(os.path.join(ckpt_dir, "weights_epoch_*.pt")),
        key=lambda p: int(re.search(r"epoch_(\d+)", p).group(1)),
    )
    print(f"=== {run_name} ({len(ckpts)} checkpoints) ===")
    for ckpt in ckpts:
        epoch = re.search(r"epoch_(\d+)", ckpt).group(1)
        out = subprocess.run(
            [PY, "stage1_vector/baseline_eval.py", "--h5", val_h5, "--ckpt", ckpt, *model_args],
            capture_output=True, text=True, cwd=REPO,
        )
        lines = [l for l in out.stdout.splitlines() if "pred_loss" in l]
        print(f"  epoch {epoch:>4}: " + " | ".join(lines))
        if out.returncode != 0:
            print("  stderr:", out.stderr[-500:])
PYEOF

log "ALL DONE (round 2)"
