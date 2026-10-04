#!/bin/bash
# Overnight batch: push epoch count further on the existing 300-episode
# dataset, generate a 3x-larger dataset (CPU-only, runs concurrently with
# the GPU training step), then train on the bigger dataset two ways
# (from scratch, and warm-started from the small-data checkpoint). Each
# step logs to its own file under ~/lewm_runs/overnight_logs/ and failures
# are caught per-step so one crash doesn't stop the rest of the batch.
#
# Context: run2 (lr=5e-4, 300 train episodes) showed a real, widening
# margin over a trivial "predict no change" baseline through 200 epochs
# (-7% at epoch 40 -> -11..-15% at epoch 180-200), but the growth rate was
# clearly decelerating. This batch tests the two obvious next levers --
# more epochs, more data -- back to back, unattended.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY=/home/ubuntu/miniconda3/envs/t3d/bin/python
LOGDIR=~/lewm_runs/overnight_logs
CKPTDIR=~/lewm_runs/stage1_checkpoints
mkdir -p "$LOGDIR"

export WANDB_API_KEY=$("$PY" -c "
for line in open('/home/ubuntu/world_models/le-wm/.env'):
    line = line.strip()
    if line.startswith('WAND_API='):
        print(line.split('=', 1)[1])
")

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGDIR/_timeline.log"; }

run_step() {
    local name="$1"; shift
    log "START $name"
    if "$@" > "$LOGDIR/$name.log" 2>&1; then
        log "OK    $name"
    else
        log "FAIL  $name (see $LOGDIR/$name.log)"
    fi
}

# --- Step 1: generate the 3x dataset (CPU-only -- no GPU contention with
# the training step launched in parallel below). seed=2, distinct from the
# pilot (20261004) and the first stage1 dataset (1).
QT_QPA_PLATFORM=offscreen run_step "01_collect_bigdata" \
    "$PY" -m lewm collect ~/lewm_runs/stage1_vector_900 \
        --n-train 900 --n-val 150 --n-test 150 --seed 2 \
        --note "Stage 1 vector baseline, 3x the first dataset, overnight batch" &
COLLECT_PID=$!

# --- Step 2 (parallel with Step 1, GPU): continue the healthy lr=5e-4 run
# another 400 epochs (200 -> 600 total) on the ORIGINAL 300-episode
# dataset, to see whether the margin-over-copy trend from epoch 40-200
# keeps widening, flattens, or reverses with substantially more epochs at
# the same data size.
run_step "02_continue_original_600ep" \
    "$PY" stage1_vector/train.py \
        --train-h5 /home/ubuntu/.stable-wm/datasets/stage1_vector_001_train.h5 \
        --val-h5 /home/ubuntu/.stable-wm/datasets/stage1_vector_001_val.h5 \
        --epochs 400 --start-epoch 201 --batch-size 512 --lr 5e-4 \
        --init-ckpt "$CKPTDIR/stage1_vector_run4_continue200/weights_epoch_200.pt" \
        --run-name stage1_vector_run5_continue600 --log-every 20 --ckpt-every 50

# Make sure the dataset is actually done before the next steps need it.
log "waiting on 01_collect_bigdata (pid $COLLECT_PID) if still running"
wait "$COLLECT_PID"

run_step "03_validate_bigdata" \
    "$PY" -m lewm validate ~/lewm_runs/stage1_vector_900

run_step "04_export_bigdata" \
    "$PY" -c "
import sys; sys.path.insert(0, '.')
from stage1_vector.export_hdf5 import export_dataset
export_dataset('/home/ubuntu/lewm_runs/stage1_vector_900', '/home/ubuntu/.stable-wm/datasets', name='stage1_vector_900')
"

# --- Step 5: train on the 3x dataset from scratch. Cleanest read on
# whether more data (not more training time) moves the margin -- same
# architecture and lr as the healthy small-data run, bigger batch since
# there's 3x the windows per epoch and the GPU had headroom to spare
# (1.2 GiB of 15 GiB used at batch=512).
run_step "05_bigdata_scratch_400ep" \
    "$PY" stage1_vector/train.py \
        --train-h5 /home/ubuntu/.stable-wm/datasets/stage1_vector_900_train.h5 \
        --val-h5 /home/ubuntu/.stable-wm/datasets/stage1_vector_900_val.h5 \
        --epochs 400 --batch-size 1024 --lr 5e-4 \
        --run-name stage1_vector_bigdata_scratch --log-every 20 --ckpt-every 25

# --- Step 6: train on the 3x dataset, warm-started from the best
# small-data checkpoint (epoch 180 had the widest margin, -14.8%, in the
# baseline-eval sweep). Tests whether what the model already learned on
# 300 episodes transfers and keeps improving with more data on top of it.
run_step "06_bigdata_warmstart_400ep" \
    "$PY" stage1_vector/train.py \
        --train-h5 /home/ubuntu/.stable-wm/datasets/stage1_vector_900_train.h5 \
        --val-h5 /home/ubuntu/.stable-wm/datasets/stage1_vector_900_val.h5 \
        --epochs 400 --batch-size 1024 --lr 5e-4 \
        --init-ckpt "$CKPTDIR/stage1_vector_run4_continue200/weights_epoch_180.pt" \
        --run-name stage1_vector_bigdata_warmstart --log-every 20 --ckpt-every 25

# --- Step 7: baseline-eval sweep across every run's checkpoints, written
# to one plain-text summary so there's a single file to read in the
# morning instead of re-querying wandb per run.
run_step "07_baseline_summary" "$PY" - <<'PYEOF'
import glob, os, re, subprocess

PY = "/home/ubuntu/miniconda3/envs/t3d/bin/python"
RUNS = {
    "stage1_vector_run5_continue600": "/home/ubuntu/.stable-wm/datasets/stage1_vector_001_val.h5",
    "stage1_vector_bigdata_scratch": "/home/ubuntu/.stable-wm/datasets/stage1_vector_900_val.h5",
    "stage1_vector_bigdata_warmstart": "/home/ubuntu/.stable-wm/datasets/stage1_vector_900_val.h5",
}

for run_name, val_h5 in RUNS.items():
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
            [PY, "stage1_vector/baseline_eval.py", "--h5", val_h5, "--ckpt", ckpt],
            capture_output=True, text=True, cwd=os.path.expanduser("~/world_models/lewm-car-nav"),
        )
        lines = [l for l in out.stdout.splitlines() if "pred_loss" in l]
        print(f"  epoch {epoch:>4}: " + " | ".join(lines))
PYEOF

log "ALL DONE"
