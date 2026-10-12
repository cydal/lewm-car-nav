#!/bin/bash
# Plateau sweep: 8100 episodes, frameskip=5, window_stride=1, checkpoints every epoch
# Goal: find where margin plateaus *and* where probed quantities peak/erode
# Two seeds for replication (as used in stride/scale comparisons)

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY=/home/ubuntu/miniconda3/envs/t3d/bin/python
LOGDIR=~/lewm_runs/overnight_logs
DATADIR=~/.stable-wm/datasets
CKPT_DIR=~/lewm_runs/stage3_checkpoints
mkdir -p "$LOGDIR" "$CKPT_DIR"

export WANDB_API_KEY=$("$PY" -c "
for line in open('/home/ubuntu/world_models/le-wm/.env'):
    line = line.strip()
    if line.startswith('WAND_API='):
        print(line.split('=', 1)[1])
")

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGDIR/_timeline_plateau.log"; }

run_step() {
    local name="$1"; shift
    log "START $name"
    if "$@" > "$LOGDIR/$name.log" 2>&1; then
        log "OK    $name"
    else
        log "FAIL  $name"
        tail -20 "$LOGDIR/$name.log"
    fi
}

log "=== Plateau sweep: 8100 episodes, two seeds ==="

# Warm the data files before starting (lazy-load from snapshot)
log "Warming HDF5 files..."
for split in train val test; do
    file="$DATADIR/stage1_vector_8100_pixels_${split}.h5"
    timeout 300 cat "$file" > /dev/null 2>&1 &
    sleep 1
done
wait
log "Warm complete"

# GPU0: seed 40, from scratch
run_step "plateau_8100_seed40" \
    "$PY" stage3_pixel/train.py \
        --train-h5 "$DATADIR/stage1_vector_8100_pixels_train.h5" \
        --val-h5 "$DATADIR/stage1_vector_8100_pixels_val.h5" \
        --epochs 20 --batch-size 64 --warmup-steps 500 --lr 5e-4 \
        --image-size 64 --patch-size 8 \
        --run-name plateau_8100_seed40 \
        --log-every 100 --ckpt-every 1 \
        --seed 40

log "=== Plateau sweep complete ==="
log "Next step: run probe_over_time.py on plateau_8100_seed40 checkpoints"
