#!/bin/bash
# Definitive window_stride=1 vs window_stride=5 comparison, both at
# frameskip=5, both on the 900-episode pixel set, 2 seeds each, one job
# per physical GPU, all four running the full duration in parallel.
#
# The open question (raised by the user, not settled by the paper's
# text -- see fast_loader.py's docstring): window_stride=frameskip
# (sparse, non-overlapping window starts -- the ~5x fewer-windows
# compute saving) vs window_stride=1 (dense, overlapping window starts
# at the same frameskip=5 frame grouping -- same window count as
# frameskip=1, no compute saving, denser coverage per epoch). Does the
# saving cost anything?
#
# Pre-launch micro-benchmark (actual forward+backward+opt steps, same
# model/sigreg/optimizer, 30 steps each, this box, this GPU) confirmed
# the only difference is windows-per-epoch, not per-step cost:
#   stride=5: 94485 windows,  738 batches/epoch, 202.9ms/step -> ~150s/epoch
#   stride=1: 470507 windows, 3675 batches/epoch, 203.7ms/step -> ~749s/epoch
# (~5x ratio, matching the window-count ratio exactly, as expected.)
#
# Epoch budgets below are sized so BOTH arms get ~8.3h of wall-clock --
# equal compute budget, not equal epoch count, since an "epoch" means a
# very different amount of work depending on stride. This is the fair
# comparison: given the same GPU-hours, which stride setting gets you
# further.
#
#   GPU0: stride=1, seed=10, 40 epochs  (~8.3h at ~749s/epoch)
#   GPU1: stride=5, seed=10, 200 epochs (~8.3h at ~150s/epoch)
#   GPU2: stride=1, seed=11, 40 epochs
#   GPU3: stride=5, seed=11, 200 epochs
#
# A third, pre-existing stride=5 data point (seed=0, crashed at epoch
# 110/300 during the disk-full crisis, last good checkpoint epoch 100,
# margin -72.5%) already exists as stage3_pixel_gpu3_frameskip5 and
# counts as a bonus corroboration for the stride=5 arm, not reproduced
# here.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY=/home/ubuntu/miniconda3/envs/t3d/bin/python
LOGDIR=~/lewm_runs/overnight_logs
CKPTDIR=~/lewm_runs/stage3_checkpoints
DATADIR=~/.stable-wm/datasets
mkdir -p "$LOGDIR"

export WANDB_API_KEY=$("$PY" -c "
for line in open('/home/ubuntu/world_models/le-wm/.env'):
    line = line.strip()
    if line.startswith('WAND_API='):
        print(line.split('=', 1)[1])
")

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGDIR/_timeline_stride.log"; }

common_args=(--train-h5 "$DATADIR/stage1_vector_900_pixels_train.h5"
             --val-h5 "$DATADIR/stage1_vector_900_pixels_val.h5"
             --frameskip 5 --batch-size 128 --lr 5e-4 --warmup-steps 500
             --amp --amp-dtype fp16 --log-every 50)

log "START stride1_seed10 (GPU 0)"
CUDA_VISIBLE_DEVICES=0 "$PY" stage3_pixel/train.py "${common_args[@]}" \
    --window-stride 1 --seed 10 --epochs 40 --ckpt-every 2 \
    --run-name stage3_pixel_stride1_seed10 \
    > "$LOGDIR/40_stride1_seed10.log" 2>&1 &
PID0=$!

log "START stride5_seed10 (GPU 1)"
CUDA_VISIBLE_DEVICES=1 "$PY" stage3_pixel/train.py "${common_args[@]}" \
    --window-stride 5 --seed 10 --epochs 200 --ckpt-every 10 \
    --run-name stage3_pixel_stride5_seed10 \
    > "$LOGDIR/41_stride5_seed10.log" 2>&1 &
PID1=$!

log "START stride1_seed11 (GPU 2)"
CUDA_VISIBLE_DEVICES=2 "$PY" stage3_pixel/train.py "${common_args[@]}" \
    --window-stride 1 --seed 11 --epochs 40 --ckpt-every 2 \
    --run-name stage3_pixel_stride1_seed11 \
    > "$LOGDIR/42_stride1_seed11.log" 2>&1 &
PID2=$!

log "START stride5_seed11 (GPU 3)"
CUDA_VISIBLE_DEVICES=3 "$PY" stage3_pixel/train.py "${common_args[@]}" \
    --window-stride 5 --seed 11 --epochs 200 --ckpt-every 10 \
    --run-name stage3_pixel_stride5_seed11 \
    > "$LOGDIR/43_stride5_seed11.log" 2>&1 &
PID3=$!

log "all 4 launched: PIDs $PID0 $PID1 $PID2 $PID3"

wait $PID0 $PID1 $PID2 $PID3
log "all 4 finished"
