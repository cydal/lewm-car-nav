#!/bin/bash
# Overnight data-scale + epoch-budget batch, now using the confirmed
# recipe (frameskip=5, window_stride=1) throughout -- the previous
# 900-vs-2700 data-scale comparison predates that decision and mixed
# warm-starts/interrupted budgets, so it doesn't count. This batch is
# meant to give a real answer by morning (~9-9.5h budget, user won't
# be back for 10-11h) to two separate questions:
#
#   (1) does more data help, at the recipe we now trust, and
#   (2) where does the probe-recoverable-structure plateau sit as data
#       scales up -- since margin alone was already shown (previous
#       entry) to keep climbing well past the point representation
#       quality stops improving.
#
# Per-step cost confirmed identical across dataset sizes in a pre-launch
# benchmark (~203ms/step everywhere); only windows/epoch differs:
#   900 eps:  470,507 windows  -> ~775s/epoch  (already measured, stride1_seed10/11)
#   2700 eps: 1,364,629 windows -> ~2170s/epoch (0.60h)
#   8100 eps: 4,152,269 windows -> ~6588s/epoch (1.83h)
#
# Checkpointing every epoch (ckpt-every=1) on the two new scales --
# cheap at 71.6MB/checkpoint, and fine granularity is exactly what
# probe_over_time.py needs to actually locate the plateau rather than
# guess at it between sparse checkpoints.
#
#   GPU0: 2700 eps, fresh, seed=20, 15 epochs (~9.0h) -- primary
#         data-scale test #1.
#   GPU1: 2700 eps, fresh, seed=21, 15 epochs (~9.0h) -- second seed,
#         same reason the stride comparison used two: so a difference
#         from GPU0 isn't mistaken for seed noise.
#   GPU2: 8100 eps, fresh, seed=30, 5 epochs (~9.2h) -- best-effort at
#         the largest scale we have; even a partial, pre-plateau read
#         is informative when matched epoch-for-epoch against the
#         2700 and 900 runs' early checkpoints.
#   GPU3: continue stage3_pixel_stride1_seed10 (900 eps) from its
#         existing epoch 40 checkpoint out to epoch 84 (44 more
#         epochs, ckpt-every=2, ~9.5h) -- directly extends the
#         probe-over-time curve already in hand for the small-scale
#         run, to see whether the heading/goal_bearing erosion
#         continues, stabilizes, or reverses past epoch 40, and
#         whether margin ever stops climbing.

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

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGDIR/_timeline_scale.log"; }

common_args=(--frameskip 5 --window-stride 1 --batch-size 128 --lr 5e-4
             --warmup-steps 500 --amp --amp-dtype fp16 --log-every 200)

log "START scale2700_seed20 (GPU 0)"
CUDA_VISIBLE_DEVICES=0 "$PY" stage3_pixel/train.py "${common_args[@]}" \
    --train-h5 "$DATADIR/stage1_vector_2700_pixels_train.h5" \
    --val-h5 "$DATADIR/stage1_vector_2700_pixels_val.h5" \
    --seed 20 --epochs 15 --ckpt-every 1 \
    --run-name stage3_pixel_scale2700_seed20 \
    > "$LOGDIR/50_scale2700_seed20.log" 2>&1 &
PID0=$!

log "START scale2700_seed21 (GPU 1)"
CUDA_VISIBLE_DEVICES=1 "$PY" stage3_pixel/train.py "${common_args[@]}" \
    --train-h5 "$DATADIR/stage1_vector_2700_pixels_train.h5" \
    --val-h5 "$DATADIR/stage1_vector_2700_pixels_val.h5" \
    --seed 21 --epochs 15 --ckpt-every 1 \
    --run-name stage3_pixel_scale2700_seed21 \
    > "$LOGDIR/51_scale2700_seed21.log" 2>&1 &
PID1=$!

log "START scale8100_seed30 (GPU 2)"
CUDA_VISIBLE_DEVICES=2 "$PY" stage3_pixel/train.py "${common_args[@]}" \
    --train-h5 "$DATADIR/stage1_vector_8100_pixels_train.h5" \
    --val-h5 "$DATADIR/stage1_vector_8100_pixels_val.h5" \
    --seed 30 --epochs 5 --ckpt-every 1 \
    --run-name stage3_pixel_scale8100_seed30 \
    > "$LOGDIR/52_scale8100_seed30.log" 2>&1 &
PID2=$!

log "START stride1_seed10_continue (GPU 3)"
CUDA_VISIBLE_DEVICES=3 "$PY" stage3_pixel/train.py "${common_args[@]}" \
    --train-h5 "$DATADIR/stage1_vector_900_pixels_train.h5" \
    --val-h5 "$DATADIR/stage1_vector_900_pixels_val.h5" \
    --seed 10 --epochs 44 --start-epoch 41 --warmup-steps 0 --ckpt-every 2 \
    --init-ckpt "$CKPTDIR/stage3_pixel_stride1_seed10/weights_epoch_40.pt" \
    --run-name stage3_pixel_stride1_seed10 \
    > "$LOGDIR/53_stride1_seed10_continue.log" 2>&1 &
PID3=$!

log "all 4 launched: PIDs $PID0 $PID1 $PID2 $PID3"

wait $PID0 $PID1 $PID2 $PID3
log "all 4 finished"
