#!/bin/bash
# Second pixel batch, 4-GPU box. No overnight deadline this time -- epoch
# budgets are generous on purpose, this is expected to run for many hours
# and get checked in on periodically, not necessarily finish by morning.
#
#   CPU: collect the 3rd data-scale point, 8100 train episodes (seed=4,
#        no vector-stage equivalent to match -- this scale was never
#        collected for vectors, so there's nothing to reuse a seed from).
#   GPU0: continue the 2700-episode run (warm-start ep25) -- its last 4
#         epochs gained 0.6-0.8 margin points each, nowhere near flat,
#         we stopped it on a schedule, not because it was done.
#   GPU1: reseed 2700 from scratch, different seed -- the data>>capacity
#         finding for pixels has run exactly once per config; this is the
#         first reproducibility check before leaning on it further.
#   GPU2: ViT-small on 2700, fresh -- does capacity matter once there's
#         enough data to use it? (900-episode data said no; untested at
#         this scale, and capacity x data is a real interaction to rule
#         out, not assume away from a smaller-data result.)
#   GPU3: num_preds=3 on 2700, fresh -- does the roughly-tied multi-step
#         finding (from 900-episode data) change with more data, and does
#         it help the rollout-degrades-faster-than-900 problem just found
#         on this same 2700 dataset.

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

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGDIR/_timeline_pixel2.log"; }

TRAIN_H5="$DATADIR/stage1_vector_2700_pixels_train.h5"
VAL_H5="$DATADIR/stage1_vector_2700_pixels_val.h5"

log "START gpu0_continue_2700 (GPU 0)"
CUDA_VISIBLE_DEVICES=0 "$PY" stage3_pixel/train.py \
    --train-h5 "$TRAIN_H5" --val-h5 "$VAL_H5" \
    --epochs 300 --start-epoch 26 --batch-size 128 --lr 5e-4 --warmup-steps 0 \
    --amp --amp-dtype fp16 \
    --init-ckpt "$CKPTDIR/stage3_pixel_gpu2_datascale/weights_epoch_25.pt" \
    --run-name stage3_pixel_gpu0_continue2700 --log-every 50 --ckpt-every 5 \
    > "$LOGDIR/25_pixel_gpu0_continue2700.log" 2>&1 &
PID0=$!

log "START gpu1_reseed_2700 (GPU 1)"
CUDA_VISIBLE_DEVICES=1 "$PY" stage3_pixel/train.py \
    --train-h5 "$TRAIN_H5" --val-h5 "$VAL_H5" \
    --epochs 300 --batch-size 128 --lr 5e-4 --warmup-steps 500 --seed 1 \
    --amp --amp-dtype fp16 \
    --run-name stage3_pixel_gpu1_reseed2700 --log-every 50 --ckpt-every 5 \
    > "$LOGDIR/26_pixel_gpu1_reseed2700.log" 2>&1 &
PID1=$!

log "START gpu2_capacity_2700 (GPU 2)"
CUDA_VISIBLE_DEVICES=2 "$PY" stage3_pixel/train.py \
    --train-h5 "$TRAIN_H5" --val-h5 "$VAL_H5" \
    --epochs 200 --batch-size 128 --lr 5e-4 --warmup-steps 500 \
    --amp --amp-dtype fp16 --vit-size small \
    --run-name stage3_pixel_gpu2_capacity2700 --log-every 50 --ckpt-every 3 \
    > "$LOGDIR/27_pixel_gpu2_capacity2700.log" 2>&1 &
PID2=$!

log "START gpu3_multistep_2700 (GPU 3)"
CUDA_VISIBLE_DEVICES=3 "$PY" stage3_pixel/train.py \
    --train-h5 "$TRAIN_H5" --val-h5 "$VAL_H5" \
    --epochs 250 --batch-size 128 --lr 5e-4 --warmup-steps 500 --num-preds 3 \
    --amp --amp-dtype fp16 \
    --run-name stage3_pixel_gpu3_multistep2700 --log-every 50 --ckpt-every 4 \
    > "$LOGDIR/28_pixel_gpu3_multistep2700.log" 2>&1 &
PID3=$!

log "all 4 GPU jobs launched: gpu0=$PID0 gpu1=$PID1 gpu2=$PID2 gpu3=$PID3"

log "START collect_pixels_8100 (CPU)"
QT_QPA_PLATFORM=offscreen "$PY" -m lewm collect ~/lewm_runs/stage1_vector_8100_pixels \
    --n-train 8100 --n-val 450 --n-test 450 --seed 4 --rgb \
    --note "Stage 3 data-scale point 3, no vector-stage equivalent at this size" \
    > "$LOGDIR/29_collect_pixels_8100.log" 2>&1
log "8100-pixel collection finished"

"$PY" -m lewm validate ~/lewm_runs/stage1_vector_8100_pixels --loose \
    > "$LOGDIR/29b_validate_8100_pixels.log" 2>&1
log "8100-pixel validation done (see 29b log)"

"$PY" stage3_pixel/export_hdf5.py ~/lewm_runs/stage1_vector_8100_pixels \
    --name stage1_vector_8100_pixels --out-dir "$DATADIR" \
    > "$LOGDIR/29c_export_8100_pixels.log" 2>&1
log "8100-pixel export done -- ready to train once a GPU frees up"

wait $PID0 $PID1 $PID2 $PID3
log "ALL GPU JOBS DONE"
