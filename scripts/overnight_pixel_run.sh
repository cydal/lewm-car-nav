#!/bin/bash
# Overnight pixel batch, 4-GPU box, one experiment per GPU, launched
# in parallel (true parallelism this time -- separate physical GPUs, not
# PID-chained sequential like the single-GPU overnight batches). Epoch
# budgets sized generously against measured per-step throughput (AMP
# fp16, confirmed on this box) so nothing finishes early and sits idle --
# worst case we read an earlier checkpoint in the morning, never a wasted
# idle GPU.
#
#   GPU0: continue stage3_pixel_run1 (crashed at epoch 10 during the AMI
#         copy; margin was -46.6% and still climbing every epoch, no
#         sign of plateau -- finish what was interrupted, not restart).
#   GPU1: capacity test -- ViT-small (49.4M params, ~2.76x baseline) on
#         the same 900-episode set. Never tested whether model size
#         matters for pixels at all.
#   GPU2: data-scale test -- 2700 pixel episodes (seed=3, matches the
#         vector 2700 dataset), fresh. Vector stage saw diminishing
#         returns past 900; untested whether pixels behave the same way.
#   GPU3: multi-step training objective (num_preds=3, span 3->6) --
#         directly targets the rollout-reversal finding (margin crossed
#         to worse-than-copy by ~1 second in the vector rollout test).

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

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGDIR/_timeline_pixel.log"; }

# --- GPU0: continue the baseline, warm-started, no warmup (already past
# the unstable from-scratch phase), long budget.
log "START 21_pixel_gpu0_continue (GPU 0)"
CUDA_VISIBLE_DEVICES=0 "$PY" stage3_pixel/train.py \
    --train-h5 "$DATADIR/stage1_vector_900_pixels_train.h5" \
    --val-h5 "$DATADIR/stage1_vector_900_pixels_val.h5" \
    --epochs 80 --start-epoch 11 --batch-size 128 --lr 5e-4 --warmup-steps 0 \
    --amp --amp-dtype fp16 \
    --init-ckpt "$CKPTDIR/stage3_pixel_run1/weights_epoch_10.pt" \
    --run-name stage3_pixel_gpu0_continue --log-every 50 --ckpt-every 4 \
    > "$LOGDIR/21_pixel_gpu0_continue.log" 2>&1 &
PID0=$!

# --- GPU1: capacity test, ViT-small, fresh.
log "START 22_pixel_gpu1_capacity (GPU 1)"
CUDA_VISIBLE_DEVICES=1 "$PY" stage3_pixel/train.py \
    --train-h5 "$DATADIR/stage1_vector_900_pixels_train.h5" \
    --val-h5 "$DATADIR/stage1_vector_900_pixels_val.h5" \
    --epochs 40 --batch-size 128 --lr 5e-4 --warmup-steps 500 \
    --amp --amp-dtype fp16 --vit-size small \
    --run-name stage3_pixel_gpu1_capacity --log-every 50 --ckpt-every 2 \
    > "$LOGDIR/22_pixel_gpu1_capacity.log" 2>&1 &
PID1=$!

# --- GPU3: multi-step training objective, fresh.
log "START 24_pixel_gpu3_multistep (GPU 3)"
CUDA_VISIBLE_DEVICES=3 "$PY" stage3_pixel/train.py \
    --train-h5 "$DATADIR/stage1_vector_900_pixels_train.h5" \
    --val-h5 "$DATADIR/stage1_vector_900_pixels_val.h5" \
    --epochs 55 --batch-size 128 --lr 5e-4 --warmup-steps 500 \
    --amp --amp-dtype fp16 --num-preds 3 \
    --run-name stage3_pixel_gpu3_multistep --log-every 50 --ckpt-every 3 \
    > "$LOGDIR/24_pixel_gpu3_multistep.log" 2>&1 &
PID3=$!

# --- GPU2: collect the 2700-episode pixel dataset first (CPU-only, no
# CUDA_VISIBLE_DEVICES restriction needed -- collection never touches a
# GPU -- runs concurrently with GPU0/1/3 training, same trick as every
# prior overnight batch), then train.
log "START 20_collect_pixels_2700 (CPU)"
QT_QPA_PLATFORM=offscreen "$PY" -m lewm collect ~/lewm_runs/stage1_vector_2700_pixels \
    --n-train 2700 --n-val 450 --n-test 450 --seed 3 --rgb \
    --note "Stage 3 data-scale point, matches vector 2700 seed" \
    > "$LOGDIR/20_collect_pixels_2700.log" 2>&1
log "collection finished"

"$PY" -m lewm validate ~/lewm_runs/stage1_vector_2700_pixels --loose \
    > "$LOGDIR/20b_validate_2700_pixels.log" 2>&1
log "validation done (see 20b log for pass/fail)"

"$PY" stage3_pixel/export_hdf5.py ~/lewm_runs/stage1_vector_2700_pixels \
    --name stage1_vector_2700_pixels --out-dir "$DATADIR" \
    > "$LOGDIR/20c_export_2700_pixels.log" 2>&1
log "2700-pixel export done"

log "START 23_pixel_gpu2_datascale (GPU 2)"
CUDA_VISIBLE_DEVICES=2 "$PY" stage3_pixel/train.py \
    --train-h5 "$DATADIR/stage1_vector_2700_pixels_train.h5" \
    --val-h5 "$DATADIR/stage1_vector_2700_pixels_val.h5" \
    --epochs 25 --batch-size 128 --lr 5e-4 --warmup-steps 500 \
    --amp --amp-dtype fp16 \
    --run-name stage3_pixel_gpu2_datascale --log-every 50 --ckpt-every 2 \
    > "$LOGDIR/23_pixel_gpu2_datascale.log" 2>&1 &
PID2=$!

log "all 4 GPU jobs launched: gpu0=$PID0 gpu1=$PID1 gpu2=$PID2 gpu3=$PID3"
wait $PID0 $PID1 $PID2 $PID3
log "ALL GPU JOBS DONE"
