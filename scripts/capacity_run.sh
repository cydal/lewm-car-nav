#!/bin/bash
# Model-capacity test: every experiment so far (300/900/2700 episodes,
# restricted/full obs, scratch/warm-started) used the exact same ~1.9M
# param architecture. The 900->2700 data-scaling jump mostly failing to
# repeat the 300->900 jump is evidence something other than data is now
# the binding constraint. This holds data fixed (900 episodes, the
# best-understood baseline) and the only thing varied is model size, to
# find out whether that constraint is capacity.

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

# ~2.5x the baseline (4.84M vs 1.92M params): embed_dim 128->192,
# encoder_hidden 256->384, predictor_depth 4->5, dim_head 32->40,
# mlp_dim 512->768. Batch size, lr, dataset, epoch budget all identical
# to the original 900-episode scratch run (which plateaued ~-40% to -43%
# by epoch 200-250) -- model size is the only changed variable.
run_step "15_capacity_900_big1_250ep" \
    "$PY" stage1_vector/train.py \
        --train-h5 "$DATADIR/stage1_vector_900_train.h5" \
        --val-h5 "$DATADIR/stage1_vector_900_val.h5" \
        --epochs 250 --batch-size 1024 --lr 5e-4 \
        --embed-dim 192 --encoder-hidden 384 --encoder-depth 2 \
        --predictor-depth 5 --predictor-heads 8 --predictor-dim-head 40 --predictor-mlp-dim 768 \
        --run-name stage_capacity_900_big1 --log-every 20 --ckpt-every 20

run_step "16_capacity_summary" "$PY" - <<'PYEOF'
import glob, os, re, subprocess

PY = "/home/ubuntu/miniconda3/envs/t3d/bin/python"
DATADIR = os.path.expanduser("~/.stable-wm/datasets")
REPO = os.path.expanduser("~/world_models/lewm-car-nav")
VAL_H5 = f"{DATADIR}/stage1_vector_900_val.h5"
RUN = "stage_capacity_900_big1"

MODEL_ARGS = ["--embed-dim", "192", "--encoder-hidden", "384", "--encoder-depth", "2",
              "--predictor-depth", "5", "--predictor-heads", "8",
              "--predictor-dim-head", "40", "--predictor-mlp-dim", "768"]

ckpt_dir = os.path.expanduser(f"~/lewm_runs/stage1_checkpoints/{RUN}")
ckpts = sorted(
    glob.glob(os.path.join(ckpt_dir, "weights_epoch_*.pt")),
    key=lambda p: int(re.search(r"epoch_(\d+)", p).group(1)),
) if os.path.isdir(ckpt_dir) else []

print(f"=== {RUN} ({len(ckpts)} checkpoints) ===")
for ckpt in ckpts:
    epoch = re.search(r"epoch_(\d+)", ckpt).group(1)
    out = subprocess.run(
        [PY, "stage1_vector/baseline_eval.py", "--h5", VAL_H5, "--ckpt", ckpt, *MODEL_ARGS],
        capture_output=True, text=True, cwd=REPO,
    )
    lines = [l for l in out.stdout.splitlines() if "pred_loss" in l]
    print(f"  epoch {epoch:>4}: " + " | ".join(lines))
    if out.returncode != 0:
        print("  stderr:", out.stderr[-500:])
PYEOF

log "ALL DONE"
