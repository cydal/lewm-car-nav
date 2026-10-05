#!/bin/bash
# Stage 2 (restricted observations) + one more point on the data-scaling
# curve (2700 episodes) + one hardening reseed. Sequential on the single
# GPU; the 2700-episode collection (CPU-only) runs concurrently with Stage
# 2's GPU training to not cost extra wall time, same trick as
# overnight_run.sh. Each step logs to its own file; failures are caught
# per-step.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY=/home/ubuntu/miniconda3/envs/t3d/bin/python
LOGDIR=~/lewm_runs/overnight_logs
CKPTDIR=~/lewm_runs/stage1_checkpoints
DATADIR=~/.stable-wm/datasets
mkdir -p "$LOGDIR"

export WANDB_API_KEY=$("$PY" -c "
for line in open('/home/ubuntu/world_models/le-wm/.env'):
    line = line.strip()
    if line.startswith('WAND_API='):
        print(line.split('=', 1)[1])
")

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGDIR/_timeline2.log"; }

run_step() {
    local name="$1"; shift
    log "START $name"
    if "$@" > "$LOGDIR/$name.log" 2>&1; then
        log "OK    $name"
    else
        log "FAIL  $name (see $LOGDIR/$name.log)"
    fi
}

# --- Step 1 (GPU): Stage 2, restricted observations (59D: drop nav +
# dynamics). Same 900 episodes, same actions as the full-vector run --
# already exported to $DATADIR/stage1_vector_900_restricted_{split}.h5.
# 300 epochs (not 400): the full-vector run on the same data plateaued by
# ~epoch 250, no reason to expect this needs longer.
run_step "08_stage2_restricted_300ep" \
    "$PY" stage1_vector/train.py \
        --train-h5 "$DATADIR/stage1_vector_900_restricted_train.h5" \
        --val-h5 "$DATADIR/stage1_vector_900_restricted_val.h5" \
        --epochs 300 --batch-size 1024 --lr 5e-4 \
        --run-name stage2_restricted_900 --log-every 20 --ckpt-every 25 &
STAGE2_PID=$!

# --- Step 2 (parallel, CPU): collect the 3rd data-scaling point. 3x the
# 900-episode set, seed=3 (distinct from seed=1 (300 eps) and seed=2
# (900 eps)).
QT_QPA_PLATFORM=offscreen run_step "09_collect_2700" \
    "$PY" -m lewm collect ~/lewm_runs/stage1_vector_2700 \
        --n-train 2700 --n-val 450 --n-test 450 --seed 3 \
        --note "Stage 1 data-scaling curve, 3rd point (300 -> 900 -> 2700)"
COLLECT_PID=$!

wait "$STAGE2_PID"
log "stage2 training finished"

log "waiting on 09_collect_2700 (pid $COLLECT_PID) if still running"
wait "$COLLECT_PID"

run_step "10_validate_2700" \
    "$PY" -m lewm validate ~/lewm_runs/stage1_vector_2700

run_step "11_export_2700" \
    "$PY" stage1_vector/export_hdf5.py ~/lewm_runs/stage1_vector_2700 \
        --name stage1_vector_2700 --out-dir "$DATADIR"

# --- Step 3 (GPU): train on the 2700-episode set, from scratch, full
# vector view -- the third point on the data-scaling curve. Batch bumped
# to 2048: GPU memory was nowhere near used even at batch 1024
# (2.5 GiB of 15 GiB), and there's 3x the windows per epoch now.
run_step "12_scaling_2700_scratch_300ep" \
    "$PY" stage1_vector/train.py \
        --train-h5 "$DATADIR/stage1_vector_2700_train.h5" \
        --val-h5 "$DATADIR/stage1_vector_2700_val.h5" \
        --epochs 300 --batch-size 2048 --lr 5e-4 \
        --run-name stage1_vector_2700_scratch --log-every 20 --ckpt-every 25

# --- Step 4 (GPU): one hardening reseed of the 900-episode scratch run,
# to check whether the scratch-vs-warmstart gap (-40.5% vs -37.9% at
# epoch 400) is a real difference or within seed noise. Shorter budget
# (300 not 400 epochs) since the plateau is already known.
run_step "13_hardening_900scratch_seed1" \
    "$PY" stage1_vector/train.py \
        --train-h5 "$DATADIR/stage1_vector_900_train.h5" \
        --val-h5 "$DATADIR/stage1_vector_900_val.h5" \
        --epochs 300 --batch-size 1024 --lr 5e-4 --seed 1 \
        --run-name stage1_vector_bigdata_scratch_seed1 --log-every 20 --ckpt-every 25

# --- Step 5: baseline + rollout eval sweep across every new checkpoint,
# one summary file.
run_step "14_summary" "$PY" - <<'PYEOF'
import glob, os, re, subprocess

PY = "/home/ubuntu/miniconda3/envs/t3d/bin/python"
DATADIR = os.path.expanduser("~/.stable-wm/datasets")
REPO = os.path.expanduser("~/world_models/lewm-car-nav")

RUNS = {
    "stage2_restricted_900": f"{DATADIR}/stage1_vector_900_restricted_val.h5",
    "stage1_vector_2700_scratch": f"{DATADIR}/stage1_vector_2700_val.h5",
    "stage1_vector_bigdata_scratch_seed1": f"{DATADIR}/stage1_vector_900_val.h5",
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
            capture_output=True, text=True, cwd=REPO,
        )
        lines = [l for l in out.stdout.splitlines() if "pred_loss" in l]
        print(f"  epoch {epoch:>4}: " + " | ".join(lines))

    # rollout eval on the final checkpoint only
    last_ckpt = ckpts[-1]
    out = subprocess.run(
        [PY, "stage1_vector/rollout_eval.py", "--h5", val_h5, "--ckpt", last_ckpt, "--horizon", "25"],
        capture_output=True, text=True, cwd=REPO,
    )
    print(f"  rollout @ {os.path.basename(last_ckpt)}:")
    print("\n".join("    " + l for l in out.stdout.splitlines()))
PYEOF

log "ALL DONE"
