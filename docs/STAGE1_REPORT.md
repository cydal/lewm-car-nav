# Stage 1 report: full-vector baseline

Reference for `stage1_vector/`: what was trained, on what, how it was
evaluated, and the measured result. Written against
*LeWM_Dataset_Creation_Brief.md*; section numbers in brackets point back at
it. The companion document for the dataset layer itself is
[LEWM_DATASET.md](LEWM_DATASET.md) (stage 0).

## 1. What Stage 1 asks, and what counts as answering it

> Train a LeWM-style predictive world model using the full vector
> observation/state representation... Establish whether the LeWM-style
> objective can learn the simulator's dynamics when perception is not the
> bottleneck. This is a controlled baseline, not the final experiment. [§1]

This is a feasibility question, not a benchmark to optimize. Section 22 is
explicit that a huge dataset is not automatically better and the goal is to
establish learnability, not to maximize a number. The bar cleared here is:
**does the model learn something about the simulator's dynamics that a
trivial baseline does not**, measured on held-out episodes, and is that
effect real rather than an artifact of a specific run.

**Status: met.** See section 5.

## 2. Model

The official LeWM code ([`le-wm`](../../le-wm), Maes et al.), imported
unmodified except for one method. `jepa.JEPA.encode` is hardcoded to a ViT
pixel encoder (`info['pixels']` -> HuggingFace ViT -> CLS token);
`stage1_vector/model.py`'s `VectorJEPA` overrides only `encode()` to read a
`state` vector through an MLP stack instead. Everything downstream
(`predict`, `rollout`, `criterion`, the autoregressive predictor, the action
embedder, the SIGReg regularizer) is theirs, untouched.

| component | config | why |
|---|---|---|
| encoder | `VectorEncoder`, depth 2, hidden 256 -> embed_dim | stack of `module.MLP` blocks, residual; see `model.py` docstring for why depth 2 (not a ViT's 12) is the right order of magnitude for an already-feature-engineered vector |
| action encoder | `module.Embedder`, theirs, unmodified | `Conv1d(k=1)` + 2-layer MLP |
| predictor | `module.ARPredictor`, theirs, unmodified | depth 4, 8 heads, dim_head 32, mlp_dim 512, causal, AdaLN-zero action conditioning |
| regularizer | `module.SIGReg`, theirs, unmodified | knots=17, num_proj=1024, weight (lambda)=0.09 |
| embed_dim | 128 | |
| total params | 1,916,416 | |
| history_size / num_preds | 3 / 1 (4-step windows) | one-step-ahead prediction given 3 steps of context |

Loss: `pred_loss + lambda * sigreg_loss`, where `pred_loss =
(pred_emb - tgt_emb).pow(2).mean()` -- copied verbatim from
`le-wm/train.py:lejepa_forward`.

## 3. Data

Two datasets, both collected with the Stage 0 collector (`lewm/`),
vector-only (`capture_rgb=False`, cheap: ~540 steps/s):

| dataset | train / val / test episodes | seed | transitions (train) |
|---|---|---|---|
| `stage1_vector_001` | 300 / 50 / 50 | 1 | 160,086 |
| `stage1_vector_900` | 900 / 150 / 150 | 2 | ~480,000 |

Both pass the full Stage 0 validator (structural/temporal/visual/action/
dynamics/coverage checks). Exported to `stable_worldmodel`'s HDF5 format by
`stage1_vector/export_hdf5.py`, one file per split (not the official
`train.py`'s `random_split`, which cuts at the sample-window level and would
leak across the episode-level split -- see that module's docstring).

## 4. Training infrastructure notes

Two things worth recording because they weren't about the model:

- **Data loading was the bottleneck, not compute.** The generic
  `HDF5Dataset`/`DataLoader` path left the GPU at 67% utilization (460 MiB
  of 15 GiB used) -- one Python call per sample is noise for a video
  dataset, dead weight for a ~50 MB vector one. `stage1_vector/fast_loader.py`
  loads each split's arrays once, onto the GPU, and gathers a whole batch
  with one vectorized index op. Utilization 67% -> ~90%, epoch time 26s ->
  15s (300-episode set, batch 512).
- **cuDNN 9.24 can't load its runtime-compiled conv engine on this box's
  Tesla T4** (`CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`), unrelated to this
  model. Worked around with `torch.backends.cudnn.enabled = False`; the
  model's only conv is a kernel-size-1 `Conv1d`, so nothing is lost.

## 5. The baseline that makes the numbers mean something

A `pred_loss` number is meaningless without a reference point -- at
`dt=0.05s` (20 Hz), a 73-D state barely changes in one step, so a model
could post a "low" loss by doing nothing. `stage1_vector/baseline_eval.py`
compares the trained predictor against:

- **copy**: next embedding = current embedding (ignores the action entirely)
- **batch-mean**: ignores the input too (catches representational collapse)

Collapse is cleanly ruled out everywhere: batch-mean baseline sits at
0.78-0.85 against a trained loss of 0.01-0.02, a 40-80x gap, in every run
below. The informative comparison is **margin over copy**,
`(pred_loss - copy_loss) / copy_loss`.

## 6. Results

Five training regimes, all on held-out validation episodes:

| run | data | lr | epochs | margin over copy |
|---|---|---|---|---|
| run2 | 300 eps | 5e-4 | 40 | -7.1% (still widening when stopped) |
| run3 | 300 eps | **2e-3** (scaled to batch size) | 60 | **never beats copy**, +0.3% to +0.6% (worse) at every checkpoint |
| run4+run5 | 300 eps | 5e-4 | 600 (continued) | plateaus at **-11% to -16%**, flat from epoch ~250 on |
| bigdata_scratch | 900 eps | 5e-4 | 400 | rises to **-40% to -43%**, flat from epoch ~250 on |
| bigdata_warmstart | 900 eps | 5e-4 (warm-started from run4 ep 180) | 400 | peaks **-44.8%** at epoch 100, erodes to -37.9% by epoch 400 |

Three findings, in the order they were established:

1. **The effect is real, not a training-length artifact.** run3 is the
   control that shows what "doesn't generalize" looks like: it drives
   train loss down fast but never beats the trivial baseline on held-out
   data, at any of 5 checkpoints across 60 epochs. run2/4/5, at a sane
   learning rate, do the opposite -- they cross from worse-than-copy to
   better-than-copy and the margin holds under 600 epochs of further
   training. Same model, same data; the only thing that changes the
   outcome is a training-stability choice, not something about the task
   being unlearnable.
2. **The 300-episode result understated it by about 3x.** Pushing epoch
   count on 300 episodes plateaus at -11% to -16%, flat across 350 further
   epochs -- that's a real ceiling, not impatience. Tripling the dataset to
   900 episodes reaches -40% to -43%, in about a third of the epochs. Data
   scale was the actual lever; more training time on fixed data was not.
3. **Warm-starting trades a faster peak for a worse final answer.** The
   warm-started run posts the single best number of the whole batch
   (-44.8%, epoch 100) but drifts down afterward, ending below the
   from-scratch run on the same data (-37.9% vs -40.5% at epoch 400) --
   plausibly because it carries ~180 prior epochs of exposure into the run,
   so by epoch 400 it has seen far more total gradient steps than scratch
   has. One run each; a hypothesis, not a confirmed effect. Practical
   takeaway regardless: a warm-started run needs its own margin-vs-epoch
   curve, not the donor run's epoch budget copied over.

**Multi-step rollout (`rollout_eval.py`).** The training loss only ever
asks for one-step-ahead prediction. Feeding the trained predictor its own
predictions autoregressively (the same loop `JEPA.rollout` uses, flattened
to a single trace since that method is shaped for CEM's multi-candidate
sampling) shows the margin over copy *shrinks monotonically* with horizon
and **crosses to zero and reverses** -- the model becomes worse than doing
nothing -- around step 21-22 for `bigdata_scratch` and step 18-19 for
`bigdata_warmstart` (20 Hz: roughly 0.9-1.1 seconds):

| step | scratch margin | warmstart margin |
|---|---|---|
| 1 | -43.0% | -42.6% |
| 10 | -24.7% | -18.8% |
| 20 | -4.3% | +3.3% |
| 25 | +6.4% | +13.2% |

Good one-step prediction, trained with a one-step loss, does not imply
good multi-step rollout, and here it measurably stops helping past about a
second. This matters directly for Stage 5 (CEM/MPC needs a rollout, not a
single step) -- planning over this model as-is would be operating past the
horizon where it's known to help.

Full per-checkpoint numbers: `~/lewm_runs/overnight_logs/07_baseline_summary.log`
(not committed -- regenerable from the checkpoints, which also aren't
committed; see section 8). wandb project:
[wandb.ai/sijpapi/lewm-car-nav](https://wandb.ai/sijpapi/lewm-car-nav).

## 7. What this does not establish

Same spirit as [LEWM_DATASET.md §15](LEWM_DATASET.md#15-what-this-does-not-establish):

- **Single-step prediction only** (`num_preds=1`) was the *training*
  objective, and that's what the headline margins measure. Multi-step
  rollout *was* measured (section 6) and does compound badly: the margin
  reverses around 1 second. Any use of this model beyond one-step
  prediction needs its own evaluation at the relevant horizon, not an
  extrapolation from the one-step number.
- **One seed per configuration.** The margin numbers have no error bars;
  "-40.5% vs -37.9%" (scratch vs warm-start) is suggestive, not proven,
  without repeats.
- **The 900-episode margin may not be a ceiling.** Two data points
  (300, 900 episodes) show a large jump with no sign of bending yet.
  Whether another 3x keeps paying off is unknown and untested.
- **Not compared against Stage 2 or 3 yet.** The whole point of a
  "controlled baseline" is what later stages get measured against; that
  comparison doesn't exist until Stage 2 runs.
- **Not a claim about what the model represents.** Beating "copy" says the
  action and history carry predictive signal the model uses; it says
  nothing about *what* -- that's Stage 4 (latent probing).

## 8. Reproducing

```bash
python stage1_vector/export_hdf5.py ~/lewm_runs/stage1_vector_900   # or _001
python stage1_vector/train.py \
    --train-h5 ~/.stable-wm/datasets/stage1_vector_900_train.h5 \
    --val-h5 ~/.stable-wm/datasets/stage1_vector_900_val.h5 \
    --epochs 400 --batch-size 1024 --lr 5e-4 --run-name my_run
python stage1_vector/baseline_eval.py \
    --h5 ~/.stable-wm/datasets/stage1_vector_900_val.h5 \
    --ckpt ~/lewm_runs/stage1_checkpoints/my_run/weights_epoch_400.pt
```

Checkpoints, exported HDF5 files, and raw per-run logs are not committed
(regenerable from the collected dataset + a seed; see `.gitignore`). The
collected datasets themselves (`~/lewm_runs/stage1_vector_001`,
`~/lewm_runs/stage1_vector_900`) are also not committed, same reasoning as
the Stage 0 pilot -- but unlike the pilot, neither is checked into
`docs/` as a fixed reference artifact, since Stage 1's point was the
comparison across runs, not one dataset to inspect by hand.
