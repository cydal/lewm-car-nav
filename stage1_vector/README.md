# stage1_vector

Stage 1 of the brief: train the official LeWM model on the full 73-D vector
observation instead of pixels, to check whether the LeWM objective learns the
simulator's dynamics before perception is in the loop. Uses the paper's own
code (predictor, action embedder, SIGReg loss, training conventions) via a
sibling checkout of [`le-wm`](../../le-wm) — see
[../README.md](../README.md#finding-the-official-lewm-code). The only piece of
their code this doesn't reuse unmodified is `jepa.JEPA.encode`, which is
hardcoded to a ViT pixel encoder; `model.VectorJEPA` overrides just that one
method to read the `state` column through `module.MLP` instead. Everything
downstream — `predict`, `rollout`, `criterion`, `get_cost` — only touches
embeddings and needed no changes.

## Install

```bash
pip install "stable-worldmodel[train,format]"
```

No extra for `env` (ogbench/pygame/craftax/...) — we bring our own
environment and don't need theirs.

## Files

| path | what it is |
|---|---|
| `export_hdf5.py` | converts a `lewm` dataset split into the HDF5 layout `stable_worldmodel` reads |
| `model.py` | `VectorJEPA` and `build_model()` |
| `fast_loader.py` | in-memory windowed batching (bypasses `HDF5Dataset`'s per-sample Python overhead) |
| `wandb_env.py` | loads `WANDB_API_KEY` from `le-wm/.env` |
| `train.py` | the training loop: wandb logging, periodic validation and checkpointing, `--init-ckpt`/`--start-epoch` for continuing a run |
| `baseline_eval.py` | compares a checkpoint against trivial "copy" and "batch-mean" baselines -- see `docs/STAGE1_REPORT.md` for why this matters |
| `rollout_eval.py` | multi-step autoregressive rollout error vs. the same baselines, extended to a horizon |

## Why one `.h5` per split, not one dataset `random_split` like `train.py`

The official `train.py` loads a single dataset and calls
`spt.data.random_split` on it to get train/val. That splits at the
sample-window level, which can put two overlapping windows from the same
episode on opposite sides of the split — exactly what brief section 16 rules
out ("do not randomly split frames from the same episode"). We already have
a correct episode-level train/val/test split from Stage 0 collection, so
`export_hdf5.export_dataset` writes one file per split and callers load each
with its own `HDF5Dataset` / `DataLoader` — no combined dataset for
`random_split` to cut across.

## Status

**Complete.** The objective learns real, held-out-validated dynamics that
scale with data (margin over a trivial baseline: -14% at 300 episodes,
-43% at 900), and that effect was shown to be real rather than a
training-length artifact (a mistuned learning rate that overfits never
clears the baseline at all, at any checkpoint). Full results, the
evaluation methodology, and what this does and doesn't establish:
[docs/STAGE1_REPORT.md](../docs/STAGE1_REPORT.md).
