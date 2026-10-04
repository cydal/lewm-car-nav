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

Wired and smoke-tested (`scripts/stage1_smoke.py`): the export, the model,
and the loss all run end-to-end on the 8-episode pilot training split, loss
drops as expected, no NaNs. That is not the Stage 1 experiment — it is the
same "prove the pipe before spending budget on it" step the brief asked for
at the dataset layer (section 20), applied to the training integration.
Running the actual experiment (real dataset size, real training budget,
reporting whether the model learns the dynamics) is still to come.
