"""Export a `lewm` dataset split to the HDF5 layout `stable_worldmodel` reads.

Stage 1 (brief section 1) trains the official LeWM code on our *vector*
observation, not pixels -- that is the whole point of the baseline, to find
out whether the objective learns the dynamics before perception is in the
loop. `stable_worldmodel.data.formats.hdf5.HDF5Writer` wants one flat
per-step array per column, length-matched within an episode; our collector
stores T+1 observations against T actions (the alignment invariant other
code in this repo checks), so `action` gets one NaN padding row at the end
to make the two line up. `train.py` in the official repo already expects
exactly this -- it does `action = torch.nan_to_num(action, 0.0)` because,
per its own comment, NaNs "occur at sequence boundaries".

We write one `.h5` file per split rather than one file for the whole
dataset: the official `train.py` calls `spt.data.random_split` on a single
loaded dataset, which splits at the sample-window level, not the episode
level. That would let train and val windows share an episode -- exactly what
brief section 16 ("do not randomly split frames from the same episode")
rules out. Writing separate files sidesteps the question instead of
re-arguing it: there is no combined dataset for `random_split` to cut
across, since the episode-level split already happened at collection time.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from lewm.dataset import Dataset  # noqa: E402


def _episode_to_columns(ep, obs_key="obs_vector", keep_mask=None):
    """One episode -> `{column: (T+1, ...) array}`, action NaN-padded to T+1.

    `keep_mask` is the Stage 2 (brief section 2) hook: a boolean channel
    mask (from `Dataset.restricted_mask`) applied to the vector observation
    before writing, so "restricted" and "full" views are exported from the
    exact same underlying episodes -- the ablation is purely about what the
    model sees, not a different collection.
    """
    obs = np.asarray(ep[obs_key], dtype=np.float32)
    if keep_mask is not None:
        obs = obs[:, keep_mask]
    action = np.asarray(ep["action"], dtype=np.float32)
    pad = np.full((1, action.shape[1]), np.nan, dtype=np.float32)
    action = np.concatenate([action, pad], axis=0)
    assert obs.shape[0] == action.shape[0], (
        f"obs/action length mismatch after padding: {obs.shape[0]} vs {action.shape[0]}")
    return {"state": obs, "action": action}


def export_split(dataset_root, split, out_path, obs_key="obs_vector", mode="overwrite",
                  keep_mask=None):
    """Write every episode in `split` to `out_path` as a stable_worldmodel HDF5 file.

    Returns (n_episodes, n_steps) written.
    """
    from stable_worldmodel.data.formats.hdf5 import HDF5Writer

    ds = Dataset(dataset_root)
    episode_ids = ds.episode_ids(split)
    if not episode_ids:
        raise ValueError(f"split {split!r} has no episodes in {dataset_root!r}")

    n_steps = 0
    with HDF5Writer(out_path, mode=mode) as writer:
        for eid in episode_ids:
            with ds.episode(eid) as ep:
                cols = _episode_to_columns(ep, obs_key=obs_key, keep_mask=keep_mask)
                writer.write_episode(cols)
                n_steps += cols["state"].shape[0]
    return len(episode_ids), n_steps


def export_dataset(dataset_root, out_dir, splits=("train", "val", "test"), name="carnav_pilot",
                    view="vector_full", drop=None):
    """Write one `<name>_<split>.h5` per split into `out_dir`. Returns a dict of paths.

    `view="vector_restricted"` applies `Dataset.restricted_mask(drop)` (brief
    section 2 / Stage 2) before writing -- same episodes, same actions,
    fewer observation channels.
    """
    if view not in ("vector_full", "vector_restricted"):
        raise ValueError(f"view must be 'vector_full' or 'vector_restricted', got {view!r}")
    keep_mask = Dataset(dataset_root).restricted_mask(drop) if view == "vector_restricted" else None

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for split in splits:
        out_path = os.path.join(out_dir, f"{name}_{split}.h5")
        n_ep, n_steps = export_split(dataset_root, split, out_path, keep_mask=keep_mask)
        print(f"  {split}: {n_ep} episodes, {n_steps} steps -> {out_path}")
        paths[split] = out_path
    return paths


if __name__ == "__main__":
    import argparse

    from stable_worldmodel.data.utils import get_cache_dir

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset_root", help="a `lewm` dataset directory, e.g. ~/lewm_runs/pilot_001")
    p.add_argument("--name", default="carnav_pilot")
    p.add_argument("--out-dir", default=None,
                   help="defaults to $STABLEWM_HOME/datasets (same cache stable_worldmodel reads from)")
    p.add_argument("--view", default="vector_full", choices=["vector_full", "vector_restricted"],
                   help="vector_restricted drops the dataset's configured restricted_drop blocks "
                        "(default: nav, dynamics) -- Stage 2")
    args = p.parse_args()

    out_dir = args.out_dir or str(get_cache_dir(sub_folder="datasets"))
    print(f"exporting {args.dataset_root!r} ({args.view}) -> {out_dir!r}")
    export_dataset(args.dataset_root, out_dir, name=args.name, view=args.view)
