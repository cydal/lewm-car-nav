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

import numpy as np

from lewm.dataset import Dataset


def _episode_to_columns(ep, obs_key="obs_vector"):
    """One episode -> `{column: (T+1, ...) array}`, action NaN-padded to T+1."""
    obs = np.asarray(ep[obs_key], dtype=np.float32)
    action = np.asarray(ep["action"], dtype=np.float32)
    pad = np.full((1, action.shape[1]), np.nan, dtype=np.float32)
    action = np.concatenate([action, pad], axis=0)
    assert obs.shape[0] == action.shape[0], (
        f"obs/action length mismatch after padding: {obs.shape[0]} vs {action.shape[0]}")
    return {"state": obs, "action": action}


def export_split(dataset_root, split, out_path, obs_key="obs_vector", mode="overwrite"):
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
                cols = _episode_to_columns(ep, obs_key=obs_key)
                writer.write_episode(cols)
                n_steps += cols["state"].shape[0]
    return len(episode_ids), n_steps


def export_dataset(dataset_root, out_dir, splits=("train", "val", "test"), name="carnav_pilot"):
    """Write one `<name>_<split>.h5` per split into `out_dir`. Returns a dict of paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for split in splits:
        out_path = os.path.join(out_dir, f"{name}_{split}.h5")
        n_ep, n_steps = export_split(dataset_root, split, out_path)
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
    args = p.parse_args()

    out_dir = args.out_dir or str(get_cache_dir(sub_folder="datasets"))
    print(f"exporting {args.dataset_root!r} -> {out_dir!r}")
    export_dataset(args.dataset_root, out_dir, name=args.name)
