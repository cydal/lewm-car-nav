"""Export a `lewm` dataset split to the HDF5 layout `stable_worldmodel` reads.

Same alignment logic as `stage1_vector/export_hdf5.py` -- action gets one
NaN-padded row so it matches the T+1 length of the observation column --
just swapping `obs_vector` for `rgb` and the column name `state` for
`pixels`. See that module's docstring for why one `.h5` per split, not the
official `train.py`'s `random_split`.

Privileged/vector state is *not* carried into this export. It still exists
in the original `lewm` dataset directory (`ds.episode(eid)["privileged"]`,
`["obs_vector"]`) for whenever Stage 4 (latent probing) needs it; there is
no reason to duplicate it into a file whose whole purpose is pixel-only
training input.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from lewm.dataset import Dataset  # noqa: E402


def _episode_to_columns(ep):
    """One episode -> `{column: (T+1, ...) array}`, action NaN-padded to T+1."""
    pixels = np.asarray(ep["rgb"], dtype=np.uint8)
    action = np.asarray(ep["action"], dtype=np.float32)
    pad = np.full((1, action.shape[1]), np.nan, dtype=np.float32)
    action = np.concatenate([action, pad], axis=0)
    assert pixels.shape[0] == action.shape[0], (
        f"pixels/action length mismatch after padding: {pixels.shape[0]} vs {action.shape[0]}")
    return {"pixels": pixels, "action": action}


def export_split(dataset_root, split, out_path, mode="overwrite"):
    """Write every episode in `split` to `out_path`. Returns (n_episodes, n_steps)."""
    from stable_worldmodel.data.formats.hdf5 import HDF5Writer

    ds = Dataset(dataset_root)
    episode_ids = ds.episode_ids(split)
    if not episode_ids:
        raise ValueError(f"split {split!r} has no episodes in {dataset_root!r}")
    if not ds.config.get("capture_rgb"):
        raise ValueError(f"{dataset_root!r} was collected with capture_rgb=False; "
                          "no 'rgb' array to export")

    n_steps = 0
    with HDF5Writer(out_path, mode=mode) as writer:
        for eid in episode_ids:
            with ds.episode(eid) as ep:
                cols = _episode_to_columns(ep)
                writer.write_episode(cols)
                n_steps += cols["pixels"].shape[0]
    return len(episode_ids), n_steps


def export_dataset(dataset_root, out_dir, splits=("train", "val", "test"), name="pixels"):
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
    p.add_argument("dataset_root")
    p.add_argument("--name", default="pixels")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    out_dir = args.out_dir or str(get_cache_dir(sub_folder="datasets"))
    print(f"exporting {args.dataset_root!r} -> {out_dir!r}")
    export_dataset(args.dataset_root, out_dir, name=args.name)
