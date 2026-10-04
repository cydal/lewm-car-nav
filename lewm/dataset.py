"""
On-disk dataset: schema, writer, reader, splits and the three dataset views.

Layout
------

    <root>/
        manifest.json                 dataset-level config, schema, episode index, splits
        config.json                   the CollectConfig that produced it
        episodes/
            train/ep_000000.npz
            train/ep_000001.npz
            val/ep_000008.npz
            test/ep_000010.npz
        samples/                      (written by the pilot report only)
        PILOT_REPORT.md               (written by the pilot report only)

Splits are directories, not a column, so a frame from a training episode
physically cannot appear in the test set without moving a file. The manifest
repeats the mapping and `validate_dataset` checks the two agree.

One `.npz` per episode, each self-describing: it carries its own `meta` JSON
alongside the arrays, so a single episode file is readable and checkable
without the manifest. Whole episodes rather than flattened transitions because
LeWM is trained on temporal context (brief sections 14 and 17) -- a shuffled
transition store cannot hand back a contiguous run, which is the same reason
`replay/buffer.py` has an `EpisodeBuffer` next to its `ReplayBuffer`.

Temporal alignment
------------------

The only convention in here, and everything depends on it:

    observation-side records have length T+1
    action-side records have length T
    transition t is  (obs[t], action[t]) -> obs[t+1]

So `obs_vector[t]`, `rgb[t]` and `privileged[t]` are all the state the policy
*saw* when it chose `action[t]`, and index `t+1` is the consequence. There is
no separate `next_obs` array: storing one would double the RGB on disk and
create a second place for the off-by-one error the brief (section 13) is
specifically worried about. `transitions()` builds the `(obs, action,
next_obs)` triples by slicing, and `tests/test_lewm.py` pins the alignment with
a deterministic-action rollout that re-simulates the transition.

Arrays
------

Observation-side, length T+1:

| array              | dtype   | shape          | present when |
|--------------------|---------|----------------|--------------|
| `timestep`         | int32   | (T+1,)         | always |
| `obs_vector`       | float32 | (T+1, D)       | `capture_vector` |
| `rgb`              | uint8   | (T+1, H, W, 3) | `capture_rgb` |
| `privileged`       | float32 | (T+1, P)       | `capture_privileged_state` |
| `traffic_state`    | float32 | (T+1, K, 6)    | traffic on + `capture_traffic_state` |
| `pedestrian_state` | float32 | (T+1, Kp, 5)   | pedestrians on + `capture_pedestrian_state` |
| `signal_state`     | float32 | (T+1, S, 3)    | lights on + `capture_signal_state` |

Action-side, length T:

| array        | dtype   | shape  | notes |
|--------------|---------|--------|-------|
| `action`     | float32 | (T, 3) | exactly what was passed to `env.step`, already clipped |
| `reward`     | float32 | (T,)   | the env's own reward, unmodified |
| `terminated` | bool    | (T,)   | true only on the final step of a terminated episode |
| `truncated`  | bool    | (T,)   | likewise for a timeout |
| `crashed`    | bool    | (T,)   | `info["crashed"]` |
| `reached`    | int8    | (T,)   | `info["targets_reached_this_step"]` |
| `injected`   | bool    | (T,)   | the step was an off-nominal burst, not the controller's choice |

Metadata (`meta`, a JSON string): `episode_id`, `episode_index`, `split`,
`policy`, `environment_seed`, `episode_seed`, `length`, `reason`,
`episode_reward`, `targets` (the full waypoint list, world metres),
`signal_xy`, `obs_slices`, `privileged_names`, the block column names, the
camera configuration, `show_goal_beacon`, `dt`, `action_repeat`,
`vector_dim`, `image_size` and the wall-clock collection time.
"""

import json
import os

import numpy as np

SCHEMA_VERSION = 2

OBS_SIDE = ("timestep", "obs_vector", "rgb", "privileged",
            "traffic_state", "pedestrian_state", "signal_state")
ACT_SIDE = ("action", "reward", "terminated", "truncated", "crashed",
            "reached", "injected")

VIEW_MODES = ("vector_full", "vector_restricted", "rgb")


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------
def write_episode(path, arrays, meta, compress=False):
    """Write one episode to `path`, checking the length contract first.

    The check is here, at the one place episodes are created, rather than in
    the collector: a file that violates it is unusable and the error is far
    cheaper now than after a long collection run.
    """
    t_obs = len(arrays["timestep"])
    t_act = len(arrays["action"])
    if t_obs != t_act + 1:
        raise ValueError(f"observation-side length {t_obs} must be action-side "
                         f"length {t_act} + 1")
    for name in OBS_SIDE:
        if name in arrays and len(arrays[name]) != t_obs:
            raise ValueError(f"{name} has length {len(arrays[name])}, expected {t_obs}")
    for name in ACT_SIDE:
        if name in arrays and len(arrays[name]) != t_act:
            raise ValueError(f"{name} has length {len(arrays[name])}, expected {t_act}")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = dict(arrays)
    payload["meta"] = np.asarray(json.dumps(meta, sort_keys=True))
    save = np.savez_compressed if compress else np.savez
    save(path, **payload)
    return path


def write_manifest(root, config, episodes, extra=None):
    """Write `manifest.json` + `config.json`. `episodes` is a list of meta dicts."""
    os.makedirs(root, exist_ok=True)
    splits = {}
    for m in episodes:
        splits.setdefault(m["split"], []).append(m["episode_id"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "config": config.to_dict(),
        "splits": {k: sorted(v) for k, v in splits.items()},
        "n_episodes": len(episodes),
        "n_transitions": int(sum(m["length"] for m in episodes)),
        "episodes": episodes,
    }
    manifest.update(extra or {})
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    config.save(os.path.join(root, "config.json"))
    return manifest


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------
class Episode:
    """One episode's arrays plus its metadata.

    `Dataset.episode(...)` builds these lazily per call rather than caching:
    an RGB episode is ~7 MB at 64x64 and a dataset is meant to be streamed, not
    held. Arrays are materialised on access (`ep["rgb"]`) so opening an episode
    to read its metadata does not pay for its pixels.
    """

    def __init__(self, path):
        self.path = path
        self._npz = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(self._npz["meta"]))

    # --- dict-ish access to the arrays
    def __contains__(self, name):
        return name in self._npz.files

    def __getitem__(self, name):
        return self._npz[name]

    def get(self, name, default=None):
        return self._npz[name] if name in self._npz.files else default

    @property
    def arrays(self):
        return [f for f in self._npz.files if f != "meta"]

    @property
    def length(self):
        """T -- the number of transitions (one fewer than the observations)."""
        return int(self.meta["length"])

    @property
    def episode_id(self):
        return self.meta["episode_id"]

    def close(self):
        self._npz.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return (f"Episode({self.episode_id!r}, T={self.length}, "
                f"policy={self.meta.get('policy')!r}, split={self.meta.get('split')!r})")


def read_episode(path):
    return Episode(path)


class Dataset:
    """A collected dataset on disk: manifest, episodes, splits, views."""

    def __init__(self, root):
        self.root = root
        with open(os.path.join(root, "manifest.json")) as f:
            self.manifest = json.load(f)
        self._by_id = {m["episode_id"]: m for m in self.manifest["episodes"]}

    # ------------------------------------------------------------------
    @property
    def config(self):
        return self.manifest["config"]

    @property
    def splits(self):
        return self.manifest["splits"]

    def episode_ids(self, split=None):
        if split is None:
            return [m["episode_id"] for m in self.manifest["episodes"]]
        return list(self.manifest["splits"].get(split, []))

    def path_for(self, episode_id):
        m = self._by_id[episode_id]
        return os.path.join(self.root, m["path"])

    def episode(self, episode_id):
        return Episode(self.path_for(episode_id))

    def iter_episodes(self, split=None):
        for eid in self.episode_ids(split):
            with self.episode(eid) as ep:
                yield ep

    # ------------------------------------------------------------------
    def restricted_mask(self, drop=None):
        """Boolean keep-mask over the vector observation for the restricted view.

        Built from the `obs_slices` the env declared at collection time, which
        is why it is recorded per dataset: block widths move with
        `n_beams`/`n_lookahead`/`n_traffic_obs`, so a hard-coded offset here
        would silently mask the wrong channels for a differently configured run.
        """
        drop = tuple(self.config["restricted_drop"] if drop is None else drop)
        slices = self.manifest["obs_slices"]
        dim = self.manifest["vector_dim"]
        keep = np.ones(dim, dtype=bool)
        for name in drop:
            if name not in slices:
                raise KeyError(f"no observation block {name!r}; "
                               f"blocks: {sorted(slices)}")
            lo, hi = slices[name]
            keep[lo:hi] = False
        return keep

    def transitions(self, episode, mode=None, drop=None):
        """`(obs, action, next_obs)` plus labels for one episode, in one view.

        `mode` defaults to the dataset's own `observation_mode`. The privileged
        state comes back in its own keys and is *never* merged into `obs` --
        that separation is the point of section 6 of the brief.
        """
        mode = mode or self.config["observation_mode"]
        keep = self.restricted_mask(drop) if mode == "vector_restricted" else None
        return transitions(episode, mode, keep)


def transitions(episode, mode="vector_full", keep=None):
    """Slice one `Episode` into aligned transition arrays for a given view.

    `keep` is the boolean channel mask for `vector_restricted` (see
    `Dataset.restricted_mask`); ignored for the other modes.
    """
    if mode not in VIEW_MODES:
        raise ValueError(f"mode must be one of {VIEW_MODES}, got {mode!r}")

    if mode == "rgb":
        if "rgb" not in episode:
            raise KeyError(f"{episode.path} has no rgb array; "
                           "it was collected with capture_rgb=False")
        obs_all = episode["rgb"]
    else:
        if "obs_vector" not in episode:
            raise KeyError(f"{episode.path} has no obs_vector array; "
                           "it was collected with capture_vector=False")
        obs_all = episode["obs_vector"]
        if mode == "vector_restricted":
            if keep is None:
                raise ValueError("vector_restricted needs a channel mask "
                                 "(use Dataset.transitions, which builds one)")
            obs_all = obs_all[:, np.asarray(keep, dtype=bool)]

    out = {
        "obs": obs_all[:-1],
        "next_obs": obs_all[1:],
        "action": episode["action"],
        "reward": episode["reward"],
        "terminated": episode["terminated"],
        "truncated": episode["truncated"],
        "timestep": episode["timestep"][:-1],
        "episode_id": episode.episode_id,
    }
    priv = episode.get("privileged")
    if priv is not None:
        out["privileged"] = priv[:-1]
        out["next_privileged"] = priv[1:]
        out["privileged_names"] = episode.meta["privileged_names"]
    return out
