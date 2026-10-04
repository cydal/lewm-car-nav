"""
The collector: wraps `CarNavEnv` from the outside and records trajectories.

`LeWMDatasetEnv` is plain composition, not a `gymnasium.Wrapper` -- the same
choice, for the same reason, as `replay.wrapper.RecordingWrapper` and
`wrappers.RewardOverrideWrapper`: gymnasium is an optional dependency in this
repo. `reset`/`step` forward unchanged and return exactly what the env
returned, so anything that can drive `CarNavEnv` can drive this, and the
simulator never learns that it exists.

What it records, and when:

    reset()      -> observation record 0       (obs[0], privileged[0], rgb[0])
    step(a_0)    -> action record 0            (a_0, r_0, flags)
                    observation record 1       (obs[1], ...)
    step(a_1)    -> action record 1
                    observation record 2
    ...

so the invariant `obs[t] + action[t] -> obs[t+1]` is a property of the call
order rather than something to be reassembled later. The RGB frame is taken
from the env's own `obs_type="both"` dict, not from a separate `env.render()`
call after the step, so the frame and the vector are two halves of one
`_observe()` and cannot drift apart by a step.

`collect()` is the batch entry point: it builds the env (and the renderer, if
RGB is wanted), allocates seeds and the policy mixture per split, runs the
episodes and writes them out with a manifest.

Seeding, which is what makes a trajectory reproducible from its metadata alone:

    SeedSequence(environment_seed)
        |- train block   -> per-episode child -> (episode_seed, policy_seed)
        |- val block     -> ...
        |- test block    -> ...
        `- mixture block -> which profile each episode gets

Every episode records both integers, so

    env.reset(seed=meta["episode_seed"])
    policy = make_policy(meta["policy"], env, default_rng(meta["policy_seed"]))

replays it exactly -- the check `lewm.validate` and `tests/test_lewm.py` both
run. Train/val/test draw from *different* spawned blocks, so no episode seed is
shared between splits by construction rather than by a filtering step that
could be forgotten.
"""

import os
import platform
import subprocess
import sys
import time

import numpy as np

import carnav

from .config import CollectConfig
from .dataset import SCHEMA_VERSION, write_episode, write_manifest
from .paths import carnav_root
from .policies import make_policy, policy_schedule
from .state import (PEDESTRIAN_STATE_COLS, PrivilegedRecorder,
                    SIGNAL_STATE_COLS, TRAFFIC_STATE_COLS)

SPLITS = ("train", "val", "test")


def _git_sha(path=None):
    """Commit at `path` (default: this repo), or None if it is not a checkout.

    Two shas go in every manifest, not one: the collector and the simulator are
    separate repositories, so reproducing a dataset means pinning both. A
    dataset that records only this one would be untraceable the moment the env
    moved a line of physics.
    """
    try:
        here = path or os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run(["git", "-C", here, "rev-parse", "--verify", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None  # e.g. a repo with no commits yet: bare rev-parse would
                          # otherwise echo the literal string "HEAD" as if it were a sha
        return out.stdout.strip() or None
    except Exception:
        return None


_RENDERER = None        # process-wide, for the same reason ShowBase is


def build_renderer(config):
    """The Panda3D renderer for an RGB run, or None for a vector-only run.

    Reused process-wide, not rebuilt per call. One `ShowBase` per process is a
    hard Panda3D constraint (INTEGRATION.md) and `PandaRenderer` attaches its
    lights, fog, car and waypoint posts to *that shared* scene graph, while
    `close()` deliberately drops only the per-episode geometry. So a second
    renderer in one process does not replace the first, it adds to it: double
    ambient and sun light, a ghost car parked wherever the previous run ended,
    and a second render texture on the window. The frames are then subtly
    brighter than the ones already on disk, which is exactly the kind of
    difference that would quietly invalidate a dataset -- it was caught here by
    the replay check failing on `rgb at t=0` while the vector matched.

    `size` is the one setting that cannot be changed after construction (the
    offscreen buffer is sized when the window is made), so a request for a
    different size is an error telling the caller to use its own process. The
    camera rig and the beacon flag are re-applied on reuse, since both are read
    per frame.
    """
    global _RENDERER
    if not config.capture_rgb:
        return None
    from render.panda_renderer import PandaRenderer
    want = (config.image_size, config.image_size)
    if _RENDERER is None:
        _RENDERER = PandaRenderer(offscreen=True, size=config.image_size,
                                  show_goal_beacon=config.show_goal_beacon,
                                  **config.camera)
        return _RENDERER
    if _RENDERER.size != want:
        raise RuntimeError(
            f"this process already has a {_RENDERER.size} renderer and Panda3D "
            f"fixes the offscreen buffer size at construction; collecting at "
            f"{want} needs a fresh process (see lewm/pilot.py, which shells out "
            f"to `python -m lewm beacon` for precisely this reason)")
    _RENDERER.show_goal_beacon = config.show_goal_beacon
    for key, val in config.camera.items():
        if key == "fov":
            _RENDERER.base.camLens.setFov(val)
        else:
            setattr(_RENDERER, key, val)
    return _RENDERER


def build_env(config, renderer=None):
    """`carnav.make` with the collection config's simulator settings.

    Everything simulator-side comes from `config.env_kwargs` and is routed by
    `carnav.make`, which raises on an unknown keyword -- so a typo in a
    collection config fails here rather than producing a valid-looking dataset
    of the wrong task.
    """
    return carnav.make(obs_type=config.obs_type, renderer=renderer,
                       image_size=config.image_size, **config.env_kwargs)


class LeWMDatasetEnv:
    """Records a trajectory around any `CarNavEnv`-shaped env. Opt-in, additive.

        env = carnav.make(obs_type="both", renderer=r, image_size=64)
        rec = LeWMDatasetEnv(env, CollectConfig(capture_rgb=True))
        obs, info = rec.reset(seed=7)
        while True:
            obs, r, te, tr, info = rec.step(policy.act(obs["vector"]))
            if te or tr:
                break
        arrays = rec.arrays()          # the episode, ready for write_episode

    Not a buffer: one episode at a time, cleared by `reset`. Use `arrays()`
    before the next reset.
    """

    def __init__(self, env, config=None):
        self.env = env
        self.config = config or CollectConfig()
        c = self.config
        self.recorder = PrivilegedRecorder(
            env,
            n_traffic_state=c.n_traffic_state,
            n_pedestrian_state=c.n_pedestrian_state,
            traffic_state=c.capture_traffic_state,
            signal_state=c.capture_signal_state,
            pedestrian_state=c.capture_pedestrian_state,
        ) if c.capture_privileged_state else None
        self._clear()

    # ------------------------------------------------------------------
    def _clear(self):
        self._obs_vector = []
        self._rgb = []
        self._privileged = []
        self._traffic_state = []
        self._pedestrian_state = []
        self._signal_state = []
        self._action = []
        self._reward = []
        self._terminated = []
        self._truncated = []
        self._crashed = []
        self._reached = []
        self._annotations = {}
        self.last_info = None

    def _record_obs(self, obs):
        c = self.config
        if c.capture_vector:
            vec = obs["vector"] if isinstance(obs, dict) else obs
            self._obs_vector.append(np.asarray(vec, dtype=np.float32))
        if c.capture_rgb:
            img = obs["image"] if isinstance(obs, dict) else obs
            self._rgb.append(np.asarray(img, dtype=np.uint8))
        if self.recorder is not None:
            self._privileged.append(self.recorder.row())
            if self.recorder.n_traffic_state:
                self._traffic_state.append(self.recorder.traffic_state())
            if self.recorder.n_pedestrian_state:
                self._pedestrian_state.append(self.recorder.pedestrian_state())
            if self.recorder.want_signal_state:
                self._signal_state.append(self.recorder.signal_state())

    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        self._clear()
        obs, info = self.env.reset(**kwargs)
        self._record_obs(obs)
        self.last_info = info
        return obs, info

    def step(self, action):
        # Record the action *as passed*, so the stored action is the one the
        # simulator integrated. `CarNavEnv.step` clips internally, so an
        # unclipped command would be stored as something that never happened;
        # the policies in `lewm/policies.py` clip for this reason, and this
        # mirror-clip covers any other caller.
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(3), -1.0, 1.0)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._action.append(action)
        self._reward.append(np.float32(reward))
        self._terminated.append(bool(terminated))
        self._truncated.append(bool(truncated))
        self._crashed.append(bool(info.get("crashed", False)))
        self._reached.append(np.int8(info.get("targets_reached_this_step", 0)))
        self._record_obs(obs)
        self.last_info = info
        return obs, reward, terminated, truncated, info

    def annotate(self, **kwargs):
        """Append caller-side per-step labels (one value per `step`).

        Used for `injected`: whether the action came from the off-nominal
        injector rather than the controller. That is knowledge only the caller
        has -- the env cannot tell a deliberate hard brake from a chosen one --
        and it is what makes the recovery slice of the dataset addressable
        instead of just present.
        """
        for k, v in kwargs.items():
            self._annotations.setdefault(k, []).append(v)

    # ------------------------------------------------------------------
    def arrays(self):
        """The recorded episode as the arrays `write_episode` expects."""
        t_act = len(self._action)
        out = {
            "timestep": np.arange(t_act + 1, dtype=np.int32),
            "action": np.asarray(self._action, dtype=np.float32).reshape(t_act, 3),
            "reward": np.asarray(self._reward, dtype=np.float32),
            "terminated": np.asarray(self._terminated, dtype=bool),
            "truncated": np.asarray(self._truncated, dtype=bool),
            "crashed": np.asarray(self._crashed, dtype=bool),
            "reached": np.asarray(self._reached, dtype=np.int8),
        }
        if self._obs_vector:
            out["obs_vector"] = np.stack(self._obs_vector)
        if self._rgb:
            out["rgb"] = np.stack(self._rgb)
        if self._privileged:
            out["privileged"] = np.stack(self._privileged)
        if self._traffic_state:
            out["traffic_state"] = np.stack(self._traffic_state)
        if self._pedestrian_state:
            out["pedestrian_state"] = np.stack(self._pedestrian_state)
        if self._signal_state:
            out["signal_state"] = np.stack(self._signal_state)
        for k, v in self._annotations.items():
            arr = np.asarray(v)
            if len(arr) != t_act:
                raise ValueError(f"annotation {k!r} has {len(arr)} values for "
                                 f"{t_act} steps")
            out[k] = arr
        return out

    def __getattr__(self, name):
        # Passthrough for .cfg, .car, .city, .action_space, .obs_slices, ...
        # Only reached when this wrapper has no such attribute, so reset/step
        # above always win.
        return getattr(self.env, name)


# ----------------------------------------------------------------------
# Batch collection
# ----------------------------------------------------------------------
def _seed_plan(config):
    """Per-split lists of (episode_seed, policy_seed, policy_name).

    Separate spawned blocks per split, so train/val/test seeds cannot collide,
    and a separate block for the mixture draw, so changing the policy shares
    does not move which cities the episodes are collected in.

    The mixture is apportioned once over the whole dataset and then dealt out
    to the splits, not apportioned per split. `policy_mix` is a dataset-level
    statement (brief section 7) and the two only agree when every split is
    large: at the pilot's 8/2/2, per-split largest-remainder rounding drops the
    5% `recovery` share to zero episodes in all three splits, so the dataset
    would contain none of the off-nominal bursts the profile exists to produce
    while its manifest still claimed 5%. Dealing from one shuffled schedule
    gives the exact overall composition at any size; small val/test splits then
    hold a sample of the mixture rather than a miniature of it, which is the
    honest trade -- two episodes cannot represent six profiles either way.
    """
    root = np.random.SeedSequence(config.environment_seed)
    train_ss, val_ss, test_ss, mix_ss = root.spawn(4)
    blocks = dict(zip(SPLITS, (train_ss, val_ss, test_ss)))
    mix_rng = np.random.default_rng(mix_ss)
    schedule = policy_schedule(config.n_episodes, config.policy_mix, mix_rng)

    plan = {}
    cursor = 0
    for split in SPLITS:
        n = config.split_counts[split]
        names = schedule[cursor:cursor + n]
        cursor += n
        rows = []
        for i, child in enumerate(blocks[split].spawn(n)):
            env_ss, pol_ss = child.spawn(2)
            rows.append({
                "episode_seed": int(env_ss.generate_state(1, dtype=np.uint32)[0]),
                "policy_seed": int(pol_ss.generate_state(1, dtype=np.uint32)[0]),
                "policy": names[i],
            })
        plan[split] = rows
    return plan


def collect_episode(rec, policy, episode_seed, max_steps=None):
    """Drive one episode to its end and return (arrays, stats).

    `max_steps` is a collection-side cap only -- it never changes the env's own
    `max_episode_steps`, and an episode cut by it is marked `reason="cut"` with
    neither `terminated` nor `truncated` set on its last step, so a consumer
    bootstrapping values can tell a cut from a real timeout.
    """
    obs, info = rec.reset(seed=episode_seed)
    policy.reset()
    t0 = time.perf_counter()
    steps = 0
    reason = "cut"
    while True:
        action = policy.act(obs, info)
        obs, _, terminated, truncated, info = rec.step(action)
        rec.annotate(injected=bool(getattr(policy, "last_injected", False)))
        steps += 1
        if terminated or truncated:
            reason = info.get("reason")
            break
        if max_steps is not None and steps >= max_steps:
            break
    wall = time.perf_counter() - t0
    stats = {
        "length": steps,
        "reason": reason,
        "episode_reward": float(info.get("episode_reward", 0.0)),
        "targets_reached": int(info.get("targets_reached", 0)),
        "red_light_violations": int(info.get("red_light_violations", 0)),
        "crash_with": info.get("crash_with"),
        "wall_s": wall,
        "steps_per_s": steps / wall if wall > 0 else float("nan"),
    }
    return rec.arrays(), stats


def _episode_meta(rec, config, split, index, plan_row, stats, path, when):
    env = rec.env
    recorder = rec.recorder
    meta = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": f"{split}/ep_{index:06d}",
        "episode_index": index,
        "split": split,
        "path": path,
        "policy": plan_row["policy"],
        "environment_seed": config.environment_seed,
        "episode_seed": plan_row["episode_seed"],
        "policy_seed": plan_row["policy_seed"],
        "targets": [[float(x), float(y)] for x, y in env.targets],
        "n_targets": len(env.targets),
        "signal_xy": (recorder.signal_xy().tolist() if recorder is not None
                      else [[float(x), float(y)] for x, y in env.city.signals]),
        "obs_slices": {k: [sl.start, sl.stop] for k, sl in env.obs_slices.items()},
        "vector_dim": int(env.vector_dim),
        "image_size": config.image_size if config.capture_rgb else None,
        "camera": dict(config.camera),
        "show_goal_beacon": config.show_goal_beacon,
        "dt": float(env.cfg.dt),
        "action_repeat": int(env.cfg.action_repeat),
        "max_episode_steps": int(env.cfg.max_episode_steps),
        "privileged_names": list(recorder.names) if recorder is not None else [],
        "block_columns": {
            "traffic_state": list(TRAFFIC_STATE_COLS),
            "pedestrian_state": list(PEDESTRIAN_STATE_COLS),
            "signal_state": list(SIGNAL_STATE_COLS),
        },
        "collected_at": when,
    }
    meta.update(stats)
    return meta


def collect(config, out_dir, progress=True, log=print):
    """Collect a dataset into `out_dir` and write its manifest. Returns a summary.

    Deliberately single-process: the renderer cannot cross a process boundary
    (INTEGRATION.md), and a pilot is meant to be cheap to regenerate rather than
    fast. Scaling this out is a per-split job for later -- the seed plan already
    makes it embarrassingly parallel, since each episode's seeds are a pure
    function of `(environment_seed, split, index)`.
    """
    if isinstance(config, dict):
        config = CollectConfig.from_dict(config)
    renderer = build_renderer(config)
    env = build_env(config, renderer)
    rec = LeWMDatasetEnv(env, config)
    plan = _seed_plan(config)
    when = time.strftime("%Y-%m-%dT%H:%M:%S")

    metas = []
    t_start = time.perf_counter()
    bytes_written = 0
    try:
        index = 0
        for split in SPLITS:
            for row in plan[split]:
                policy = make_policy(row["policy"], env,
                                     np.random.default_rng(row["policy_seed"]))
                arrays, stats = collect_episode(rec, policy, row["episode_seed"],
                                                max_steps=config.max_steps)
                rel = os.path.join("episodes", split, f"ep_{index:06d}.npz")
                meta = _episode_meta(rec, config, split, index, row, stats, rel, when)
                write_episode(os.path.join(out_dir, rel), arrays, meta,
                              compress=config.compress)
                bytes_written += os.path.getsize(os.path.join(out_dir, rel))
                metas.append(meta)
                if progress:
                    log(f"  [{index + 1:>3}/{config.n_episodes}] {split:<5} "
                        f"{row['policy']:<16} T={stats['length']:>4} "
                        f"{str(stats['reason']):<8} "
                        f"{stats['steps_per_s']:>6.0f} steps/s")
                index += 1
    finally:
        # The env owns the renderer it was given only in the sense that it will
        # close it; closing here keeps a long-running process (tests, a report
        # script) from leaking the scene graph between collections.
        env.close()

    wall = time.perf_counter() - t_start
    n_steps = int(sum(m["length"] for m in metas))
    extra = {
        "obs_slices": metas[0]["obs_slices"] if metas else {},
        "vector_dim": metas[0]["vector_dim"] if metas else 0,
        "privileged_names": metas[0]["privileged_names"] if metas else [],
        "block_columns": metas[0]["block_columns"] if metas else {},
        "collected_at": when,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "git_sha": _git_sha(),                        # this repo (lewm)
            "carnav_root": carnav_root(),
            "carnav_git_sha": _git_sha(carnav_root()),    # the simulator
        },
        "throughput": {
            "wall_s": wall,
            "episodes_per_s": len(metas) / wall if wall else None,
            "transitions_per_s": n_steps / wall if wall else None,
            "bytes_on_disk": bytes_written,
            "bytes_per_transition": bytes_written / n_steps if n_steps else None,
        },
    }
    manifest = write_manifest(out_dir, config, metas, extra)
    if progress:
        log(f"wrote {len(metas)} episodes / {n_steps} transitions "
            f"({bytes_written / 1e6:.1f} MB) to {out_dir} in {wall:.1f}s")
    return manifest
