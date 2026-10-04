"""
Configuration for LeWM dataset collection.

Why a plain Python class with a JSON round-trip rather than YAML: this repo
already has exactly one configuration convention and it is this one --
`EnvConfig`, `CityConfig` and `CarParams` are plain classes with explicit
keyword defaults, routed from a flat namespace by `carnav.make`, and the only
on-disk config format in the tree is the small JSON file `agents/loader.py`
reads. numpy is the project's *only* hard dependency (see pyproject.toml), so
adding PyYAML to describe a collection run would be a second config system and
a new dependency for no capability. `CollectConfig.to_dict()` /
`from_dict()` give the same thing a YAML file would, and the dict is what lands
in the dataset manifest -- so every dataset carries the exact configuration
that produced it.

Settings that belong to the *simulator* are not duplicated here. They go in
`env_kwargs`, which is forwarded verbatim to `carnav.make(**env_kwargs)`, so
anything `EnvConfig`/`CityConfig`/`CarParams` accepts is available and typos
still raise there rather than being silently dropped. `CollectConfig` only owns
the things collection itself decides.
"""

import copy
import json

import numpy as np


# PandaRenderer's own constructor defaults, restated so a dataset manifest
# records the camera rig it was collected with even when nothing was overridden.
# Kept in sync by `tests/test_lewm.py`, which asserts these are the renderer's
# actual defaults rather than a stale copy.
CAMERA_DEFAULTS = {
    "fov": 60.0,
    "cam_dist": 13.0,
    "cam_height": 6.5,
    "look_ahead": 9.0,
}

# Trajectory mixture (brief section 7). Shares, not counts: `policy_schedule`
# turns them into exact per-split quotas. The headline reason this is not just
# "random actions" is in `lewm/policies.py` -- a uniform random action has
# E[brake] = 0.5 and E[throttle] = 0, so it mostly idles.
POLICY_MIX = {
    "competent": 0.25,
    "noisy_competent": 0.15,
    "aggressive": 0.25,
    "intermediate": 0.10,
    "conservative": 0.20,
    "recovery": 0.05,
}

OBSERVATION_MODES = ("vector_full", "vector_restricted", "rgb")

# Which named blocks of the vector observation the *restricted* view drops.
# Block names are `CarNavEnv.obs_slices` keys, so the mask is computed from the
# layout the env itself declares rather than from hard-coded offsets -- the same
# mistake `baselines/scripted.py` documents having made once already.
#
# Defaults follow the brief's stage 2: remove explicit navigation information
# ("nav") and the explicit dynamics variables ("dynamics": speed, yaw rate,
# steer angle, accel, slip). What is left is the LIDAR scan plus whatever
# traffic/signal/pedestrian blocks are switched on.
RESTRICTED_DROP = ("nav", "dynamics")


class CollectConfig:
    """Everything a collection run decides, with the defaults the pilot used.

    The capture flags are independent of `observation_mode` on purpose. The mode
    says what a *model* is allowed to see (and so which view
    `Dataset.transitions` hands back by default); the capture flags say what is
    written to disk. Recording RGB while training on the vector is the whole
    point of section 17 of the brief -- one set of trajectories, three views --
    so the two must not be the same switch.
    """

    def __init__(
        self,
        # --- what the model is allowed to see (recorded; selects the default view)
        observation_mode="vector_full",
        restricted_drop=RESTRICTED_DROP,
        # --- what gets written to disk
        capture_vector=True,
        capture_rgb=False,
        capture_privileged_state=True,
        capture_traffic_state=True,      # per-timestep nearest-K vehicle states
        capture_signal_state=True,       # per-timestep signal phases
        capture_pedestrian_state=True,   # per-timestep nearest-K pedestrian states
        n_traffic_state=8,               # K for the vehicle block
        n_pedestrian_state=4,            # K for the pedestrian block
        # --- visual observation
        show_goal_beacon=True,           # False hides the green waypoint posts
        image_size=64,
        camera=None,                     # overrides for CAMERA_DEFAULTS
        # --- seeds and episode counts, per split
        environment_seed=0,
        n_train=8,
        n_val=2,
        n_test=2,
        max_steps=None,                  # None = the env's own max_episode_steps
        # --- trajectory mixture
        policy_mix=None,
        # --- simulator settings, forwarded to carnav.make
        env_kwargs=None,
        # --- storage
        compress=False,                  # np.savez_compressed instead of savez
        note="",
    ):
        if observation_mode not in OBSERVATION_MODES:
            raise ValueError(f"observation_mode must be one of {OBSERVATION_MODES}, "
                             f"got {observation_mode!r}")
        self.observation_mode = observation_mode
        self.restricted_drop = tuple(restricted_drop)

        self.capture_vector = bool(capture_vector)
        self.capture_rgb = bool(capture_rgb)
        self.capture_privileged_state = bool(capture_privileged_state)
        self.capture_traffic_state = bool(capture_traffic_state)
        self.capture_signal_state = bool(capture_signal_state)
        self.capture_pedestrian_state = bool(capture_pedestrian_state)
        self.n_traffic_state = int(n_traffic_state)
        self.n_pedestrian_state = int(n_pedestrian_state)

        self.show_goal_beacon = bool(show_goal_beacon)
        self.image_size = int(image_size)
        self.camera = dict(CAMERA_DEFAULTS, **(camera or {}))
        unknown_cam = set(self.camera) - set(CAMERA_DEFAULTS)
        if unknown_cam:
            raise ValueError(f"unknown camera setting(s) {sorted(unknown_cam)}; "
                             f"known: {sorted(CAMERA_DEFAULTS)}")

        self.environment_seed = int(environment_seed)
        self.n_train = int(n_train)
        self.n_val = int(n_val)
        self.n_test = int(n_test)
        self.max_steps = None if max_steps is None else int(max_steps)

        self.policy_mix = dict(policy_mix or POLICY_MIX)
        total = sum(self.policy_mix.values())
        if not np.isclose(total, 1.0):
            raise ValueError(f"policy_mix shares must sum to 1.0, got {total}")

        self.env_kwargs = dict(env_kwargs or {})
        self.compress = bool(compress)
        self.note = str(note)

        # An RGB experiment with no RGB on disk is always a mistake, and it is a
        # mistake that only shows up hours later when the dataset is loaded.
        if self.observation_mode == "rgb" and not self.capture_rgb:
            raise ValueError("observation_mode='rgb' needs capture_rgb=True")
        if self.observation_mode.startswith("vector") and not self.capture_vector:
            raise ValueError(f"observation_mode={self.observation_mode!r} needs "
                             "capture_vector=True")
        if not (self.capture_vector or self.capture_rgb):
            raise ValueError("nothing to capture: set capture_vector and/or capture_rgb")

    # ------------------------------------------------------------------
    @property
    def n_episodes(self):
        return self.n_train + self.n_val + self.n_test

    @property
    def split_counts(self):
        return {"train": self.n_train, "val": self.n_val, "test": self.n_test}

    @property
    def obs_type(self):
        """What `carnav.make` must be asked for to satisfy the capture flags.

        `"both"` whenever RGB is captured, even for a vector-mode run: it makes
        the env itself produce the vector and the frame from one `_observe()`
        call, so the two cannot drift out of alignment by a step. Capturing the
        frame separately via `env.render()` after `step()` would also work today
        but puts the alignment guarantee in this package instead of in the env.
        """
        if self.capture_rgb and self.capture_vector:
            return "both"
        if self.capture_rgb:
            return "image"
        return "vector"

    # ------------------------------------------------------------------
    def to_dict(self):
        return {
            "observation_mode": self.observation_mode,
            "restricted_drop": list(self.restricted_drop),
            "capture_vector": self.capture_vector,
            "capture_rgb": self.capture_rgb,
            "capture_privileged_state": self.capture_privileged_state,
            "capture_traffic_state": self.capture_traffic_state,
            "capture_signal_state": self.capture_signal_state,
            "capture_pedestrian_state": self.capture_pedestrian_state,
            "n_traffic_state": self.n_traffic_state,
            "n_pedestrian_state": self.n_pedestrian_state,
            "show_goal_beacon": self.show_goal_beacon,
            "image_size": self.image_size,
            "camera": dict(self.camera),
            "environment_seed": self.environment_seed,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "max_steps": self.max_steps,
            "policy_mix": dict(self.policy_mix),
            "env_kwargs": copy.deepcopy(self.env_kwargs),
            "compress": self.compress,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    def __repr__(self):
        return f"CollectConfig({json.dumps(self.to_dict(), sort_keys=True)})"
