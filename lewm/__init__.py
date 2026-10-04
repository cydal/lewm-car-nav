"""
LeWM dataset instrumentation: trajectory collection around `CarNavEnv`.

This package lives in its own repository and reads the simulator from a
sibling `Car-Navigation-Env` checkout (see `lewm/paths.py`). The dependency
points one way and always has: we import the env, the env has never heard of
us. Nothing under its `env/`, `render/`, `baselines/`, `agents/` or `replay/`
imports `lewm`, and its default behaviour is untouched -- the same discipline
its own `replay/wrapper.py` and `wrappers.py` follow (compose from the
outside; the env never knows).

    from lewm import CollectConfig, collect

    cfg = CollectConfig(capture_rgb=True, n_train=8, n_val=2, n_test=2)
    summary = collect(cfg, out_dir="runs/pilot")

What it gives you:

* `CollectConfig`     -- one config object for observation mode, capture flags,
                         the goal beacon, seeds, camera and the policy mixture.
* `collect`           -- generate episodes and write them to disk.
* `Dataset`           -- read them back, with train/val/test splits and the
                         three dataset *views* (full vector / restricted vector /
                         RGB) derived from the same recorded trajectories.
* `validate_dataset`  -- structural, temporal, visual, action, dynamics and
                         coverage checks (`python -m lewm validate <dir>`).

The one guarantee the whole package is built around, because everything
downstream is wrong if it slips:

    obs[t] + action[t] -> obs[t+1]

Observation-side records (`obs_vector`, `rgb`, `privileged`, ...) have length
T+1; action-side records (`action`, `reward`, `terminated`, ...) have length T.
See `lewm/dataset.py` for the full schema and `tests/test_lewm.py` for the
deterministic-action test that pins the alignment down.
"""

from .paths import add_carnav_to_path, carnav_root

# Before anything else: every module below imports `carnav`, `env.car` or
# `baselines.scripted` at module scope, so the simulator has to be importable
# by the time the next line runs. Doing it here, once, is what lets `python -m
# lewm`, `import lewm.collector` and `tests/test_lewm.py` all work in a fresh
# clone without each repeating the path setup.
add_carnav_to_path()

from .config import CollectConfig, CAMERA_DEFAULTS, POLICY_MIX
from .collector import LeWMDatasetEnv, collect, collect_episode
from .dataset import (Dataset, Episode, SCHEMA_VERSION, read_episode,
                      write_episode, transitions)
from .pilot import benchmark, pilot_config, run_pilot
from .policies import PROFILES, make_policy, policy_schedule
from .state import PrivilegedRecorder
from .validate import rgb_equivalent, validate_dataset

__all__ = [
    "CollectConfig", "CAMERA_DEFAULTS", "POLICY_MIX",
    "LeWMDatasetEnv", "collect", "collect_episode",
    "Dataset", "Episode", "SCHEMA_VERSION", "read_episode", "write_episode",
    "transitions",
    "PROFILES", "make_policy", "policy_schedule",
    "PrivilegedRecorder",
    "rgb_equivalent", "validate_dataset",
    "benchmark", "pilot_config", "run_pilot",
    "add_carnav_to_path", "carnav_root",
]
