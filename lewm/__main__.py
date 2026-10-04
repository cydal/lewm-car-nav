"""
Command line for the LeWM dataset tools.

    python -m lewm pilot ~/lewm_runs/pilot_001        # the whole stage-0 deliverable
    python -m lewm collect out/ --n-train 64 --rgb    # just collect
    python -m lewm validate out/ --replay 5           # just check an existing dataset
    python -m lewm bench                              # just the throughput table
    python -m lewm beacon out.png                     # just the beacon A/B figure
    python -m lewm info out/                          # what is in a dataset

`collect` takes a `--config file.json` for anything beyond the handful of flags
below; the flags are the ones that get changed often enough to be worth a flag,
and everything else lives in the config file so that it ends up recorded in the
manifest rather than lost in a shell history.
"""

import argparse
import json
import os
import sys

from .config import CollectConfig
from .dataset import Dataset


def _overrides(a, base):
    """Only the settings actually given on the command line, applied over `base`.

    Kept separate from the base dict so `pilot` can layer the same flags over
    `pilot_config()` while `collect` layers them over the plain defaults, with
    no flag silently reverting a value it was not asked about.
    """
    out = dict(base)
    for key, val in (("n_train", a.n_train), ("n_val", a.n_val), ("n_test", a.n_test),
                     ("environment_seed", a.seed), ("max_steps", a.max_steps),
                     ("image_size", a.image_size), ("note", a.note),
                     ("observation_mode", a.observation_mode)):
        if val is not None:
            out[key] = val
    if a.rgb:
        out["capture_rgb"] = True
    if a.no_rgb:
        out["capture_rgb"] = False
    if a.no_beacon:
        out["show_goal_beacon"] = False
    if a.compress:
        out["compress"] = True
    if a.env_kwargs:
        out["env_kwargs"] = {**out.get("env_kwargs", {}), **json.loads(a.env_kwargs)}
    return CollectConfig.from_dict(out)


def _add_collect_flags(p):
    p.add_argument("--config", help="JSON CollectConfig to start from")
    p.add_argument("--n-train", type=int, dest="n_train")
    p.add_argument("--n-val", type=int, dest="n_val")
    p.add_argument("--n-test", type=int, dest="n_test")
    p.add_argument("--seed", type=int, help="environment_seed")
    p.add_argument("--max-steps", type=int, dest="max_steps",
                   help="collection-side cap per episode (not the env's limit)")
    p.add_argument("--image-size", type=int, dest="image_size")
    p.add_argument("--rgb", action="store_true", help="capture RGB frames")
    p.add_argument("--no-rgb", action="store_true")
    p.add_argument("--observation-mode", dest="observation_mode",
                   choices=("vector_full", "vector_restricted", "rgb"))
    p.add_argument("--no-beacon", action="store_true",
                   help="hide the goal beacon posts (visual control condition)")
    p.add_argument("--compress", action="store_true")
    p.add_argument("--env-kwargs", dest="env_kwargs",
                   help='JSON forwarded to carnav.make, e.g. \'{"traffic": false}\'')
    p.add_argument("--note", help="free text recorded in the manifest")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m lewm", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("collect", help="collect a dataset")
    p.add_argument("out_dir")
    _add_collect_flags(p)

    p = sub.add_parser("pilot", help="collect + validate + figures + report")
    p.add_argument("out_dir", nargs="?", default="runs/lewm_pilot")
    p.add_argument("--replay", type=int, default=3,
                   help="how many episodes to replay bit-exactly")
    p.add_argument("--no-bench", action="store_true")
    p.add_argument("--no-beacon-figure", action="store_true")
    _add_collect_flags(p)

    p = sub.add_parser("validate", help="check an existing dataset")
    p.add_argument("root")
    p.add_argument("--replay", type=int, default=2)
    p.add_argument("--loose", action="store_true",
                   help="report coverage shortfalls as notes rather than failures")

    p = sub.add_parser("bench", help="throughput per capture configuration")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--max-steps", type=int, dest="max_steps", default=250)

    p = sub.add_parser("beacon", help="write the beacon on/off A/B figure")
    p.add_argument("out_path")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--image-size", type=int, dest="image_size", default=192)

    p = sub.add_parser("info", help="summarise a dataset on disk")
    p.add_argument("root")

    a = ap.parse_args(argv)

    if a.cmd == "collect":
        from .collector import collect
        base = CollectConfig.load(a.config) if a.config else CollectConfig()
        collect(_overrides(a, base.to_dict()), a.out_dir)
        return 0

    if a.cmd == "pilot":
        from .pilot import pilot_config, run_pilot
        base = CollectConfig.load(a.config) if a.config else pilot_config()
        ok, _ = run_pilot(_overrides(a, base.to_dict()), a.out_dir,
                          bench=not a.no_bench,
                          beacon=not a.no_beacon_figure, replay=a.replay)
        return 0 if ok else 1

    if a.cmd == "validate":
        from .validate import validate_dataset
        ok, _ = validate_dataset(a.root, replay=a.replay,
                                 strict_coverage=not a.loose)
        return 0 if ok else 1

    if a.cmd == "bench":
        from .pilot import benchmark
        rows = benchmark(n_episodes=a.episodes, max_steps=a.max_steps)
        print(json.dumps(rows, indent=2))
        return 0

    if a.cmd == "beacon":
        from .pilot import beacon_figure
        info = beacon_figure(a.out_path, seed=a.seed, n_frames=a.frames,
                             image_size=a.image_size)
        if info:
            # Sidecar so the parent process (`run_pilot` shells out to get a
            # bigger render than its own ShowBase allows) can pick the numbers
            # up without parsing stdout.
            with open(os.path.splitext(a.out_path)[0] + ".json", "w") as f:
                json.dump(info, f, indent=2)
        return 0

    if a.cmd == "info":
        ds = Dataset(a.root)
        man = ds.manifest
        print(f"{a.root}: schema v{man['schema_version']}, "
              f"{man['n_episodes']} episodes, {man['n_transitions']} transitions")
        print(f"splits: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(man["splits"].items())))
        print(f"mode: {man['config']['observation_mode']}  "
              f"vector={man['config']['capture_vector']} "
              f"rgb={man['config']['capture_rgb']} "
              f"privileged={man['config']['capture_privileged_state']}")
        print(f"vector_dim: {man['vector_dim']}  blocks: "
              + ", ".join(f"{k}{v}" for k, v in sorted(man["obs_slices"].items())))
        print(f"privileged: {', '.join(man['privileged_names'])}")
        for m in man["episodes"]:
            print(f"  {m['episode_id']:<18} {m['policy']:<16} T={m['length']:>4} "
                  f"{str(m['reason']):<8} seed={m['episode_seed']}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
