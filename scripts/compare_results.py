"""Pivot results/probe_results.jsonl into one table per target: epoch x run.

Doesn't recompute anything -- reads whatever probe_over_time.py has already
appended. Run this after probing two or more checkpoint sets to put their
per-epoch r2 gaps side by side (e.g. does heading erode at the same epoch
for 900 and 8100 episodes, or does more data push erosion out?).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from results.store import load  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", default=None,
                   help="probe target to show, e.g. heading, goal_bearing, speed, "
                        "yaw_rate -- default: show all targets found")
    p.add_argument("--runs", nargs="*", default=None,
                   help="restrict to these run_names; default: all")
    args = p.parse_args()

    records = load()
    if not records:
        print("no results yet -- run probe_over_time.py with --run-name first")
        return

    runs = args.runs or sorted({r["run_name"] for r in records})
    targets = [args.target] if args.target else sorted(
        {k for r in records for k in r.get("r2_gap", {})}
    )

    for target in targets:
        print(f"\n=== {target} (r2 gap vs untrained baseline) ===")
        print(f"{'epoch':>6}" + "".join(f"{run:>20}" for run in runs))
        epochs = sorted({r["epoch"] for r in records if r["run_name"] in runs})
        by_key = {(r["run_name"], r["epoch"]): r for r in records}
        for epoch in epochs:
            row = f"{epoch:6d}"
            for run in runs:
                rec = by_key.get((run, epoch))
                val = rec["r2_gap"].get(target) if rec else None
                row += f"{val:+20.3f}" if val is not None else f"{'--':>20}"
            print(row)


if __name__ == "__main__":
    main()
