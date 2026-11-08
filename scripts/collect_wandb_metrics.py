"""Pull per-epoch train/val loss + margin history from wandb for every run
referenced in the erosion/overfitting investigation, and cache it to
notes/figure_data/wandb_metrics.json -- so article figures can be redrawn
without re-hitting the wandb API every time.

Some run names map to more than one wandb run id (crashed/restarted
attempts, or a resume that got a fresh id) -- all are merged into one
per-epoch table, later entries overwriting earlier ones for the same
epoch.
"""
import json
import os

import wandb

RUN_IDS = {
    "plateau_900_seed40": ["8gp96fr7", "7gpf0ba7"],
    "plateau_8100_seed40": ["ziapnzu3"],
    "gpu0_continue_900": ["btdnapj5"],
    "scale2700_seed20": ["jnccuxaf"],
    "scale2700_seed21": ["dbz3ku5e"],
    "scale8100_seed30_5ep": ["rr14vpd6"],
}

OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "notes", "figure_data", "wandb_metrics.json",
)

KEYS = ["train/pred_loss", "val/pred_loss", "val/margin_pct", "train/loss", "val/loss"]


def main():
    api = wandb.Api()
    out = {}
    for run_name, ids in RUN_IDS.items():
        per_epoch = {}
        for run_id in ids:
            run = api.run(f"sijpapi/lewm-car-nav/{run_id}")
            h = run.history(samples=50000, pandas=True)
            h = h.sort_values("_step")
            h["epoch"] = h["epoch"].ffill()
            agg = h.groupby("epoch").agg(
                {k: ("mean" if k == "train/pred_loss" or k == "train/loss" else "last")
                 for k in KEYS if k in h.columns}
            )
            for epoch, row in agg.dropna(how="all").iterrows():
                per_epoch[int(epoch)] = {k: (None if pd_isna(v) else float(v))
                                          for k, v in row.items()}
        out[run_name] = per_epoch
        print(f"{run_name}: {len(per_epoch)} epochs")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"wrote {OUT_PATH}")


def pd_isna(v):
    import math
    try:
        return v is None or (isinstance(v, float) and math.isnan(v))
    except Exception:
        return False


if __name__ == "__main__":
    main()
