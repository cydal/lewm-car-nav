"""Does heading's declining linear-probe score (seen in probe_over_time.py,
e.g. gpu0_continue_900's heading r2 gap: +0.31 at ep12 -> +0.29 at ep40 ->
+0.135 at ep90) reflect the signal getting *diluted* across more embedding
dimensions, or genuinely *destroyed*?

A full-192-dim Ridge probe (what probe_over_time.py already runs) can't
distinguish these: a linear read-out finds the best linear combination of
all dimensions regardless of how concentrated or spread-out the signal is,
so a drop in the full-probe R2 already means less linearly-recoverable
signal exists in *any* combination -- that alone argues for "destroyed",
not "moved to dimensions a full probe would still find". This script makes
that concrete: restrict the probe to only the top-k PCA directions (by
variance) and see whether heading's R2 recoverable from just those top
directions tracks the full-probe R2 at the same checkpoint, or whether the
gap between "top-k" and "full" widens -- the latter would mean heading's
signal moved to lower-variance directions, i.e. true dilution rather than
destruction.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.backends.cudnn.enabled = False

from lewm.dataset import Dataset  # noqa: E402
from lewm.paths import add_lewm_official_to_path  # noqa: E402
from stage3_pixel.model import build_model  # noqa: E402
from stage3_pixel.probe_eval import (  # noqa: E402
    build_targets, encode_episodes, split_episodes, stack,
)

add_lewm_official_to_path()
from utils import get_img_preprocessor  # noqa: E402


def r2_for_k(emb_tr, emb_val, y_tr, y_val, k):
    """Project onto top-k PCA directions (fit on train), then Ridge from
    those k scores to the target -- k=None means use the full embedding
    (equivalent to what probe_over_time.py already does)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    from sklearn.decomposition import PCA

    scaler = StandardScaler().fit(emb_tr)
    tr_s, val_s = scaler.transform(emb_tr), scaler.transform(emb_val)

    if k is not None:
        pca = PCA(n_components=k).fit(tr_s)
        tr_s, val_s = pca.transform(tr_s), pca.transform(val_s)

    y_scaler = StandardScaler().fit(y_tr)
    reg = Ridge(alpha=1.0).fit(tr_s, y_scaler.transform(y_tr))
    pred = y_scaler.inverse_transform(reg.predict(val_s).reshape(len(val_s), -1))
    return r2_score(y_val, pred)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--epochs", type=int, nargs="+", required=True,
                   help="which checkpoint epochs to compare, e.g. 12 40 60 90")
    p.add_argument("--vit-size", default="tiny", choices=["tiny", "small", "base", "large"])
    p.add_argument("--action-dim", type=int, default=15)
    p.add_argument("--n-episodes", type=int, default=150)
    p.add_argument("--targets", nargs="+", default=["heading", "goal_bearing"])
    p.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20, 50])
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = Dataset(args.dataset_root)
    episode_ids = ds.episode_ids("val")[:args.n_episodes]
    transform = get_img_preprocessor(source="pixels", target="pixels", img_size=64)

    def build():
        return build_model(action_dim=args.action_dim, vit_size=args.vit_size).to(device).eval()

    ks = args.ks + [None]  # None = full embedding, for reference
    print(f"{'epoch':>6}{'target':>14}" + "".join(f"{'k='+str(k) if k else 'full':>10}" for k in ks))
    for epoch in args.epochs:
        ckpt = os.path.join(args.ckpt_dir, f"weights_epoch_{epoch}.pt")
        model = build()
        model.load_state_dict(torch.load(ckpt, map_location=device))
        results = encode_episodes(model, episode_ids, ds, transform, device)
        train_set, val_set = split_episodes(results)
        emb_tr, priv_tr, _ = stack(train_set)
        emb_val, priv_val, _ = stack(val_set)
        targets_tr, targets_val = build_targets(priv_tr), build_targets(priv_val)

        for name in args.targets:
            row = f"{epoch:6d}{name:>14}"
            for k in ks:
                r2 = r2_for_k(emb_tr, emb_val, targets_tr[name], targets_val[name], k)
                row += f"{r2:10.3f}"
            print(row)


if __name__ == "__main__":
    main()
