"""Track embedding *geometry* across training time, independent of any
downstream probe target -- to test whether heading/goal_bearing erosion
(seen in probe_over_time.py once a run gets long enough, e.g.
gpu0_continue_900 by epoch 90) is explained by SIGReg's own isotropy
pressure overwhelming the embedding once pred_loss stops improving.

SIGReg (`le-wm/module.py:SIGReg`) pushes every random 1D projection of
the embedding toward a standard Gaussian (Epps-Pulley normality test).
If that pressure keeps "winning" after the predictor loss plateaus,
embeddings should get *more* isotropic over training time -- a shrinking
gap between the largest and smallest covariance eigenvalues, and a
rising effective rank (participation ratio) as structure spreads evenly
across dimensions rather than concentrating along a few informative
axes (one of which may be heading-aligned).

Reuses encode_episodes()/stack() from stage3_pixel/probe_eval.py --
same val episodes, same transform -- so these numbers line up epoch-for-
epoch with whatever's already in results/probe_results.jsonl.
"""
import argparse
import glob
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.backends.cudnn.enabled = False

from lewm.dataset import Dataset  # noqa: E402
from lewm.paths import add_lewm_official_to_path  # noqa: E402
from stage3_pixel.model import build_model  # noqa: E402
from stage3_pixel.probe_eval import encode_episodes, stack  # noqa: E402
from results.store import append as append_result  # noqa: E402

add_lewm_official_to_path()
from utils import get_img_preprocessor  # noqa: E402
from module import SIGReg  # noqa: E402


def geometry_metrics(emb, sigreg, device):
    """emb: (N, D) raw embeddings, un-normalized -- same tensor
    r2_per_target feeds into Ridge after its own StandardScaler, so
    geometry is computed on the same un-scaled embeddings the model
    actually produced, not on a probe-specific rescaling."""
    emb_t = torch.from_numpy(emb).float().to(device)
    centered = emb_t - emb_t.mean(0, keepdim=True)
    cov = (centered.T @ centered) / (centered.shape[0] - 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0)

    # participation ratio: scale-free "how many dimensions are really
    # used" -- D if eigenvalues are all equal (fully isotropic), 1 if a
    # single direction carries all the variance.
    eff_rank = (eigvals.sum() ** 2 / (eigvals.square().sum() + 1e-12)).item()

    # anisotropy: top eigenvalue vs. median -- shrinks toward 1 as SIGReg
    # pushes every direction to look the same.
    sorted_eigs = eigvals.sort(descending=True).values
    anisotropy = (sorted_eigs[0] / (sorted_eigs[len(sorted_eigs) // 2] + 1e-12)).item()

    norm_mean = emb_t.norm(dim=1).mean().item()
    norm_std = emb_t.norm(dim=1).std().item()

    # SIGReg's own statistic, recomputed directly on these embeddings --
    # lower = closer to isotropic Gaussian under its own random
    # projections. Needs a (T, B, D) shape; T=1 here since we're not
    # tracking a temporal window, just the pooled frame embeddings.
    with torch.no_grad():
        sigreg_stat = sigreg(emb_t.unsqueeze(0)).item()

    return {
        "effective_rank": eff_rank,
        "anisotropy_top_over_median": anisotropy,
        "embed_norm_mean": norm_mean,
        "embed_norm_std": norm_std,
        "sigreg_statistic": sigreg_stat,
        "embed_dim": emb.shape[1],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--vit-size", default="tiny", choices=["tiny", "small", "base", "large"])
    p.add_argument("--action-dim", type=int, default=15)
    p.add_argument("--n-episodes", type=int, default=150)
    p.add_argument("--run-name", default=None)
    p.add_argument("--dataset-scale", type=int, default=None)
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()
    run_name = args.run_name or os.path.basename(os.path.normpath(args.ckpt_dir))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = Dataset(args.dataset_root)
    episode_ids = ds.episode_ids("val")[:args.n_episodes]
    transform = get_img_preprocessor(source="pixels", target="pixels", img_size=64)
    sigreg = SIGReg().to(device)

    def build():
        return build_model(action_dim=args.action_dim, vit_size=args.vit_size).to(device).eval()

    torch.manual_seed(0)
    untrained = build()
    u_results = encode_episodes(untrained, episode_ids, ds, transform, device)
    u_emb, _, _ = stack(u_results)
    untrained_geom = geometry_metrics(u_emb, sigreg, device)
    print("untrained baseline:", {k: round(v, 4) for k, v in untrained_geom.items()})
    print()

    ckpts = sorted(
        glob.glob(os.path.join(args.ckpt_dir, "weights_epoch_*.pt")),
        key=lambda path: int(re.search(r"epoch_(\d+)", path).group(1)),
    )

    names = list(untrained_geom.keys())
    print(f"{'epoch':>6}" + "".join(f"{n:>28}" for n in names))
    for ckpt in ckpts:
        epoch = int(re.search(r"epoch_(\d+)", ckpt).group(1))
        model = build()
        model.load_state_dict(torch.load(ckpt, map_location=device))
        results = encode_episodes(model, episode_ids, ds, transform, device)
        emb, _, _ = stack(results)
        geom = geometry_metrics(emb, sigreg, device)
        print(f"{epoch:6d}" + "".join(f"{geom[n]:28.4f}" for n in names))

        if not args.no_save:
            append_result({
                "run_name": run_name,
                "epoch": epoch,
                "dataset_scale": args.dataset_scale,
                "kind": "geometry",
                "geometry": geom,
                "untrained_geometry": untrained_geom,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })


if __name__ == "__main__":
    main()
