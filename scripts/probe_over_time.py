"""Probe one run's checkpoints across training time: when does each
physical quantity's representation emerge, if it does at all?

Loads the val episodes once (reused across every checkpoint -- only the
model weights change), and the untrained baseline once (same seed every
time, no reason to recompute it per checkpoint). `action_dim` doesn't
matter here and isn't exposed as a flag: `encode_episodes` only ever
calls the frame encoder (`model.encode({"pixels": ...})`), never the
action-conditioned predictor, so this is unaffected by a checkpoint's
training-time frameskip/window_stride -- `vit_size` is the only model
shape knob that matters for probing.
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def r2_per_target(train_set, val_set):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    emb_tr, priv_tr, _ = stack(train_set)
    emb_val, priv_val, _ = stack(val_set)
    emb_scaler = StandardScaler().fit(emb_tr)
    emb_tr_s, emb_val_s = emb_scaler.transform(emb_tr), emb_scaler.transform(emb_val)
    targets_tr, targets_val = build_targets(priv_tr), build_targets(priv_val)

    out = {}
    for name in targets_tr:
        y_tr, y_val = targets_tr[name], targets_val[name]
        y_scaler = StandardScaler().fit(y_tr)
        reg = Ridge(alpha=1.0).fit(emb_tr_s, y_scaler.transform(y_tr))
        pred = y_scaler.inverse_transform(reg.predict(emb_val_s).reshape(len(emb_val_s), -1))
        out[name] = r2_score(y_val, pred)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--vit-size", default="tiny", choices=["tiny", "small", "base", "large"])
    p.add_argument("--action-dim", type=int, default=3,
                   help="must match the checkpoint's raw_action_dim * frameskip used at "
                        "train time -- only affects the unused action_encoder's shape, "
                        "irrelevant to probing itself, but load_state_dict still needs it")
    p.add_argument("--n-episodes", type=int, default=150)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = Dataset(args.dataset_root)
    episode_ids = ds.episode_ids("val")[:args.n_episodes]
    transform = get_img_preprocessor(source="pixels", target="pixels", img_size=64)

    def build():
        return build_model(action_dim=args.action_dim, vit_size=args.vit_size).to(device).eval()

    # Untrained baseline, once.
    torch.manual_seed(0)
    untrained = build()
    u_results = encode_episodes(untrained, episode_ids, ds, transform, device)
    u_train, u_val = split_episodes(u_results)
    untrained_r2 = r2_per_target(u_train, u_val)
    print("untrained baseline:", {k: round(v, 3) for k, v in untrained_r2.items()})
    print()

    ckpts = sorted(
        glob.glob(os.path.join(args.ckpt_dir, "weights_epoch_*.pt")),
        key=lambda p: int(re.search(r"epoch_(\d+)", p).group(1)),
    )

    names = list(untrained_r2.keys())
    print(f"{'epoch':>6}" + "".join(f"{n:>16}" for n in names))
    for ckpt in ckpts:
        epoch = int(re.search(r"epoch_(\d+)", ckpt).group(1))
        model = build()
        model.load_state_dict(torch.load(ckpt, map_location=device))
        results = encode_episodes(model, episode_ids, ds, transform, device)
        train_set, val_set = split_episodes(results)
        r2 = r2_per_target(train_set, val_set)
        gaps = {k: r2[k] - untrained_r2[k] for k in names}
        print(f"{epoch:6d}" + "".join(f"{gaps[n]:+16.3f}" for n in names))


if __name__ == "__main__":
    main()
