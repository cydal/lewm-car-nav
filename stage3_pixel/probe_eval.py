"""A quick linear probe: does the trained embedding represent anything
physically meaningful, or does it just beat a trivial baseline at
predicting itself one step ahead?

Margin-over-copy (`baseline_eval.py`) only ever answers "is something
being learned" -- collapse ruled out, better than guessing nothing. It
was never meant to be the quality bar. The official paper's own
validation is latent probing: "Latent space analysis demonstrates
encoding of physical structure through probing physical quantities."
This is that check, done cheaply: fit a linear (Ridge) probe from frozen
embeddings to privileged ground truth that was recorded but never fed to
the model (brief section 24 -- evaluation only, never training input).

Encoding is per-frame independent -- the ViT has no temporal context of
its own, only the predictor does -- so this doesn't need the windowed
loader at all. Every frame in an episode gets encoded once, directly.

Compared against an identically-shaped but *randomly initialized*
encoder, same principle as every other baseline in this project: a
probe that fits well on an untrained model's embeddings isn't evidence
the model learned anything, just that a 192-dim random projection of a
64x64 image retains enough signal for a *linear* probe to exploit
trivially (which is a real, known phenomenon in representation
learning, not a bug) -- the trained-vs-untrained gap is the number that
means something.
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

add_lewm_official_to_path()
from utils import get_img_preprocessor  # noqa: E402

PRIV_NAMES = ['x', 'y', 'heading', 'speed', 'vx', 'vy', 'accel', 'yaw_rate',
              'steer_angle', 'slip', 'target_idx', 'dist_to_goal', 'goal_x',
              'goal_y', 'goal_bearing', 'dist_to_obstacle', 'dist_to_vehicle',
              'bearing_to_vehicle', 'vehicle_speed', 'dist_to_moving_vehicle',
              'bearing_to_moving_vehicle', 'moving_vehicle_speed',
              'dist_to_signal', 'signal_state', 'signal_steps_remaining',
              'red_light_violations']
ANGULAR = {'heading', 'goal_bearing'}
LINEAR_TARGETS = ['speed', 'yaw_rate', 'dist_to_goal', 'dist_to_obstacle']


def build_targets(privileged):
    """privileged: (N, 26) -> dict of {name: (N,) or (N,2) for angular}."""
    out = {}
    for name in LINEAR_TARGETS:
        out[name] = privileged[:, PRIV_NAMES.index(name)]
    for name in ANGULAR:
        theta = privileged[:, PRIV_NAMES.index(name)]
        out[name] = np.stack([np.sin(theta), np.cos(theta)], axis=1)
    return out


def encode_episodes(model, episode_ids, ds, transform, device, batch_size=256):
    """Returns (embeddings (N,D), privileged (N,26)) across all episodes, frame-independent."""
    embs, privs = [], []
    with torch.no_grad():
        for eid in episode_ids:
            with ds.episode(eid) as ep:
                rgb = np.asarray(ep["rgb"])  # (T+1, H, W, 3) uint8
                priv = np.asarray(ep["privileged"])  # (T+1, 26)
            for i in range(0, len(rgb), batch_size):
                chunk = rgb[i:i + batch_size]
                px = torch.from_numpy(chunk).permute(0, 3, 1, 2).to(device)  # (B,C,H,W)
                px = transform({"pixels": px})["pixels"]
                out = model.encode({"pixels": px.unsqueeze(1)})  # (B,1,C,H,W) -> (B,1,D)
                embs.append(out["emb"].squeeze(1).cpu().numpy())
                privs.append(priv[i:i + batch_size])
    return np.concatenate(embs), np.concatenate(privs)


def run_probe(emb, priv, label):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    targets = build_targets(priv)
    n = emb.shape[0]
    idx_tr, idx_val = train_test_split(np.arange(n), test_size=0.2, random_state=0)

    print(f"--- {label} ---")
    for name, y in targets.items():
        y_tr, y_val = y[idx_tr], y[idx_val]
        reg = Ridge(alpha=1.0).fit(emb[idx_tr], y_tr)
        pred = reg.predict(emb[idx_val])
        r2 = r2_score(y_val, pred)
        print(f"  {name:14s}  R^2 = {r2:+.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True, help="original lewm dataset dir (has 'rgb'+'privileged')")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--n-episodes", type=int, default=30)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--vit-size", default="tiny", choices=["tiny", "small", "base", "large"])
    p.add_argument("--embed-dim", type=int, default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = Dataset(args.dataset_root)
    episode_ids = ds.episode_ids(args.split)[:args.n_episodes]

    model = build_model(
        action_dim=3, image_size=args.image_size, patch_size=args.patch_size,
        vit_size=args.vit_size, embed_dim=args.embed_dim,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    transform = get_img_preprocessor(source="pixels", target="pixels", img_size=args.image_size)

    emb, priv = encode_episodes(model, episode_ids, ds, transform, device)
    print(f"{len(episode_ids)} episodes, {emb.shape[0]} frames, embed_dim={emb.shape[1]}")
    run_probe(emb, priv, f"TRAINED ({os.path.basename(args.ckpt)})")

    # Untrained baseline: same architecture, random init, same frames.
    torch.manual_seed(0)
    untrained = build_model(
        action_dim=3, image_size=args.image_size, patch_size=args.patch_size,
        vit_size=args.vit_size, embed_dim=args.embed_dim,
    ).to(device).eval()
    emb_u, _ = encode_episodes(untrained, episode_ids, ds, transform, device)
    run_probe(emb_u, priv, "UNTRAINED baseline (random init, same architecture)")


if __name__ == "__main__":
    main()
