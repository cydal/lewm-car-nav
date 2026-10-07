"""Linear probes: does the trained embedding represent anything physically
meaningful, or does it just beat a trivial baseline at predicting itself
one step ahead -- and does it needlessly represent things that shouldn't
matter?

Margin-over-copy (`baseline_eval.py`) only ever answers "is something
being learned" -- collapse ruled out, better than guessing nothing. It
was never meant to be the quality bar. The official paper's own
validation is latent probing: "Latent space analysis demonstrates
encoding of physical structure through probing physical quantities."
This is that check, done cheaply: fit a linear (Ridge) probe from frozen
embeddings to privileged ground truth that was recorded but never fed to
the model (brief section 24 -- evaluation only, never training input).

Two kinds of probe here, both against the same split:
  * physical quantities (speed, heading, dist_to_goal, ...) -- high R^2
    is the hoped-for outcome: the representation keeps what's relevant.
  * policy identity (which of the 6 driving profiles generated this
    episode) -- a *nuisance* probe. High accuracy here would be a bad
    sign: the model wasting capacity on something task-irrelevant
    instead of discarding it. There is no reason the dynamics-prediction
    objective should need to know which policy is driving.

Encoding is per-frame independent -- the ViT has no temporal context of
its own, only the predictor does -- so this doesn't need the windowed
loader at all. Every frame in an episode gets encoded once, directly.

Split is by *episode*, not frame -- splitting individual frames at
random would put near-duplicate adjacent frames (pixels change ~10% per
step) on both sides of the probe-train/probe-val boundary, the same
category of leakage the brief rules out for the main dataset split.

Compared against an identically-shaped but *randomly initialized*
encoder, same principle as every other baseline in this project: a
probe that fits well on an untrained model's embeddings isn't evidence
the model learned anything, just that a 192-dim random projection of a
64x64 image retains enough signal for a *linear* probe to exploit
trivially (a real, known phenomenon in representation learning, not a
bug) -- the trained-vs-untrained gap is the number that means something.
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
    """privileged: (N, 26) -> dict of {name: (N, 1) or (N, 2) for angular}."""
    out = {}
    for name in LINEAR_TARGETS:
        out[name] = privileged[:, [PRIV_NAMES.index(name)]]
    for name in ANGULAR:
        theta = privileged[:, PRIV_NAMES.index(name)]
        out[name] = np.stack([np.sin(theta), np.cos(theta)], axis=1)
    return out


def encode_episodes(model, episode_ids, ds, transform, device, batch_size=256):
    """One entry per episode: {'emb': (T,D), 'priv': (T,26), 'policy': str}.

    Kept per-episode (not concatenated) so callers split at the episode
    level, not the frame level.
    """
    results = []
    with torch.no_grad():
        for eid in episode_ids:
            with ds.episode(eid) as ep:
                rgb = np.asarray(ep["rgb"])
                priv = np.asarray(ep["privileged"])
                policy = ep.meta.get("policy")
            embs = []
            for i in range(0, len(rgb), batch_size):
                chunk = rgb[i:i + batch_size]
                px = torch.from_numpy(chunk).permute(0, 3, 1, 2).to(device)
                px = transform({"pixels": px})["pixels"]
                out = model.encode({"pixels": px.unsqueeze(1)})
                embs.append(out["emb"].squeeze(1).cpu().numpy())
            results.append({"emb": np.concatenate(embs), "priv": priv, "policy": policy})
    return results


def split_episodes(results, test_size=0.2, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(results))
    n_val = max(1, int(round(len(results) * test_size)))
    val_idx, train_idx = order[:n_val], order[n_val:]
    return [results[i] for i in train_idx], [results[i] for i in val_idx]


def stack(results):
    emb = np.concatenate([r["emb"] for r in results])
    priv = np.concatenate([r["priv"] for r in results])
    policy = np.concatenate([[r["policy"]] * len(r["emb"]) for r in results])
    return emb, priv, policy


def run_physical_probe(train_set, val_set, label):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    emb_tr, priv_tr, _ = stack(train_set)
    emb_val, priv_val, _ = stack(val_set)

    emb_scaler = StandardScaler().fit(emb_tr)
    emb_tr_s, emb_val_s = emb_scaler.transform(emb_tr), emb_scaler.transform(emb_val)

    targets_tr, targets_val = build_targets(priv_tr), build_targets(priv_val)

    print(f"--- {label}: physical quantities (episode-level split, "
          f"{len(train_set)} train / {len(val_set)} val episodes) ---")
    out = {}
    for name in targets_tr:
        y_tr, y_val = targets_tr[name], targets_val[name]
        y_scaler = StandardScaler().fit(y_tr)
        reg = Ridge(alpha=1.0).fit(emb_tr_s, y_scaler.transform(y_tr))
        pred = y_scaler.inverse_transform(reg.predict(emb_val_s).reshape(len(emb_val_s), -1))
        r2 = r2_score(y_val, pred)
        out[name] = r2
        print(f"  {name:14s}  R^2 = {r2:+.3f}")
    return out


def run_nuisance_probe(train_set, val_set, label):
    """Policy identity: a *good* model should score LOW here."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    emb_tr, _, policy_tr = stack(train_set)
    emb_val, _, policy_val = stack(val_set)

    emb_scaler = StandardScaler().fit(emb_tr)
    emb_tr_s, emb_val_s = emb_scaler.transform(emb_tr), emb_scaler.transform(emb_val)

    clf = LogisticRegression(max_iter=1000).fit(emb_tr_s, policy_tr)
    acc = clf.score(emb_val_s, policy_val)
    classes, counts = np.unique(policy_val, return_counts=True)
    chance = counts.max() / counts.sum()
    print(f"--- {label}: nuisance probe (policy identity, {len(classes)} classes) ---")
    print(f"  accuracy = {acc:.3f}  (majority-class chance = {chance:.3f})")
    return acc, chance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True, help="original lewm dataset dir (has 'rgb'+'privileged')")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--n-episodes", type=int, default=60)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--vit-size", default="tiny", choices=["tiny", "small", "base", "large"])
    p.add_argument("--embed-dim", type=int, default=None)
    p.add_argument("--skip-untrained", action="store_true",
                   help="skip the random-init baseline (useful when sweeping many checkpoints)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = Dataset(args.dataset_root)
    episode_ids = ds.episode_ids(args.split)[:args.n_episodes]
    transform = get_img_preprocessor(source="pixels", target="pixels", img_size=args.image_size)

    def build(ckpt_path, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        m = build_model(
            action_dim=3, image_size=args.image_size, patch_size=args.patch_size,
            vit_size=args.vit_size, embed_dim=args.embed_dim,
        ).to(device).eval()
        if ckpt_path:
            m.load_state_dict(torch.load(ckpt_path, map_location=device))
        return m

    model = build(args.ckpt)
    results = encode_episodes(model, episode_ids, ds, transform, device)
    train_set, val_set = split_episodes(results)
    print(f"{len(episode_ids)} episodes, {sum(len(r['emb']) for r in results)} frames, "
          f"embed_dim={results[0]['emb'].shape[1]}")

    label = f"TRAINED ({os.path.basename(args.ckpt)})"
    run_physical_probe(train_set, val_set, label)
    run_nuisance_probe(train_set, val_set, label)

    if not args.skip_untrained:
        untrained = build(None, seed=0)
        u_results = encode_episodes(untrained, episode_ids, ds, transform, device)
        u_train, u_val = split_episodes(u_results)
        run_physical_probe(u_train, u_val, "UNTRAINED baseline")
        run_nuisance_probe(u_train, u_val, "UNTRAINED baseline")


if __name__ == "__main__":
    main()
