"""Does the restricted-observation model (Stage 2: nav + dynamics dropped
from input) reconstruct what was taken away?

Same methodology as `stage3_pixel/probe_eval.py` -- episode-level split
(not frame-level, which leaks near-duplicate adjacent steps across the
boundary), StandardScaler, trained-vs-untrained gap, nuisance probe for
policy identity. See that module's docstring for the full reasoning.

The interesting targets here are specifically the ones Stage 2 removed
from the input (`restricted_drop = ['nav', 'dynamics']`): speed,
yaw_rate, steer_angle, accel, slip (dynamics) and dist_to_goal,
goal_bearing (nav-related). If the restricted model shows a real gap
over an untrained one here, that's evidence it's inferring the removed
state from action history alone, the thing Stage 2's margin result
(-14% peak vs full-vector's -40%) already suggested it mostly isn't
doing -- this is the direct, quantity-by-quantity version of that
question instead of the aggregate one-step-prediction version.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lewm.dataset import Dataset  # noqa: E402
from stage1_vector.model import build_model  # noqa: E402

PRIV_NAMES = ['x', 'y', 'heading', 'speed', 'vx', 'vy', 'accel', 'yaw_rate',
              'steer_angle', 'slip', 'target_idx', 'dist_to_goal', 'goal_x',
              'goal_y', 'goal_bearing', 'dist_to_obstacle', 'dist_to_vehicle',
              'bearing_to_vehicle', 'vehicle_speed', 'dist_to_moving_vehicle',
              'bearing_to_moving_vehicle', 'moving_vehicle_speed',
              'dist_to_signal', 'signal_state', 'signal_steps_remaining',
              'red_light_violations']
ANGULAR = {'heading', 'goal_bearing'}
# The quantities Stage 2 actually removed from this model's input.
LINEAR_TARGETS = ['speed', 'yaw_rate', 'steer_angle', 'accel', 'slip', 'dist_to_goal']


def build_targets(privileged):
    out = {}
    for name in LINEAR_TARGETS:
        out[name] = privileged[:, [PRIV_NAMES.index(name)]]
    for name in ANGULAR:
        theta = privileged[:, PRIV_NAMES.index(name)]
        out[name] = np.stack([np.sin(theta), np.cos(theta)], axis=1)
    return out


def encode_episodes(model, episode_ids, ds, keep_mask, device, batch_size=512):
    results = []
    with torch.no_grad():
        for eid in episode_ids:
            with ds.episode(eid) as ep:
                obs = np.asarray(ep["obs_vector"], dtype=np.float32)[:, keep_mask]
                priv = np.asarray(ep["privileged"])
                policy = ep.meta.get("policy")
            embs = []
            for i in range(0, len(obs), batch_size):
                chunk = torch.from_numpy(obs[i:i + batch_size]).to(device)
                out = model.encode({"state": chunk.unsqueeze(1)})  # (B,1,D) -> emb
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True, help="original lewm dataset dir (vector, has 'privileged')")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--n-episodes", type=int, default=150)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--encoder-hidden", type=int, default=256)
    p.add_argument("--encoder-depth", type=int, default=2)
    p.add_argument("--predictor-depth", type=int, default=4)
    p.add_argument("--predictor-heads", type=int, default=8)
    p.add_argument("--predictor-dim-head", type=int, default=32)
    p.add_argument("--predictor-mlp-dim", type=int, default=512)
    p.add_argument("--skip-untrained", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = Dataset(args.dataset_root)
    episode_ids = ds.episode_ids(args.split)[:args.n_episodes]
    keep_mask = ds.restricted_mask()  # drops nav + dynamics, matching the checkpoint's training input
    state_dim = int(keep_mask.sum())

    def build(ckpt_path, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        m = build_model(
            state_dim=state_dim, action_dim=3, embed_dim=args.embed_dim,
            encoder_hidden=args.encoder_hidden, encoder_depth=args.encoder_depth,
            predictor_depth=args.predictor_depth, predictor_heads=args.predictor_heads,
            predictor_dim_head=args.predictor_dim_head, predictor_mlp_dim=args.predictor_mlp_dim,
        ).to(device).eval()
        if ckpt_path:
            m.load_state_dict(torch.load(ckpt_path, map_location=device))
        return m

    model = build(args.ckpt)
    results = encode_episodes(model, episode_ids, ds, keep_mask, device)
    train_set, val_set = split_episodes(results)
    print(f"{len(episode_ids)} episodes, {sum(len(r['emb']) for r in results)} frames, "
          f"state_dim={state_dim}, embed_dim={results[0]['emb'].shape[1]}")

    label = f"TRAINED ({os.path.basename(args.ckpt)})"
    run_physical_probe(train_set, val_set, label)
    run_nuisance_probe(train_set, val_set, label)

    if not args.skip_untrained:
        untrained = build(None, seed=0)
        u_results = encode_episodes(untrained, episode_ids, ds, keep_mask, device)
        u_train, u_val = split_episodes(u_results)
        run_physical_probe(u_train, u_val, "UNTRAINED baseline")
        run_nuisance_probe(u_train, u_val, "UNTRAINED baseline")


if __name__ == "__main__":
    main()
