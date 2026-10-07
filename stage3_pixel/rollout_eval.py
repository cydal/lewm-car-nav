"""Multi-step rollout error for pixels -- same question, methodology, and
caveats as `stage1_vector/rollout_eval.py`: does one-step accuracy hold up
over a horizon, measured by feeding the model its own predictions back in
as context (not `JEPA.rollout`, which assumes CEM's (B, S, T, ...) sample
dimension; this is a flattened single-trace version of the same loop).

Also used to answer a Stage-3-specific question directly: is chaining
`num_preds=1` predictions k times better or worse than training with
`num_preds=k` and jumping straight there? Run this against a num_preds=1
checkpoint with `--horizon k` and compare its margin at step k to that
checkpoint's own `baseline_eval.py --num-preds k` margin.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

torch.backends.cudnn.enabled = False

from stage3_pixel.fast_loader import WindowedPixelDataset  # noqa: E402
from stage3_pixel.model import build_model  # noqa: E402


def rollout_errors(model, batch, history_size, horizon):
    out = model.encode(dict(batch))
    emb, actions = out["emb"], batch["action"]

    ctx_emb = emb[:, :history_size]
    cur_emb = ctx_emb
    cur_actions = actions[:, :history_size]
    batch_mean = emb.mean(dim=(0, 1))

    real_errs, copy_errs, mean_errs = [], [], []
    for t in range(horizon):
        act_emb = model.action_encoder(cur_actions)
        pred_next = model.predict(cur_emb[:, -history_size:], act_emb[:, -history_size:])[:, -1:]
        tgt = emb[:, history_size + t: history_size + t + 1]

        real_errs.append((pred_next - tgt).pow(2).mean().item())
        copy_errs.append((ctx_emb[:, -1:] - tgt).pow(2).mean().item())
        mean_errs.append((batch_mean.view(1, 1, -1) - tgt).pow(2).mean().item())

        next_action = actions[:, history_size + t: history_size + t + 1]
        cur_actions = torch.cat([cur_actions, next_action], dim=1)
        cur_emb = torch.cat([cur_emb, pred_next], dim=1)

    return real_errs, copy_errs, mean_errs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--vit-size", default="tiny", choices=["tiny", "small", "base", "large"])
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--embed-dim", type=int, default=None)
    p.add_argument("--predictor-depth", type=int, default=6)
    p.add_argument("--predictor-heads", type=int, default=16)
    p.add_argument("--predictor-dim-head", type=int, default=64)
    p.add_argument("--predictor-mlp-dim", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_steps = args.history_size + args.horizon
    ds = WindowedPixelDataset(args.h5, num_steps, args.image_size, device=device)

    model = build_model(
        action_dim=ds.action_dim, image_size=args.image_size, patch_size=args.patch_size,
        vit_size=args.vit_size, embed_dim=args.embed_dim, history_size=args.history_size,
        predictor_depth=args.predictor_depth, predictor_heads=args.predictor_heads,
        predictor_dim_head=args.predictor_dim_head, predictor_mlp_dim=args.predictor_mlp_dim,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    sums_real = [0.0] * args.horizon
    sums_copy = [0.0] * args.horizon
    sums_mean = [0.0] * args.horizon
    n = 0
    with torch.no_grad():
        for batch in ds.epoch_batches(args.batch_size, shuffle=False, drop_last=False):
            batch = dict(batch)
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
            real, copy_, mean_ = rollout_errors(model, batch, args.history_size, args.horizon)
            for t in range(args.horizon):
                sums_real[t] += real[t]
                sums_copy[t] += copy_[t]
                sums_mean[t] += mean_[t]
            n += 1

    print(f"windows: {len(ds)}  batches: {n}  horizon: {args.horizon} steps")
    print(f"{'step':>5} {'real':>10} {'copy':>10} {'mean':>10} {'margin%':>10}")
    for t in range(args.horizon):
        r, c, m = sums_real[t] / n, sums_copy[t] / n, sums_mean[t] / n
        print(f"{t + 1:5d} {r:10.5f} {c:10.5f} {m:10.5f} {100 * (r - c) / c:9.1f}%")


if __name__ == "__main__":
    main()
