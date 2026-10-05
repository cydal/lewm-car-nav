"""Multi-step rollout error: does one-step accuracy hold up over a horizon?

Stage 1's training loss only ever asks for 1-step-ahead prediction
(`num_preds=1`). Planning (Stage 5) cares about *compounding* error over a
horizon -- a model can be excellent at predicting one step ahead and still
be useless for control if error grows fast when it has to feed its own
predictions back in as context.

This does not reuse `JEPA.rollout`: that method asserts `"pixels" in info`
and is shaped for CEM-style planning, (B, S, T, ...) with S candidate action
sequences per batch item. A single rollout trace doesn't need the sample
dimension, so this is a flattened version of the same autoregressive loop,
written against `state` -- but every call it makes
(`model.action_encoder`, `model.predict`) is still their code, unmodified.

Baselines, the same two ideas as `baseline_eval.py` extended to a horizon:
  * copy: hold the last known embedding constant for every future step
  * mean: ignore the input, predict the batch's mean embedding
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

torch.backends.cudnn.enabled = False

from stage1_vector.fast_loader import WindowedTensorDataset  # noqa: E402
from stage1_vector.model import build_model  # noqa: E402


def rollout_errors(model, batch, history_size, horizon):
    """Returns (real, copy, mean) each a list of length `horizon` of per-step MSE."""
    out = model.encode(batch)
    emb, actions = out["emb"], batch["action"]  # emb: (B, span, D); actions: (B, span, A)

    ctx_emb = emb[:, :history_size]          # (B, H, D) ground-truth context
    cur_emb = ctx_emb
    cur_actions = actions[:, :history_size]  # raw actions fed to action_encoder each step

    batch_mean = emb.mean(dim=(0, 1))

    real_errs, copy_errs, mean_errs = [], [], []
    for t in range(horizon):
        act_emb = model.action_encoder(cur_actions)
        pred_next = model.predict(cur_emb[:, -history_size:], act_emb[:, -history_size:])[:, -1:]

        tgt = emb[:, history_size + t: history_size + t + 1]  # ground truth at this step

        real_errs.append((pred_next - tgt).pow(2).mean().item())
        copy_errs.append((ctx_emb[:, -1:] - tgt).pow(2).mean().item())
        mean_errs.append((batch_mean.view(1, 1, -1) - tgt).pow(2).mean().item())

        next_action = actions[:, history_size + t: history_size + t + 1]
        cur_actions = torch.cat([cur_actions, next_action], dim=1)
        cur_emb = torch.cat([cur_emb, pred_next], dim=1)  # feed the model its own prediction

    return real_errs, copy_errs, mean_errs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--encoder-hidden", type=int, default=256)
    p.add_argument("--encoder-depth", type=int, default=2)
    p.add_argument("--predictor-depth", type=int, default=4)
    p.add_argument("--predictor-heads", type=int, default=8)
    p.add_argument("--predictor-dim-head", type=int, default=32)
    p.add_argument("--predictor-mlp-dim", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=512)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_steps = args.history_size + args.horizon
    ds = WindowedTensorDataset(args.h5, num_steps, device=device)

    model = build_model(
        state_dim=ds.state_dim, action_dim=ds.action_dim,
        embed_dim=args.embed_dim, history_size=args.history_size,
        encoder_hidden=args.encoder_hidden, encoder_depth=args.encoder_depth,
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
