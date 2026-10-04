"""Is `pred_loss` actually low, or just small because embeddings barely move?

A `pred_loss` of 0.02 means nothing on its own -- it needs a trivial
reference point. Two:

  * **copy**: predict next embedding = current embedding (zero-order hold,
    ignores the action entirely). If the trained predictor doesn't beat
    this, it isn't using the action/history for anything the embedding
    wasn't already going to do on its own.
  * **mean**: predict the batch's mean embedding regardless of input
    (ignores context AND action). If *this* is close to the real pred_loss,
    the embedding space has mostly collapsed to a point and SIGReg isn't
    doing its job; if `copy` beats `mean` by a lot but the trained model
    only edges out `copy`, the dynamics are close to a random walk around
    the current state and the model isn't adding much on top of that.

Both run through the same trained encoder (a fixed checkpoint) so the
comparison is apples-to-apples in the same embedding space the real
pred_loss was measured in.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

torch.backends.cudnn.enabled = False

from stage1_vector.fast_loader import WindowedTensorDataset  # noqa: E402
from stage1_vector.model import build_model  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--num-preds", type=int, default=1)
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
    num_steps = args.history_size + args.num_preds
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

    real, copy_, mean_ = [], [], []
    with torch.no_grad():
        for batch in ds.epoch_batches(args.batch_size, shuffle=False, drop_last=False):
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
            out = model.encode(batch)
            emb, act_emb = out["emb"], out["act_emb"]

            ctx_emb = emb[:, :args.history_size]
            ctx_act = act_emb[:, :args.history_size]
            tgt_emb = emb[:, args.num_preds:]

            pred_emb = model.predict(ctx_emb, ctx_act)
            real.append((pred_emb - tgt_emb).pow(2).mean().item())

            copy_pred = ctx_emb  # zero-order hold: next = current
            copy_.append((copy_pred - tgt_emb).pow(2).mean().item())

            batch_mean = emb.mean(dim=(0, 1), keepdim=True).expand_as(tgt_emb)
            mean_.append((batch_mean - tgt_emb).pow(2).mean().item())

    n = len(real)
    print(f"windows: {len(ds)}  batches: {n}")
    print(f"trained predictor   pred_loss = {sum(real) / n:.5f}")
    print(f"copy (zero-order)   pred_loss = {sum(copy_) / n:.5f}")
    print(f"batch-mean baseline pred_loss = {sum(mean_) / n:.5f}")


if __name__ == "__main__":
    main()
