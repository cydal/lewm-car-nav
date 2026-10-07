"""Same trivial-baseline check as stage1_vector/baseline_eval.py, for pixels.

See that module's docstring for why this comparison is necessary at all --
nothing here is pixel-specific except loading `WindowedPixelDataset`
instead of `WindowedTensorDataset` and building the ViT-based model.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

torch.backends.cudnn.enabled = False

from stage3_pixel.fast_loader import WindowedPixelDataset  # noqa: E402
from stage3_pixel.model import build_model  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--vit-size", default="tiny", choices=["tiny", "small", "base", "large"])
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--num-preds", type=int, default=1)
    p.add_argument("--frameskip", type=int, default=1)
    p.add_argument("--window-stride", type=int, default=None)
    p.add_argument("--embed-dim", type=int, default=None)
    p.add_argument("--predictor-depth", type=int, default=6)
    p.add_argument("--predictor-heads", type=int, default=16)
    p.add_argument("--predictor-dim-head", type=int, default=64)
    p.add_argument("--predictor-mlp-dim", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_steps = args.history_size + args.num_preds
    ds = WindowedPixelDataset(args.h5, num_steps, args.image_size, device=device,
                               frameskip=args.frameskip, window_stride=args.window_stride)

    model = build_model(
        action_dim=ds.action_dim, image_size=args.image_size, patch_size=args.patch_size,
        vit_size=args.vit_size, embed_dim=args.embed_dim, history_size=args.history_size,
        predictor_depth=args.predictor_depth, predictor_heads=args.predictor_heads,
        predictor_dim_head=args.predictor_dim_head, predictor_mlp_dim=args.predictor_mlp_dim,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    real, copy_, mean_ = [], [], []
    with torch.no_grad():
        for batch in ds.epoch_batches(args.batch_size, shuffle=False, drop_last=False):
            batch = dict(batch)
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
            out = model.encode(batch)
            emb, act_emb = out["emb"], out["act_emb"]

            ctx_emb = emb[:, :args.history_size]
            ctx_act = act_emb[:, :args.history_size]
            tgt_emb = emb[:, args.num_preds:]

            pred_emb = model.predict(ctx_emb, ctx_act)
            real.append((pred_emb - tgt_emb).pow(2).mean().item())
            copy_.append((ctx_emb - tgt_emb).pow(2).mean().item())
            batch_mean = emb.mean(dim=(0, 1))
            mean_.append((batch_mean.view(1, 1, -1) - tgt_emb).pow(2).mean().item())

    n = len(real)
    print(f"windows: {len(ds)}  batches: {n}")
    print(f"trained predictor   pred_loss = {sum(real) / n:.5f}")
    print(f"copy (zero-order)   pred_loss = {sum(copy_) / n:.5f}")
    print(f"batch-mean baseline pred_loss = {sum(mean_) / n:.5f}")


if __name__ == "__main__":
    main()
