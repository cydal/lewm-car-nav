"""Stage 3 training: the official JEPA on pixels, unmodified, with wandb.

Structurally identical to `stage1_vector/train.py` -- same loss formula
(copied verbatim from `le-wm/train.py:lejepa_forward`), same wandb/
checkpoint/margin-over-copy logging, same reasoning for a plain loop
instead of their Lightning+Hydra `train.py` (episode-level splits, not
`random_split`). The only real difference is the data loader
(`fast_loader.WindowedPixelDataset`, CPU-resident pixels -- see that
module's docstring) and that `model.encode` here is their own unmodified
`JEPA.encode`, not a fork.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

# See stage1_vector/scripts/stage1_smoke.py: this box's cuDNN 9.24 can't
# load its runtime-compiled conv engine on a T4. ViT has no conv at all
# (patch embed here is a strided Conv2d via HF's ViT -- that one conv call
# per forward is cheap enough that losing cuDNN's conv kernels is a
# non-issue, same as the kernel-size-1 Conv1d in the vector model).
torch.backends.cudnn.enabled = False

from stage3_pixel.fast_loader import WindowedPixelDataset  # noqa: E402
from stage3_pixel.model import build_model  # noqa: E402
from stage1_vector.wandb_env import load_wandb_key  # noqa: E402

from lewm.paths import add_lewm_official_to_path  # noqa: E402

add_lewm_official_to_path()
from module import SIGReg  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-h5", required=True)
    p.add_argument("--val-h5", required=True)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--num-preds", type=int, default=1)
    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--predictor-depth", type=int, default=6)
    p.add_argument("--predictor-heads", type=int, default=16)
    p.add_argument("--predictor-dim-head", type=int, default=64)
    p.add_argument("--predictor-mlp-dim", type=int, default=2048)
    p.add_argument("--sigreg-weight", type=float, default=0.09)
    p.add_argument("--sigreg-knots", type=int, default=17)
    p.add_argument("--sigreg-num-proj", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup-steps", type=int, default=500,
                   help="linear LR warmup from 0 -- ViT-from-scratch is seed-sensitive "
                        "without it (1/4 seeds collapsed within 100 steps in testing); "
                        "the vector models never needed this")
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--ckpt-every", type=int, default=5)
    p.add_argument("--ckpt-dir", default=os.path.expanduser("~/lewm_runs/stage3_checkpoints"))
    p.add_argument("--run-name", default=None)
    p.add_argument("--wandb-project", default="lewm-car-nav")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-ckpt", default=None)
    p.add_argument("--start-epoch", type=int, default=1)
    return p.parse_args()


def forward_loss(model, sigreg, batch, history_size, num_preds, lambd):
    batch = dict(batch)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    out = model.encode(batch)
    emb, act_emb = out["emb"], out["act_emb"]

    ctx_emb = emb[:, :history_size]
    ctx_act = act_emb[:, :history_size]
    tgt_emb = emb[:, num_preds:]
    pred_emb = model.predict(ctx_emb, ctx_act)

    pred_loss = (pred_emb - tgt_emb).pow(2).mean()
    reg_loss = sigreg(emb.transpose(0, 1))
    loss = pred_loss + lambd * reg_loss
    copy_loss = (ctx_emb - tgt_emb).pow(2).mean()
    return loss, pred_loss, reg_loss, copy_loss


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    run_name = args.run_name or time.strftime("stage3_pixel_%Y%m%d_%H%M%S")

    use_wandb = not args.no_wandb
    if use_wandb:
        if not load_wandb_key():
            print("no WANDB_API_KEY found; continuing without wandb")
            use_wandb = False
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    num_steps = args.history_size + args.num_preds
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = WindowedPixelDataset(args.train_h5, num_steps, args.image_size, device=device)
    val_ds = WindowedPixelDataset(args.val_h5, num_steps, args.image_size, device=device)
    print(f"train windows: {len(train_ds)}  val windows: {len(val_ds)}  "
          f"(image_size={args.image_size}, action_dim={train_ds.action_dim})")

    model = build_model(
        action_dim=train_ds.action_dim, image_size=args.image_size, patch_size=args.patch_size,
        embed_dim=args.embed_dim, history_size=args.history_size,
        predictor_depth=args.predictor_depth, predictor_heads=args.predictor_heads,
        predictor_dim_head=args.predictor_dim_head, predictor_mlp_dim=args.predictor_mlp_dim,
    ).to(device)
    if args.init_ckpt:
        model.load_state_dict(torch.load(args.init_ckpt, map_location=device))
        print(f"warm-started from {args.init_ckpt}")
    sigreg = SIGReg(knots=args.sigreg_knots, num_proj=args.sigreg_num_proj).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  model params={n_params}")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    global_step = 0

    for i in range(1, args.epochs + 1):
        epoch = args.start_epoch + i - 1
        is_last = i == args.epochs
        model.train()
        t0 = time.time()
        epoch_losses = []
        for batch in train_ds.epoch_batches(args.batch_size, shuffle=True, drop_last=True):
            loss, pred_loss, reg_loss, _ = forward_loss(
                model, sigreg, batch, args.history_size, args.num_preds, args.sigreg_weight)
            opt.zero_grad()
            loss.backward()
            if args.warmup_steps > 0:
                for group in opt.param_groups:
                    group["lr"] = args.lr * min(1.0, (global_step + 1) / args.warmup_steps)
            opt.step()

            global_step += 1
            epoch_losses.append(loss.item())
            if use_wandb and global_step % args.log_every == 0:
                wandb.log({
                    "train/loss": loss.item(), "train/pred_loss": pred_loss.item(),
                    "train/sigreg": reg_loss.item(), "epoch": epoch,
                }, step=global_step)

        train_mean = sum(epoch_losses) / len(epoch_losses)
        msg = f"epoch {epoch:3d}  train/loss={train_mean:.4f}  ({time.time() - t0:.1f}s)"

        if epoch % args.val_every == 0 or is_last:
            model.eval()
            val_losses, val_pred, val_reg, val_copy = [], [], [], []
            with torch.no_grad():
                for batch in val_ds.epoch_batches(args.batch_size, shuffle=False, drop_last=False):
                    loss, pred_loss, reg_loss, copy_loss = forward_loss(
                        model, sigreg, batch, args.history_size, args.num_preds, args.sigreg_weight)
                    val_losses.append(loss.item())
                    val_pred.append(pred_loss.item())
                    val_reg.append(reg_loss.item())
                    val_copy.append(copy_loss.item())
            val_mean = sum(val_losses) / len(val_losses)
            val_pred_mean = sum(val_pred) / len(val_pred)
            val_reg_mean = sum(val_reg) / len(val_reg)
            val_copy_mean = sum(val_copy) / len(val_copy)
            val_margin_pct = 100 * (val_pred_mean - val_copy_mean) / val_copy_mean
            msg += (f"  val/loss={val_mean:.4f}  val/pred_loss={val_pred_mean:.4f}"
                    f"  val/margin={val_margin_pct:+.1f}%")
            if use_wandb:
                wandb.log({
                    "val/loss": val_mean, "val/pred_loss": val_pred_mean,
                    "val/sigreg": val_reg_mean, "val/copy_loss": val_copy_mean,
                    "val/margin_pct": val_margin_pct, "epoch": epoch,
                }, step=global_step)

        print(msg)

        if epoch % args.ckpt_every == 0 or is_last:
            ckpt_path = os.path.join(args.ckpt_dir, run_name, f"weights_epoch_{epoch}.pt")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"  saved {ckpt_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
