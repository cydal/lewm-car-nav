"""Stage 1 training: VectorJEPA on the full vector observation, with wandb.

Deliberately a plain PyTorch loop rather than the official repo's
`train.py` (Lightning + Hydra): that script's `spt.data.random_split`
splits at the sample-window level (see `export_hdf5.py`'s docstring for why
that's wrong for us), and wiring around that was simpler to write directly
than to fight through Hydra config composition for one difference. The loss
itself -- `(pred_emb - tgt_emb).pow(2).mean() + lambda * sigreg(emb)` -- is
copied verbatim from `le-wm/train.py:lejepa_forward`; everything that isn't
that five-line formula (the encoder, predictor, action embedder, SIGReg
module itself) is theirs, imported, not reimplemented.

Data loading goes through `fast_loader.WindowedTensorDataset`, not
`stable_worldmodel`'s `HDF5Dataset`/`DataLoader` -- see that module's
docstring. On a first run with the generic path, GPU utilization sat at
~67% with ~460 MiB of 15 GiB used: the model is tiny and the dataset is
tens of MB, so a one-Python-call-per-sample loader was the bottleneck, not
compute. The vectorized gather removed it.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

# See scripts/stage1_smoke.py: this box's cuDNN 9.24 can't load its
# runtime-compiled conv engine on a T4. The only conv in this model is
# action_encoder's Conv1d(k=1, s=1); losing cuDNN's conv kernels costs
# nothing here.
torch.backends.cudnn.enabled = False

from lewm.paths import add_lewm_official_to_path  # noqa: E402
from stage1_vector.fast_loader import WindowedTensorDataset  # noqa: E402
from stage1_vector.model import build_model  # noqa: E402
from stage1_vector.wandb_env import load_wandb_key  # noqa: E402

add_lewm_official_to_path()
from module import SIGReg  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-h5", required=True)
    p.add_argument("--val-h5", required=True)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--num-preds", type=int, default=1)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--encoder-hidden", type=int, default=256)
    p.add_argument("--encoder-depth", type=int, default=2)
    p.add_argument("--predictor-depth", type=int, default=4)
    p.add_argument("--predictor-heads", type=int, default=8)
    p.add_argument("--predictor-dim-head", type=int, default=32)
    p.add_argument("--predictor-mlp-dim", type=int, default=512)
    p.add_argument("--sigreg-weight", type=float, default=0.09)
    p.add_argument("--sigreg-knots", type=int, default=17)
    p.add_argument("--sigreg-num-proj", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--val-every", type=int, default=1, help="epochs between val passes")
    p.add_argument("--ckpt-every", type=int, default=5, help="epochs between checkpoints")
    p.add_argument("--ckpt-dir", default=os.path.expanduser("~/lewm_runs/stage1_checkpoints"))
    p.add_argument("--run-name", default=None)
    p.add_argument("--wandb-project", default="lewm-car-nav")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--log-every", type=int, default=20, help="steps between wandb step-logs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-ckpt", default=None,
                   help="warm-start model weights from a prior run's checkpoint "
                        "(optimizer state is not saved, so Adam restarts fresh)")
    p.add_argument("--start-epoch", type=int, default=1,
                   help="label for the first epoch of this run, for continuity "
                        "with --init-ckpt's logs/plots (does not affect training)")
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

    # Zero-order-hold baseline (next embedding = current), same embedding
    # space as pred_loss. Not used in the loss -- logged so val/margin_pct
    # is visible during training instead of needing a separate
    # baseline_eval.py pass to find out the combined val/loss (dominated by
    # sigreg, which only ever gets a gradient on train batches and has no
    # reason to track anything on held-out data) was the wrong thing to
    # read the training curve from.
    copy_loss = (ctx_emb - tgt_emb).pow(2).mean()
    return loss, pred_loss, reg_loss, copy_loss


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    run_name = args.run_name or time.strftime("stage1_vector_%Y%m%d_%H%M%S")

    use_wandb = not args.no_wandb
    if use_wandb:
        if not load_wandb_key():
            print("no WANDB_API_KEY found (checked env and le-wm/.env); "
                  "continuing without wandb")
            use_wandb = False
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    num_steps = args.history_size + args.num_preds
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Whole dataset lives on `device` as two tensors (state, action); batches
    # are gathered with one vectorized index op instead of one Python call
    # per sample -- see fast_loader.py. Both train and val arrays here are
    # tens of MB; this would need rethinking well before it stopped fitting.
    train_ds = WindowedTensorDataset(args.train_h5, num_steps, device=device)
    val_ds = WindowedTensorDataset(args.val_h5, num_steps, device=device)
    print(f"train windows: {len(train_ds)}  val windows: {len(val_ds)}  "
          f"({num_steps}-step windows, state_dim={train_ds.state_dim}, "
          f"action_dim={train_ds.action_dim})")

    model = build_model(
        state_dim=train_ds.state_dim, action_dim=train_ds.action_dim,
        embed_dim=args.embed_dim, history_size=args.history_size,
        encoder_hidden=args.encoder_hidden, encoder_depth=args.encoder_depth,
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
            opt.step()

            global_step += 1
            epoch_losses.append(loss.item())
            if use_wandb and global_step % args.log_every == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/pred_loss": pred_loss.item(),
                    "train/sigreg": reg_loss.item(),
                    "epoch": epoch,
                }, step=global_step)

        train_mean = sum(epoch_losses) / len(epoch_losses)
        msg = f"epoch {epoch:3d}  train/loss={train_mean:.4f}  ({time.time() - t0:.1f}s)"

        if epoch % args.val_every == 0 or is_last:
            model.eval()
            val_losses, val_pred, val_reg, val_copy = [], [], [], []
            with torch.no_grad():
                for batch in val_ds.epoch_batches(args.batch_size, shuffle=False, drop_last=False):
                    loss, pred_loss, reg_loss, copy_loss = forward_loss(
                        model, sigreg, batch, args.history_size, args.num_preds,
                        args.sigreg_weight)
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
