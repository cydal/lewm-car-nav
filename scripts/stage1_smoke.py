"""Stage 1 smoke test: does the VectorJEPA integration actually train?

Not the Stage 1 experiment -- a cheap end-to-end check that the pieces are
wired correctly, in the same spirit as the Stage 0 pilot: prove the
collection/conversion/model path before spending a real training budget on
it. Runs a few hundred steps on the 8-episode pilot training split and
reports the loss trend. A flat or NaN loss means something upstream is
broken; a loss that's dropping only means gradients are flowing, not that
the model has learned anything -- that is the actual Stage 1 experiment,
still to come.
"""

import sys

sys.path.insert(0, ".")

import torch

# This box's cuDNN 9.24 (CUDA 13.2 driver) fails to load its runtime-compiled
# conv engine on this T4 (CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED) -- a
# cudnn/driver mismatch unrelated to this model. The only conv in this model
# is action_encoder's Conv1d(k=1,s=1), so losing cuDNN's conv kernels and
# falling back to the generic ones costs nothing here.
torch.backends.cudnn.enabled = False

from torch.utils.data import DataLoader

from lewm.paths import add_lewm_official_to_path
from stage1_vector.model import build_model

add_lewm_official_to_path()
from module import SIGReg  # noqa: E402

from stable_worldmodel.data.formats.hdf5 import HDF5Dataset  # noqa: E402


def main():
    history_size = 3
    num_preds = 1
    num_steps = history_size + num_preds

    train_ds = HDF5Dataset(
        path="/home/ubuntu/.stable-wm/datasets/carnav_pilot_train.h5",
        frameskip=1,
        num_steps=num_steps,
        keys_to_load=["state", "action"],
        keys_to_cache=["state", "action"],
    )
    print(f"train windows: {len(train_ds)}  (from 8 episodes, {num_steps}-step windows)")

    loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(
        state_dim=train_ds.get_dim("state"), action_dim=train_ds.get_dim("action"),
        embed_dim=64, history_size=history_size,
        predictor_depth=2, predictor_heads=4, predictor_dim_head=16, predictor_mlp_dim=128,
    ).to(device)
    sigreg = SIGReg(knots=17, num_proj=256).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)

    lambd = 0.09
    losses = []
    model.train()
    step = 0
    max_steps = 300
    while step < max_steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)

            out = model.encode(batch)
            emb = out["emb"]
            act_emb = out["act_emb"]

            ctx_emb = emb[:, :history_size]
            ctx_act = act_emb[:, :history_size]
            tgt_emb = emb[:, num_preds:]
            pred_emb = model.predict(ctx_emb, ctx_act)

            pred_loss = (pred_emb - tgt_emb).pow(2).mean()
            reg_loss = sigreg(emb.transpose(0, 1))
            loss = pred_loss + lambd * reg_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append((pred_loss.item(), reg_loss.item(), loss.item()))
            step += 1
            if step % 50 == 0 or step == 1:
                print(f"step {step:4d}  pred_loss={pred_loss.item():.4f}  "
                      f"sigreg={reg_loss.item():.4f}  total={loss.item():.4f}")
            if step >= max_steps:
                break

    first10 = sum(l[2] for l in losses[:10]) / 10
    last10 = sum(l[2] for l in losses[-10:]) / 10
    print(f"\nmean total loss, first 10 steps: {first10:.4f}")
    print(f"mean total loss, last 10 steps:  {last10:.4f}")
    assert all(torch.isfinite(torch.tensor(l)).all() for l in losses), "non-finite loss encountered"
    print("no NaN/Inf losses: OK" )
    if last10 < first10:
        print(f"loss decreased ({first10:.4f} -> {last10:.4f}): gradients are flowing")
    else:
        print(f"WARNING: loss did not decrease ({first10:.4f} -> {last10:.4f})")


if __name__ == "__main__":
    main()
