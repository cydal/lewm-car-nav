"""In-memory windowed batching, bypassing `HDF5Dataset.__getitem__`.

`HDF5Dataset.__getitem__` (even with the arrays `keys_to_cache`d in RAM) does
one Python-level call per *sample*: list-index into `clip_indices`, two numpy
slices, two `torch.from_numpy` conversions, then `DataLoader`'s default
collate stacks whatever came back. For video/pixel datasets that per-sample
cost is noise next to the IO it replaces; for us it IS the cost -- the whole
training array is a few tens of MB, every batch is small, and the model's
forward/backward is a handful of microseconds on a GPU, so num_workers=0
left the GPU waiting on a Python loop (~67% utilization, ~460 MiB of 15 GiB
used on a smoke run). The fix is to not go through the per-sample path at
all: load the whole column once, move it to the GPU once, and gather an
entire batch of windows with one vectorized index op.

This intentionally does not use `stable_worldmodel`'s `Dataset` base class.
That class is designed for data too large to fit in memory and formats
(video, lance, lerobot) where per-sample decoding is unavoidable; neither
constraint applies to a few-tens-of-MB vector dataset, so matching its
generality here would be paying its cost for none of its benefit.
"""

import h5py
import numpy as np
import torch


class WindowedTensorDataset:
    """Loads `state`/`action` for a whole HDF5 split, serves batches on `device`.

    `action` keeps its NaN-padded last row per episode (see
    `export_hdf5.py`); windows that would include it get `nan_to_num`'d by
    the caller, same as the reference `train.py`/`scripts/stage1_smoke.py`.
    """

    def __init__(self, h5_path, span, device="cpu"):
        with h5py.File(h5_path, "r") as f:
            state = f["state"][:].astype(np.float32)
            action = f["action"][:].astype(np.float32)
            ep_len = f["ep_len"][:]
            ep_offset = f["ep_offset"][:]

        self.state = torch.from_numpy(state).to(device)
        self.action = torch.from_numpy(action).to(device)
        self.span = span
        self.device = device

        starts = []
        for length, offset in zip(ep_len, ep_offset):
            n_valid = int(length) - span + 1
            if n_valid > 0:
                starts.append(np.arange(offset, offset + n_valid, dtype=np.int64))
        self.window_starts = torch.from_numpy(np.concatenate(starts)).to(device)

        self.state_dim = state.shape[1]
        self.action_dim = action.shape[1]

    def __len__(self):
        return self.window_starts.shape[0]

    def epoch_batches(self, batch_size, shuffle=True, drop_last=True, generator=None):
        """Yield `{'state': (B, span, D), 'action': (B, span, A)}` batches.

        One pass over `window_starts`, matching `DataLoader(shuffle=...,
        drop_last=...)` semantics, but every batch is one gather instead of
        `batch_size` Python calls.
        """
        n = len(self)
        order = (torch.randperm(n, generator=generator, device=self.window_starts.device)
                 if shuffle else torch.arange(n, device=self.window_starts.device))

        n_batches = n // batch_size if drop_last else -(-n // batch_size)
        offsets = torch.arange(self.span, device=self.window_starts.device)

        for b in range(n_batches):
            sel = order[b * batch_size: (b + 1) * batch_size]
            starts = self.window_starts[sel]                      # (B,)
            idx = starts.unsqueeze(1) + offsets.unsqueeze(0)       # (B, span)
            yield {"state": self.state[idx], "action": self.action[idx]}
