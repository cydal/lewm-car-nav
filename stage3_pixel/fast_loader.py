"""In-memory windowed batching for pixel data -- CPU-resident, not GPU.

`stage1_vector/fast_loader.py` puts the whole dataset on the GPU because a
vector dataset is tens of MB. Pixels are not: 900 train episodes at 64x64x3
uint8 is several GB, too large to safely keep resident on a 16 GiB T4
alongside a ViT's activations and optimizer state. So the big array stays
on the CPU as `uint8`, and only the gathered *batch* (now small) moves to
the GPU and gets normalized there -- same principle as the vector loader
(no per-sample Python calls, no `HDF5Dataset.__getitem__`), different
answer to "where does the big array live".

Normalization reuses the official repo's own preprocessor
(`le-wm/utils.py:get_img_preprocessor`, ImageNet mean/std) rather than
reimplementing it -- it expects channel-first `(N, C, H, W)` uint8 input,
which this module permutes into before calling it (confirmed by testing:
it does *not* do the HWC->CHW permute itself, unlike `HDF5Dataset`, which
does that permute for its own callers).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import torch

from lewm.paths import add_lewm_official_to_path

add_lewm_official_to_path()
from utils import get_img_preprocessor  # noqa: E402


class WindowedPixelDataset:
    """Loads `pixels`/`action` for a whole HDF5 split; serves normalized batches."""

    def __init__(self, h5_path, span, image_size=64, device="cpu"):
        with h5py.File(h5_path, "r") as f:
            pixels = f["pixels"][:]  # (N, H, W, C) uint8, stays on CPU
            action = f["action"][:].astype(np.float32)
            ep_len = f["ep_len"][:]
            ep_offset = f["ep_offset"][:]

        self.pixels = torch.from_numpy(pixels)  # CPU, uint8
        self.action = torch.from_numpy(action).to(device)  # tiny, fine on GPU
        self.span = span
        self.device = device
        self.transform = get_img_preprocessor(source="pixels", target="pixels", img_size=image_size)

        starts = []
        for length, offset in zip(ep_len, ep_offset):
            n_valid = int(length) - span + 1
            if n_valid > 0:
                starts.append(np.arange(offset, offset + n_valid, dtype=np.int64))
        self.window_starts = torch.from_numpy(np.concatenate(starts))  # CPU

        self.action_dim = action.shape[1]

    def __len__(self):
        return self.window_starts.shape[0]

    def _load_pixels(self, idx):
        """CPU gather + permute, then one transfer, then normalize on `device`."""
        batch = self.pixels[idx]  # (B, span, H, W, C) uint8, CPU
        b, t, h, w, c = batch.shape
        batch = batch.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)  # (B*span, C, H, W)
        batch = batch.to(self.device)
        batch = self.transform({"pixels": batch})["pixels"]
        return batch.view(b, t, c, h, w)

    def epoch_batches(self, batch_size, shuffle=True, drop_last=True, generator=None):
        n = len(self)
        order = (torch.randperm(n, generator=generator) if shuffle else torch.arange(n))

        n_batches = n // batch_size if drop_last else -(-n // batch_size)
        offsets = torch.arange(self.span)

        for b in range(n_batches):
            sel = order[b * batch_size: (b + 1) * batch_size]
            starts = self.window_starts[sel]                      # (B,) CPU
            idx = starts.unsqueeze(1) + offsets.unsqueeze(0)       # (B, span) CPU
            pixels = self._load_pixels(idx)
            action = self.action[idx.to(self.action.device)]
            yield {"pixels": pixels, "action": action}
