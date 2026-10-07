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
    """Loads `pixels`/`action` for a whole HDF5 split; serves normalized batches.

    `frameskip=F>1` samples pixel frames every F raw steps instead of
    every one, and groups the F raw actions between each pair of sampled
    frames into a single wider action block (concatenated, not summed or
    averaged -- concatenation is the only one of the three that doesn't
    throw information away about which sub-step did what) -- matching
    the official paper's own recipe ("frame-skip of 5, grouping
    consecutive actions between frames into a single action block"),
    which exists for two reasons worth having found out empirically
    first rather than taking on faith: our own measurement showed
    consecutive raw frames differ by only ~10% (mean pixel diff 0.054-
    0.062 against within-frame std 0.6) -- a weak, close-to-trivial
    training signal -- and the ViT encode is this project's dominant
    compute cost, so encoding 1/F as many frames for the same real-world
    span is a direct, not-yet-tried answer to the compute problem AMP
    and more GPUs were brought in to work around.

    `frameskip=1` (the default) reduces exactly to the original
    behavior -- same `window_starts` formula, same action shape -- so
    every existing caller is unaffected.
    """

    def __init__(self, h5_path, span, image_size=64, device="cpu", frameskip=1):
        with h5py.File(h5_path, "r") as f:
            pixels = f["pixels"][:]  # (N, H, W, C) uint8, stays on CPU
            action = f["action"][:].astype(np.float32)
            ep_len = f["ep_len"][:]
            ep_offset = f["ep_offset"][:]

        self.pixels = torch.from_numpy(pixels)  # CPU, uint8
        self.action = torch.from_numpy(action).to(device)  # tiny, fine on GPU
        self.span = span
        self.frameskip = frameskip
        self.device = device
        self.transform = get_img_preprocessor(source="pixels", target="pixels", img_size=image_size)

        # A window needs span*frameskip raw action rows (the last block's
        # last row may land on the single NaN-pad row at an episode's end --
        # same as frameskip=1 -- but never past it, since n_valid enforces
        # start + span*frameskip - 1 <= length - 1).
        #
        # Window *starts* also stride by frameskip, not just the frames
        # within one window -- sliding starts by 1 raw step regardless of
        # frameskip would give ~length-span*F+1 windows per episode either
        # way (447 vs 431 for a 450-step episode at F=1 vs F=5), barely
        # fewer, because almost nothing changes about *how many* windows
        # exist, only which frames each one samples. Striding starts by F
        # too is what actually gives the ~F-fold reduction in total
        # training windows (and so encode calls) per epoch.
        starts = []
        for length, offset in zip(ep_len, ep_offset):
            n_valid = int(length) - span * frameskip + 1
            if n_valid > 0:
                starts.append(np.arange(offset, offset + n_valid, frameskip, dtype=np.int64))
        self.window_starts = torch.from_numpy(np.concatenate(starts))  # CPU

        self.raw_action_dim = action.shape[1]
        self.action_dim = self.raw_action_dim * frameskip

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
        pixel_offsets = torch.arange(self.span) * self.frameskip
        action_offsets = torch.arange(self.span * self.frameskip)

        for b in range(n_batches):
            sel = order[b * batch_size: (b + 1) * batch_size]
            starts = self.window_starts[sel]                                    # (B,) CPU
            pixel_idx = starts.unsqueeze(1) + pixel_offsets.unsqueeze(0)        # (B, span)
            pixels = self._load_pixels(pixel_idx)

            action_idx = starts.unsqueeze(1) + action_offsets.unsqueeze(0)      # (B, span*F)
            raw_action = self.action[action_idx.to(self.action.device)]        # (B, span*F, raw_dim)
            action = raw_action.reshape(raw_action.shape[0], self.span, self.action_dim)
            yield {"pixels": pixels, "action": action}
