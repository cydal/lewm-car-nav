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
    throw information away about which sub-step did what) -- in the
    spirit of the official paper's recipe ("frame-skip of 5, grouping
    consecutive actions between frames into a single action block"),
    which the user pointed at for two reasons worth having found out
    empirically first rather than taken on faith: our own measurement
    showed consecutive raw frames differ by only ~10% (mean pixel diff
    0.054-0.062 against within-frame std 0.6) -- a weak, close-to-trivial
    training signal -- and the ViT encode is this project's dominant
    compute cost.

    Whether the compute saving materializes depends on `window_stride`
    (see below), which the paper's text doesn't actually specify -- that
    detail is this module's own choice, not a confirmed fact about their
    implementation, and is deliberately exposed as a parameter rather
    than assumed.

    `frameskip=1` (the default) reduces exactly to the original
    behavior -- same `window_starts` formula, same action shape -- so
    every existing caller is unaffected.
    """

    def __init__(self, h5_path, span, image_size=64, device="cpu", frameskip=1, window_stride=None):
        # For very large datasets, keep the HDF5 file open and read pixels on-demand.
        # For smaller ones, load upfront (faster). Threshold: 30GB available RAM, so
        # if pixels > 20GB, use lazy loading.
        self.h5_path = h5_path
        h5_file_temp = h5py.File(h5_path, "r")
        pixels_size_gb = h5_file_temp["pixels"].nbytes / 1e9
        h5_file_temp.close()

        if pixels_size_gb > 20:
            # Lazy loading: keep file open
            self.h5_file = h5py.File(h5_path, "r")
            self.pixels = None  # Will read on-demand
        else:
            # Eager loading: read upfront
            with h5py.File(h5_path, "r") as f:
                self.pixels = torch.from_numpy(f["pixels"][:])
            self.h5_file = None

        with h5py.File(h5_path, "r") as f:
            action = f["action"][:].astype(np.float32)
            ep_len = f["ep_len"][:]
            ep_offset = f["ep_offset"][:]

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
        # `window_stride` (default: frameskip) is a *separate* knob from
        # frameskip, not the same decision: frameskip controls how far
        # apart the frames *within* one window are; window_stride controls
        # how far apart consecutive window *starts* are, i.e. how many
        # distinct windows exist per epoch. window_stride=frameskip (the
        # default) gives ~1/frameskip the windows of window_stride=1 for
        # the same frameskip (447 -> 87 windows on a 450-step episode at
        # frameskip=5) -- the actual compute saving. window_stride=1 keeps
        # every possible starting point (same window count as
        # frameskip=1), trading that saving for denser coverage per epoch.
        # Which one the paper actually does isn't confirmed from the text
        # we have -- this is deliberately a knob, not an assumption baked
        # in, so both are one flag apart to compare.
        stride = window_stride if window_stride is not None else frameskip
        starts = []
        for length, offset in zip(ep_len, ep_offset):
            n_valid = int(length) - span * frameskip + 1
            if n_valid > 0:
                starts.append(np.arange(offset, offset + n_valid, stride, dtype=np.int64))
        self.window_starts = torch.from_numpy(np.concatenate(starts))  # CPU

        self.raw_action_dim = action.shape[1]
        self.action_dim = self.raw_action_dim * frameskip

    def __len__(self):
        return self.window_starts.shape[0]

    def _load_pixels(self, idx):
        """Load pixels either from memory or from HDF5, depending on size."""
        if self.pixels is not None:
            # Eager-loaded: normal tensor indexing
            batch = self.pixels[idx]
        else:
            # Lazy-loaded: read individual frames sequentially (HDF5 fancy indexing is slow/broken)
            idx_np = idx.numpy()  # (B, span)
            b, span = idx_np.shape
            batch_list = []
            for frame_idx in idx_np.reshape(-1):
                batch_list.append(self.h5_file["pixels"][int(frame_idx)])
            batch_flat = np.array(batch_list)
            batch = torch.from_numpy(batch_flat).reshape(b, span, 64, 64, 3)

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
