"""
dataset_weighted.py -- MemmapPackedDataset with a parallel recall-weight stream.

Drop-in replacement for train_fast.py's MemmapPackedDataset. Reads the
dual-stream format written by the new pretokenize.py:
  train.bin    uint16 tokens
  weights.bin  uint8  per-token recall weights (aligned 1:1 with tokens)

__getitem__ returns input_ids, labels, AND loss_weights (the weight aligned to
the LABEL positions, i.e. weights[1:] of the window), so the loss can weight
each predicted token. If weights.bin is absent, weights default to all-ones
(behaves exactly like the original dataset).
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapPackedDataset(Dataset):
    def __init__(self, data_dir, max_seq_len, seed=42):
        meta_path = os.path.join(data_dir, "meta.json")
        bin_path = os.path.join(data_dir, "train.bin")
        w_path = os.path.join(data_dir, "weights.bin")

        with open(meta_path) as f:
            meta = json.load(f)
        self.n_tokens = meta["n_tokens"]
        self.max_seq_len = max_seq_len
        self.seed = seed
        self._epoch_offset = 0

        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        if os.path.exists(w_path):
            self.weights = np.memmap(w_path, dtype=np.uint8, mode="r")
            assert len(self.weights) == len(self.data), \
                "weights.bin length must match train.bin"
        else:
            self.weights = None

        usable = self.n_tokens - (max_seq_len - 1)
        self.n_samples = max(1, (usable - 1) // max_seq_len)

    def set_epoch(self, epoch):
        rng = np.random.RandomState(self.seed + epoch)
        self._epoch_offset = rng.randint(0, self.max_seq_len)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.max_seq_len + self._epoch_offset
        end = start + self.max_seq_len + 1
        chunk = self.data[start:end].astype(np.int64)
        input_ids = torch.from_numpy(chunk[:-1])
        labels = torch.from_numpy(chunk[1:])
        if self.weights is not None:
            w = self.weights[start:end].astype(np.float32)
            loss_weights = torch.from_numpy(w[1:])      # align to label positions
        else:
            loss_weights = torch.ones(self.max_seq_len, dtype=torch.float32)
        return {"input_ids": input_ids, "labels": labels,
                "loss_weights": loss_weights}
