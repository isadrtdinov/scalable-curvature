# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import os

# Cache full splits in RAM, keyed by (data_dir, split). First call loads from
# disk via np.fromfile; subsequent calls reuse the in-memory array.
# NOTE: uint16 assumes vocab size < 2^16.
_DATA_CACHE = {}


def _load_split(data_dir, split, load_to_ram=True):
    path = os.path.join(data_dir, f"{split}.bin")
    if not load_to_ram:
        # Recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        return np.memmap(path, dtype=np.uint16, mode='r')
    key = (data_dir, split)
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = np.fromfile(path, dtype=np.uint16)
    return _DATA_CACHE[key]


def get_batch(data_dir, split, context_len, batch_size, device, load_to_ram=True):
    data = _load_split(
        data_dir,
        'train' if split == 'train' else 'val',
        load_to_ram=load_to_ram,
    )

    # select batch_size (valid) starting points
    ix = torch.randint(len(data) - context_len, (batch_size,))
    # extract inputs
    x = torch.stack([torch.from_numpy((data[i:i+context_len]).astype(np.int64)) for i in ix])
    # extract outputs
    y = torch.stack([torch.from_numpy((data[i+1:i+1+context_len]).astype(np.int64)) for i in ix])

    if 'cuda' in device:
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking = True), y.pin_memory().to(device, non_blocking = True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
