"""enwik8 data loading. 100MB Wikipedia XML, 90M/5M/5M split, vocab=256 bytes."""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ENWIK8_PATH = os.path.join(DATA_DIR, "enwik8")


def prepare_splits():
    if not os.path.exists(ENWIK8_PATH):
        print(f"ERROR: {ENWIK8_PATH} not found. Run setup.sh first.")
        sys.exit(1)
    with open(ENWIK8_PATH, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    assert len(raw) == 100_000_000
    for name, sl in [("train", slice(0, 90_000_000)),
                     ("val", slice(90_000_000, 95_000_000)),
                     ("test", slice(95_000_000, None))]:
        path = os.path.join(DATA_DIR, f"enwik8_{name}.npy")
        np.save(path, raw[sl].astype(np.int64))
        print(f"  {name}: {sl.stop or 100_000_000} - {sl.start} bytes -> {path}")
    print("Splits saved.")


def load_split(split: str) -> np.ndarray:
    path = os.path.join(DATA_DIR, f"enwik8_{split}.npy")
    if not os.path.exists(path):
        prepare_splits()
    return np.load(path)


class CharDataset(Dataset):
    def __init__(self, split: str, seq_len: int = 512):
        self.data = load_split(split)
        self.seq_len = seq_len
        self.n_chunks = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1]
        return torch.from_numpy(chunk[:-1].copy()), torch.from_numpy(chunk[1:].copy())


def get_dataloader(split: str, seq_len: int = 512, batch_size: int = 32,
                   num_workers: int = 2) -> DataLoader:
    return DataLoader(
        CharDataset(split, seq_len), batch_size=batch_size,
        shuffle=(split == "train"), num_workers=num_workers,
        pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
    )


if __name__ == "__main__":
    if "--prepare" in sys.argv:
        prepare_splits()
    else:
        dl = get_dataloader("train", seq_len=512, batch_size=4)
        x, y = next(iter(dl))
        print(f"x={x.shape} y={y.shape} dtype={x.dtype} range=[{x.min()},{x.max()}]")
        print(f"Sample: {bytes(x[0,:60].tolist()).decode('utf-8', errors='replace')}")
