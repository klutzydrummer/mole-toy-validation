"""enwik8 data loading.

Two modes, selected by the global TOKENIZER setting:
  'char'  — raw bytes, vocab=256 (legacy character-level)
  'bpe'   — sentencepiece BPE, vocab=4096, trained on enwik8 train split

Call `set_tokenizer('bpe')` before using get_dataloader() to switch modes.
The tokenizer model is saved to data/enwik8_spm.model on first use.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ENWIK8_PATH = os.path.join(DATA_DIR, "enwik8")

# Active tokenizer mode — switch with set_tokenizer()
_TOKENIZER = "char"
VOCAB_SIZE = {"char": 256, "bpe": 4096}


def set_tokenizer(mode: str):
    global _TOKENIZER
    assert mode in VOCAB_SIZE, f"Unknown tokenizer '{mode}'. Choose 'char' or 'bpe'."
    _TOKENIZER = mode


def get_vocab_size() -> int:
    return VOCAB_SIZE[_TOKENIZER]


# ── Character splits ──────────────────────────────────────────────────────────

def _prepare_char_splits():
    if not os.path.exists(ENWIK8_PATH):
        print(f"ERROR: {ENWIK8_PATH} not found. Run setup.sh first.")
        sys.exit(1)
    with open(ENWIK8_PATH, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    assert len(raw) == 100_000_000
    for name, sl in [("train", slice(0, 90_000_000)),
                     ("val",   slice(90_000_000, 95_000_000)),
                     ("test",  slice(95_000_000, None))]:
        path = os.path.join(DATA_DIR, f"enwik8_{name}.npy")
        np.save(path, raw[sl].astype(np.int64))
        print(f"  char {name}: {path}")
    print("Character splits saved.")


def _load_char_split(split: str) -> np.ndarray:
    path = os.path.join(DATA_DIR, f"enwik8_{split}.npy")
    if not os.path.exists(path):
        _prepare_char_splits()
    return np.load(path)


# ── BPE splits ────────────────────────────────────────────────────────────────

def _spm_model_path():
    return os.path.join(DATA_DIR, "enwik8_spm.model")


def _prepare_bpe_splits():
    """Train a sentencepiece BPE model on enwik8 train, then tokenise all splits."""
    try:
        import sentencepiece as spm
    except ImportError:
        print("Installing sentencepiece...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sentencepiece", "--quiet"])
        import sentencepiece as spm

    model_path = _spm_model_path()

    # Train tokeniser on the first 10M bytes of train split (fast, representative)
    if not os.path.exists(model_path):
        print("Training sentencepiece BPE tokeniser (vocab=4096)...")
        train_text = os.path.join(DATA_DIR, "_spm_train_sample.txt")
        with open(ENWIK8_PATH, "rb") as f:
            sample = f.read(10_000_000).decode("utf-8", errors="replace")
        with open(train_text, "w", encoding="utf-8") as f:
            f.write(sample)
        spm.SentencePieceTrainer.train(
            input=train_text,
            model_prefix=os.path.join(DATA_DIR, "enwik8_spm"),
            vocab_size=4096,
            model_type="bpe",
            character_coverage=1.0,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        )
        os.remove(train_text)
        print(f"  Tokeniser saved to {model_path}")

    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    # Tokenise all three splits
    with open(ENWIK8_PATH, "rb") as f:
        raw_bytes = f.read()

    splits = [
        ("train", raw_bytes[:90_000_000]),
        ("val",   raw_bytes[90_000_000:95_000_000]),
        ("test",  raw_bytes[95_000_000:]),
    ]
    for name, data in splits:
        path = os.path.join(DATA_DIR, f"enwik8_bpe_{name}.npy")
        if not os.path.exists(path):
            print(f"  Tokenising {name} split...")
            text = data.decode("utf-8", errors="replace")
            ids = sp.encode(text, out_type=int)
            arr = np.array(ids, dtype=np.int32)
            np.save(path, arr)
            print(f"    {len(arr):,} tokens -> {path}")
        else:
            print(f"  {path} already exists, skipping.")
    print("BPE splits saved.")


def _load_bpe_split(split: str) -> np.ndarray:
    path = os.path.join(DATA_DIR, f"enwik8_bpe_{split}.npy")
    if not os.path.exists(path):
        _prepare_bpe_splits()
    return np.load(path)


# ── Unified interface ─────────────────────────────────────────────────────────

def prepare_splits():
    """Prepare both char and BPE splits."""
    _prepare_char_splits()
    _prepare_bpe_splits()


def load_split(split: str) -> np.ndarray:
    if _TOKENIZER == "bpe":
        return _load_bpe_split(split)
    return _load_char_split(split)


class TokenDataset(Dataset):
    def __init__(self, split: str, seq_len: int = 256):
        self.data = load_split(split)
        self.seq_len = seq_len
        self.n_chunks = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1]
        return torch.tensor(chunk[:-1].copy(), dtype=torch.long), \
               torch.tensor(chunk[1:].copy(),  dtype=torch.long)


# Keep old name as alias
CharDataset = TokenDataset


def get_dataloader(split: str, seq_len: int = 256, batch_size: int = 32,
                   num_workers: int = 2) -> DataLoader:
    return DataLoader(
        TokenDataset(split, seq_len), batch_size=batch_size,
        shuffle=(split == "train"), num_workers=num_workers,
        pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
    )


if __name__ == "__main__":
    if "--prepare" in sys.argv:
        prepare_splits()
    elif "--prepare-char" in sys.argv:
        _prepare_char_splits()
    elif "--prepare-bpe" in sys.argv:
        _prepare_bpe_splits()
    else:
        mode = "bpe" if "--bpe" in sys.argv else "char"
        set_tokenizer(mode)
        dl = get_dataloader("train", seq_len=256, batch_size=4)
        x, y = next(iter(dl))
        print(f"mode={mode} x={x.shape} y={y.shape} dtype={x.dtype} range=[{x.min()},{x.max()}]")
