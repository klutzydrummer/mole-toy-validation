"""Data loading for language model toy validation.

Datasets:
  'wikitext103' — WikiText-103-raw, ~517M chars of clean Wikipedia prose (default)
  'enwik8'      — enwik8, 100M bytes of Wikipedia XML (legacy)

Tokenizers:
  'bpe'  — sentencepiece BPE, vocab=4096, trained on each dataset's train split
  'char' — raw bytes, vocab=256 (enwik8 only)

Call set_dataset() and set_tokenizer() before get_dataloader().
Tokenizer models and split arrays are cached in data/ on first use.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ENWIK8_PATH = os.path.join(DATA_DIR, "enwik8")
WIKITEXT103_DIR = os.path.join(DATA_DIR, "wikitext-103-raw")
WIKITEXT103_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"

_DATASET   = "wikitext103"
_TOKENIZER = "bpe"
VOCAB_SIZE  = {"char": 256, "bpe": 4096}


def set_dataset(mode: str):
    global _DATASET
    assert mode in ("wikitext103", "enwik8"), \
        f"Unknown dataset '{mode}'. Choose 'wikitext103' or 'enwik8'."
    _DATASET = mode


def get_dataset() -> str:
    return _DATASET


def set_tokenizer(mode: str):
    global _TOKENIZER
    assert mode in VOCAB_SIZE, f"Unknown tokenizer '{mode}'. Choose 'char' or 'bpe'."
    _TOKENIZER = mode


def get_vocab_size() -> int:
    return VOCAB_SIZE[_TOKENIZER]


# ── Shared SPM helper ─────────────────────────────────────────────────────────

def _ensure_spm():
    try:
        import sentencepiece as spm
        return spm
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sentencepiece", "--quiet"])
        import sentencepiece as spm
        return spm


# ── WikiText-103 ──────────────────────────────────────────────────────────────

def _prepare_wikitext103():
    """Download WikiText-103-raw, train BPE tokenizer, tokenize all splits."""
    import zipfile

    # Download + unzip
    if not os.path.isdir(WIKITEXT103_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        zip_path = os.path.join(DATA_DIR, "wikitext-103-raw-v1.zip")
        print("Downloading WikiText-103-raw (~180MB)...")
        try:
            import urllib.request
            urllib.request.urlretrieve(WIKITEXT103_URL, zip_path)
        except Exception as e:
            print(f"urllib failed ({e}), trying wget...")
            import subprocess
            subprocess.check_call(["wget", "-q", WIKITEXT103_URL, "-O", zip_path])
        print("  Extracting...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR)
        os.remove(zip_path)
        print(f"  Extracted to {WIKITEXT103_DIR}")

    spm = _ensure_spm()

    model_path = os.path.join(DATA_DIR, "wikitext103_spm.model")
    if not os.path.exists(model_path):
        print("Training sentencepiece BPE tokenizer on WikiText-103 (vocab=4096)...")
        train_file = os.path.join(WIKITEXT103_DIR, "wiki.train.raw")
        sample_path = os.path.join(DATA_DIR, "_wt103_spm_sample.txt")
        with open(train_file, "r", encoding="utf-8") as f:
            sample = f.read(20_000_000)
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(sample)
        spm.SentencePieceTrainer.train(
            input=sample_path,
            model_prefix=os.path.join(DATA_DIR, "wikitext103_spm"),
            vocab_size=4096,
            model_type="bpe",
            character_coverage=1.0,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        )
        os.remove(sample_path)
        print(f"  Tokenizer saved to {model_path}")

    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    split_files = [
        ("train", os.path.join(WIKITEXT103_DIR, "wiki.train.raw")),
        ("val",   os.path.join(WIKITEXT103_DIR, "wiki.valid.raw")),
        ("test",  os.path.join(WIKITEXT103_DIR, "wiki.test.raw")),
    ]
    for name, src in split_files:
        out = os.path.join(DATA_DIR, f"wikitext103_bpe_{name}.npy")
        if not os.path.exists(out):
            print(f"  Tokenizing {name} split...")
            with open(src, "r", encoding="utf-8") as f:
                text = f.read()
            ids = sp.encode(text, out_type=int)
            arr = np.array(ids, dtype=np.int32)
            np.save(out, arr)
            print(f"    {len(arr):,} tokens -> {out}")
        else:
            print(f"  {out} already exists, skipping.")
    print("WikiText-103 BPE splits saved.")


def _load_wikitext103_split(split: str) -> np.ndarray:
    path = os.path.join(DATA_DIR, f"wikitext103_bpe_{split}.npy")
    if not os.path.exists(path):
        _prepare_wikitext103()
    return np.load(path)


# ── enwik8 (legacy) ───────────────────────────────────────────────────────────

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


def _prepare_enwik8_bpe_splits():
    """Train a sentencepiece BPE model on enwik8 train, then tokenize all splits."""
    spm = _ensure_spm()
    model_path = os.path.join(DATA_DIR, "enwik8_spm.model")

    if not os.path.exists(model_path):
        print("Training sentencepiece BPE tokenizer on enwik8 (vocab=4096)...")
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
        print(f"  Tokenizer saved to {model_path}")

    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    with open(ENWIK8_PATH, "rb") as f:
        raw_bytes = f.read()
    for name, data in [("train", raw_bytes[:90_000_000]),
                       ("val",   raw_bytes[90_000_000:95_000_000]),
                       ("test",  raw_bytes[95_000_000:])]:
        path = os.path.join(DATA_DIR, f"enwik8_bpe_{name}.npy")
        if not os.path.exists(path):
            print(f"  Tokenizing {name} split...")
            ids = sp.encode(data.decode("utf-8", errors="replace"), out_type=int)
            np.save(path, np.array(ids, dtype=np.int32))
            print(f"    {len(ids):,} tokens -> {path}")
        else:
            print(f"  {path} already exists, skipping.")
    print("enwik8 BPE splits saved.")


def _load_enwik8_bpe_split(split: str) -> np.ndarray:
    path = os.path.join(DATA_DIR, f"enwik8_bpe_{split}.npy")
    if not os.path.exists(path):
        _prepare_enwik8_bpe_splits()
    return np.load(path)


# ── Unified interface ─────────────────────────────────────────────────────────

def load_split(split: str) -> np.ndarray:
    if _DATASET == "wikitext103":
        return _load_wikitext103_split(split)
    # enwik8
    if _TOKENIZER == "bpe":
        return _load_enwik8_bpe_split(split)
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
    if "--prepare-wikitext103" in sys.argv:
        _prepare_wikitext103()
    elif "--prepare-enwik8-bpe" in sys.argv:
        _prepare_enwik8_bpe_splits()
    elif "--prepare-char" in sys.argv:
        _prepare_char_splits()
    else:
        ds = "wikitext103" if "--wikitext103" in sys.argv else \
             ("enwik8" if "--enwik8" in sys.argv else _DATASET)
        set_dataset(ds)
        set_tokenizer("bpe")
        dl = get_dataloader("train", seq_len=256, batch_size=4)
        x, y = next(iter(dl))
        print(f"dataset={ds} x={x.shape} y={y.shape} dtype={x.dtype} range=[{x.min()},{x.max()}]")
