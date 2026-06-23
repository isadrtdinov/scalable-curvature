"""
Download Tiny Shakespeare and tokenize it with the GPT-2 BPE encoder
into the {train,val}.bin format that utils/data_loader.py::get_batch expects
(raw uint16 token IDs, since the GPT-2 vocab is < 2^16).
"""

import os
import urllib.request

import numpy as np
import tiktoken


SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(out_dir, "input.txt")

    if not os.path.exists(input_path):
        print(f"Downloading Tiny Shakespeare to {input_path}")
        urllib.request.urlretrieve(SHAKESPEARE_URL, input_path)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    n = len(text)
    split = int(n * 0.9)
    train_text = text[:split]
    val_text = text[split:]

    enc = tiktoken.get_encoding("gpt2")
    train_ids = np.array(enc.encode_ordinary(train_text), dtype=np.uint16)
    val_ids = np.array(enc.encode_ordinary(val_text), dtype=np.uint16)

    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    print(f"train: {len(train_ids):,} tokens -> {train_path}")
    print(f"val:   {len(val_ids):,} tokens -> {val_path}")


if __name__ == "__main__":
    main()
