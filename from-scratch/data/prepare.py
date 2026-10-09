"""Download Tiny Shakespeare and encode it at the character level.

Writes train.bin / val.bin (uint16 token ids) and meta.json (vocab) next to this file.
"""
import json
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
HERE = Path(__file__).parent


def main():
    raw = HERE / "input.txt"
    if not raw.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, raw)
    text = raw.read_text(encoding="utf-8")
    print(f"{len(text):,} characters")

    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    print(f"vocab size: {len(chars)}")

    ids = np.array([stoi[c] for c in text], dtype=np.uint16)
    split = int(0.9 * len(ids))
    ids[:split].tofile(HERE / "train.bin")
    ids[split:].tofile(HERE / "val.bin")
    (HERE / "meta.json").write_text(json.dumps({"chars": chars}))
    print(f"train: {split:,} tokens, val: {len(ids) - split:,} tokens")


if __name__ == "__main__":
    main()
