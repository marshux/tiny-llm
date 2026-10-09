"""Generate text from a trained checkpoint.

    python sample.py --prompt "ROMEO:" --tokens 500
"""
import argparse
import json
from pathlib import Path

import torch

from model import GPT, GPTConfig

ROOT = Path(__file__).parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="\n")
    p.add_argument("--tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ROOT / "out" / "ckpt.pt", map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    chars = json.loads((ROOT / "data" / "meta.json").read_text())["chars"]
    stoi = {c: i for i, c in enumerate(chars)}
    unknown = set(args.prompt) - stoi.keys()
    if unknown:
        raise SystemExit(f"prompt has characters not in vocab: {unknown}")

    idx = torch.tensor([[stoi[c] for c in args.prompt]], device=device)
    out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
    print("".join(chars[i] for i in out[0].tolist()))


if __name__ == "__main__":
    main()
