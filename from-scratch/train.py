"""Train the tiny GPT on the character-level dataset in data/.

    python data/prepare.py
    python train.py                 # defaults (~10M params, a few minutes on a modern GPU)
    python train.py --max_iters 500 --n_layer 4 --n_embd 128   # quick CPU-friendly run
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "out"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def lr_at(it, args):
    """Linear warmup, then cosine decay to min_lr."""
    if it < args.warmup_iters:
        return args.lr * (it + 1) / args.warmup_iters
    progress = (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
    return args.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (args.lr - args.min_lr)


@torch.no_grad()
def estimate_loss(model, splits, args, device, ctx):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(data, args.block_size, args.batch_size, device)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bf16 autocast on GPU for speed; full precision on CPU
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)

    train_data = np.memmap(DATA / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(DATA / "val.bin", dtype=np.uint16, mode="r")
    vocab_size = len(json.loads((DATA / "meta.json").read_text())["chars"])

    cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)
    print(f"device: {device} | params: {model.num_params() / 1e6:.2f}M")

    # no weight decay on biases / LayerNorm / 1-D params
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    OUT.mkdir(exist_ok=True)
    best_val = float("inf")
    t0 = time.time()
    for it in range(args.max_iters + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(it, args)

        if it % args.eval_interval == 0:
            losses = estimate_loss(model, {"train": train_data, "val": val_data}, args, device, ctx)
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | {time.time() - t0:.1f}s")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"model": model.state_dict(), "config": cfg.__dict__}, OUT / "ckpt.pt")
        if it == args.max_iters:
            break

        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        with ctx:
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

    print(f"best val loss: {best_val:.4f} -> {OUT / 'ckpt.pt'}")


if __name__ == "__main__":
    main()
