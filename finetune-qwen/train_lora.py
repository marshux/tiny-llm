"""LoRA fine-tune a small pretrained Qwen model on an instruction dataset.

A plain PyTorch loop (no Trainer) so every step is visible:
chat-template formatting -> mask prompt tokens out of the loss -> LoRA adapters -> AdamW.

    python train_lora.py                                   # Qwen3-0.6B on 2k Alpaca examples
    python train_lora.py --model Qwen/Qwen2.5-0.5B-Instruct --max_examples 500
"""
import argparse
import math
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = Path(__file__).parent / "out"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--dataset", default="yahma/alpaca-cleaned")
    p.add_argument("--max_examples", type=int, default=2000)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def to_messages(ex):
    """Alpaca row -> chat messages."""
    user = ex["instruction"] + (f"\n\n{ex['input']}" if ex["input"] else "")
    return [{"role": "user", "content": user}, {"role": "assistant", "content": ex["output"]}]


def tokenize(ex, tok, max_len):
    """Tokenize a conversation; labels are -100 on the prompt so only the answer is learned."""
    msgs = to_messages(ex)
    # enable_thinking is a Qwen3 template flag; other templates ignore unknown kwargs
    prompt = tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
    labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids, "labels": labels}


def collate(batch, pad_id):
    T = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), T), pad_id)
    labels = torch.full((len(batch), T), -100)
    attn = torch.zeros((len(batch), T), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        attn[i, :n] = 1
    return ids, labels, attn


@torch.no_grad()
def evaluate(model, data, args, pad_id, device):
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(data), args.batch_size):
        ids, labels, attn = (t.to(device) for t in collate(data[i : i + args.batch_size], pad_id))
        total += model(input_ids=ids, attention_mask=attn, labels=labels).loss.item()
        n += 1
    model.train()
    return total / max(1, n)


def main():
    args = get_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # needed for grad checkpointing with frozen embeddings

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    ds = load_dataset(args.dataset, split="train").shuffle(seed=args.seed).select(range(args.max_examples))
    data = [tokenize(ex, tok, args.max_len) for ex in ds]
    data = [d for d in data if any(l != -100 for l in d["labels"])]  # drop rows truncated to prompt-only
    n_val = max(1, int(len(data) * args.val_frac))
    val, train = data[:n_val], data[n_val:]
    print(f"train: {len(train)} | val: {len(val)}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    total_steps = math.ceil(len(train) / (args.batch_size * args.grad_accum)) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / args.warmup_steps if s < args.warmup_steps
        else 0.5 * (1 + math.cos(math.pi * (s - args.warmup_steps) / max(1, total_steps - args.warmup_steps))),
    )

    print(f"val loss before: {evaluate(model, val, args, pad_id, device):.4f}")
    model.train()
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        random.shuffle(train)
        batches = [train[i : i + args.batch_size] for i in range(0, len(train), args.batch_size)]
        for bi, batch in enumerate(batches):
            ids, labels, attn = (t.to(device) for t in collate(batch, pad_id))
            loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss / args.grad_accum
            loss.backward()
            if (bi + 1) % args.grad_accum == 0 or bi == len(batches) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    print(f"epoch {epoch} | step {step}/{total_steps} | loss {loss.item() * args.grad_accum:.4f} "
                          f"| lr {sched.get_last_lr()[0]:.2e} | {time.time() - t0:.0f}s")
    print(f"val loss after: {evaluate(model, val, args, pad_id, device):.4f}")

    OUT.mkdir(exist_ok=True)
    model.save_pretrained(OUT / "lora")  # saves only the small adapter weights
    tok.save_pretrained(OUT / "lora")
    print(f"saved adapter -> {OUT / 'lora'}")


if __name__ == "__main__":
    main()
