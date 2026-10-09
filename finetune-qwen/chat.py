"""Compare the base model with the LoRA-tuned one on the same prompt.

    python chat.py "Give me three tips for learning to cook."
    python chat.py --base_only "..."     # skip the adapter
"""
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ADAPTER = Path(__file__).parent / "out" / "lora"


def generate(model, tok, prompt, max_new_tokens):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--base_only", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    print("=== base ===")
    print(generate(model, tok, args.prompt, args.max_new_tokens))
    if args.base_only:
        return
    if not ADAPTER.exists():
        raise SystemExit(f"no adapter at {ADAPTER}; run train_lora.py first")
    model = PeftModel.from_pretrained(model, ADAPTER).eval()
    print("\n=== base + LoRA ===")
    print(generate(model, tok, args.prompt, args.max_new_tokens))


if __name__ == "__main__":
    main()
