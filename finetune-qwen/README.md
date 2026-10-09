# finetune-qwen

Take a small pretrained model ([Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)) and teach it new behavior with **LoRA** — instead of updating all ~600M weights, train small low-rank adapter matrices (~1% of params) on top of the frozen model.

Uses a plain PyTorch training loop (no HF `Trainer`) so you can see each step.

## Run

```bash
source ../.venv/bin/activate
pip install -r requirements.txt
python chat.py --base_only "Explain gradient descent to a 10-year-old."   # baseline
python train_lora.py                                                      # ~2k examples, minutes on a 12GB GPU
python chat.py "Explain gradient descent to a 10-year-old."               # base vs. tuned
```

The model (~1.2GB) and dataset download from Hugging Face on first run.

## What's going on in `train_lora.py`

1. **Chat template** — each example is formatted the way Qwen was trained to see conversations.
2. **Loss masking** — prompt tokens get label `-100`, so the model is only trained to produce the *answer*.
3. **LoRA** — adapters on every attention and MLP projection; base weights frozen.
4. **Gradient checkpointing + grad accumulation** — trade compute for memory so it fits a consumer GPU.
5. **Eval before/after** — val loss on held-out examples shows whether it learned anything.

## Things to try next

- Use your own data: write a JSONL of `{"instruction", "input", "output"}` rows and pass `--dataset json --data_files ...` (needs a small tweak to `load_dataset`)
- Raise `--lora_r` or `--max_examples` and compare val loss
- Try `Qwen/Qwen2.5-0.5B-Instruct` or `Qwen/Qwen3-1.7B` via `--model`
- Merge the adapter into the base weights with `model.merge_and_unload()` for deployment
