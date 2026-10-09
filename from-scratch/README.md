# from-scratch

A small, readable GPT-style language model trained from scratch — a learning project for how LLM training works end to end.

Character-level model trained on Tiny Shakespeare (~1MB of text). Default config is ~10M parameters and trains in a few minutes on a single consumer GPU.

## Setup

```bash
source ../.venv/bin/activate   # see top-level README
pip install -r requirements.txt   # installs CUDA-enabled torch on Linux
```

## Train

```bash
python data/prepare.py   # download + tokenize -> data/train.bin, data/val.bin
python train.py          # writes best checkpoint to out/ckpt.pt
python sample.py --prompt "ROMEO:" --tokens 500
```

Quick CPU-sized run: `python train.py --max_iters 500 --n_layer 4 --n_head 4 --n_embd 128 --block_size 64`

## Files

| File | What it does |
|---|---|
| `data/prepare.py` | Downloads text, builds a char vocab, writes token ids to disk |
| `model.py` | The transformer: embeddings → N × (causal self-attention + MLP) → logits |
| `train.py` | Training loop: random batches, AdamW, warmup + cosine LR, eval + checkpointing |
| `sample.py` | Loads a checkpoint and generates text autoregressively |

## Things to try next

- Change model size (`--n_layer`, `--n_embd`) and watch train vs. val loss — spot overfitting
- Turn `--dropout` to 0 and see what happens to val loss
- Swap the char tokenizer for BPE (e.g. `tiktoken`) and a bigger dataset
- Add gradient accumulation to simulate larger batches
- Fine-tune instead of training from scratch
