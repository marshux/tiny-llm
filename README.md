# tiny-llm

Learning how LLMs are trained, two ways:

| Folder | Approach | What you learn |
|---|---|---|
| [`from-scratch/`](from-scratch/) | Build and pretrain a ~10M-param GPT on Tiny Shakespeare | Transformer internals, tokenization, the pretraining loop |
| [`finetune-qwen/`](finetune-qwen/) | LoRA fine-tune a pretrained Qwen3-0.6B on instruction data | Chat templates, loss masking, parameter-efficient fine-tuning |

Start with `from-scratch/` to understand what's inside a model, then `finetune-qwen/` to see how real models are adapted in practice.

## Setup

One shared virtualenv for both:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r from-scratch/requirements.txt -r finetune-qwen/requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # should print True on a CUDA GPU
```
