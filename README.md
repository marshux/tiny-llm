# tiny-llm

Learning how LLMs are trained, two ways:

| Folder | Approach | What you learn |
|---|---|---|
| [`from-scratch/`](from-scratch/) | Build and pretrain a ~10M-param GPT on Tiny Shakespeare | Transformer internals, tokenization, the pretraining loop |
| [`finetune-qwen/`](finetune-qwen/) | LoRA fine-tune a pretrained Qwen3-0.6B on instruction data | Chat templates, loss masking, parameter-efficient fine-tuning |

Start with `from-scratch/` to understand what's inside a model, then `finetune-qwen/` to see how real models are adapted in practice.

## Results and findings

### From scratch

One full run of the default config (10.75M parameters, 5,000 steps) on a single RTX 4070 Ti took **4 minutes 5 seconds**.

| Step | Train loss | Validation loss |
|---|---|---|
| 0 | 4.19 | 4.19 |
| 500 | 1.53 | 1.73 |
| 1,000 | 1.26 | 1.52 |
| **1,500** | 1.15 | **1.47** (best) |
| 2,500 | 0.96 | 1.51 |
| 5,000 | 0.63 | 1.69 |

Lower is better. Train loss is on text the model learns from; validation loss is on text it never sees.

- **It learns the shape of Shakespeare in about a minute.** Loss drops from 4.19 (random guessing over 65 characters) to under 1.5, and the output goes from noise to play-formatted text with speaker names, line breaks and mostly real words.
- **Then it overfits.** Validation loss bottoms out at step 1,500 and climbs for the remaining 3,500 steps while train loss keeps falling: the model starts memorising its 1MB of text instead of generalising. Most of the run past that point is wasted, which is why `train.py` saves the checkpoint with the best validation loss rather than the last one.
- **It gets the form, not the meaning.** A sample from the best checkpoint, prompted with `ROMEO:`:

  ```
  ROMEO:
  So that we love thee all king and old these womb
  For though thy enemies speaks, which mayst thou draw out of good
  Thy gracious suit with bright father,
  Whereto his warlike a king, if the traitor

  HENRY BOLINGBROKE:
  But he is for the consul that stands fall upon
  ```

  Spelling, rhythm and layout are convincing; the sentences mean nothing. That is what a model this small, trained on this little, can do.

### Fine-tuning Qwen

Not run yet, so there are no results to report. The script compares validation loss before and after training, and `chat.py` shows the base and tuned models answering the same prompt side by side.

## Setup

One shared virtualenv for both:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r from-scratch/requirements.txt -r finetune-qwen/requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # should print True on a CUDA GPU
```
