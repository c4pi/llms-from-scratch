# LLMs from Scratch

GPT-2 from scratch in Python + PyTorch, following Sebastian Raschka's [*Build a Large Language Model (From Scratch)*](https://github.com/rasbt/LLMs-from-scratch).

## Quickstart

```bash
uv sync
```

## Pretraining

Train GPT-2 on "The Verdict" (downloads automatically). Checkpoints saved to `data/checkpoints/`.

```bash
uv run pretrain-gpt2
```

## Text Generation

Generate text using your trained checkpoint or OpenAI's pretrained weights (`gpt2-small-124M`, `gpt2-medium-355M`, etc.):

```bash
uv run generate-gpt2 --prompt "Hello, I am" --max-tokens 50
uv run generate-gpt2 --model gpt2-medium-355M --prompt "The meaning of life"
```

## Spam Classifier

Fine-tune GPT-2 for binary classification on the UCI SMS Spam dataset (downloads on first run):

```bash
uv run finetune-classifier
uv run classify-text --text "You have won a iPhone! Click the link to claim your prize."
```

## Chat UI

Interactive Chainlit interface for generation or classification (controlled via `LLMSFS_MODE`):

```bash
uv run chainlit run app.py
```

| Generation | Spam Classification |
| :--- | :--- |
| <img src="assets/generation.png" width="960" alt="Chat UI - generation" /> | <img src="assets/classification.png" width="960" alt="Chat UI - classification" /> |

## KV-Cache Benchmark

Compare generation speed with and without KV caching:

```bash
uv run benchmark-kvcache --model gpt2-small-124M --max-tokens 100
```

## Acknowledgments

Code draws on [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) (Apache 2.0).