import argparse

import tiktoken
import torch

from llms_from_scratch.inference import generate, text_to_token_ids, token_ids_to_text
from llms_from_scratch.models.gpt2 import GPT_CONFIG, GPTModel, get_config, load_weights
from llms_from_scratch.paths import CHECKPOINTS_DIR


def main():
    parser = argparse.ArgumentParser(description="Generate text with GPT-2")
    parser.add_argument(
        "--model",
        type=str,
        default="model.pth",
        help="Model name: 'model.pth' (local), 'gpt2-small-124M', 'gpt2-medium-355M', etc.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Every effort moves you",
        help="Starting prompt for generation",
    )
    parser.add_argument("--max-tokens", type=int, default=50, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (0.0 = greedy)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling (default: 50, None to disable)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = args.model.replace(".pth", "")
    if model_name == "model":
        weights_path = CHECKPOINTS_DIR / "model.pth"
        if not weights_path.exists():
            print(f"Error: {weights_path} not found. Run scripts/pretrain_gpt2.py first.")
            return
        model = GPTModel(GPT_CONFIG)
        model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))
        model.eval()
        model.to(device)
        config = GPT_CONFIG
    else:
        model = load_weights(model_name=model_name, device=str(device))
        config = get_config(model_name)
    print(f"Loaded model: {args.model}")

    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(args.prompt, tokenizer).to(device),
        max_new_tokens=args.max_tokens,
        context_size=config["context_length"],
        temperature=args.temperature,
        top_k=args.top_k,
    )
    output_text = token_ids_to_text(token_ids, tokenizer)
    print(f"\nPrompt: {args.prompt}")
    print(f"\nGenerated text:\n{output_text}")


if __name__ == "__main__":
    main()
