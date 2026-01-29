import argparse
import time

import tiktoken
import torch

from llms_from_scratch.inference import generate, text_to_token_ids, token_ids_to_text
from llms_from_scratch.models.gpt2 import GPT_CONFIG, GPTModel, get_config, load_weights
from llms_from_scratch.paths import CHECKPOINTS_DIR


def benchmark_generation(model, encoded_tensor, max_new_tokens, context_size, use_cache):
    model.eval()
    # warmup
    with torch.no_grad():
        _ = generate(
            model=model,
            idx=encoded_tensor.clone(),
            max_new_tokens=min(5, max_new_tokens),
            context_size=context_size,
            use_cache=use_cache,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    token_ids = generate(
        model=model,
        idx=encoded_tensor.clone(),
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        use_cache=use_cache,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed, token_ids


def load_model_and_config(model_name, device):
    model_key = model_name.replace(".pth", "")
    if model_key == "model":
        print("Loading local checkpoint (model.pth)")
        config = GPT_CONFIG.copy()
        model = GPTModel(config)
        model.load_state_dict(
            torch.load(CHECKPOINTS_DIR / "model.pth", weights_only=True, map_location=device)
        )
        model.to(device)
        model.eval()
    else:
        print(f"Loading pretrained model: {model_name}")
        model = load_weights(model_name=model_key, device=str(device))
        config = get_config(model_key)
    return model, config


def print_header(start_context, encoded_tensor, max_new_tokens):
    print(f"\n{'='*60}")
    print(f"Input: {start_context!r}")
    print(f"Input tokens: {encoded_tensor.shape[1]}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"{'='*60}\n")


def print_results(time_cached, output_cached, time_uncached, output_uncached, tokenizer):
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    decoded_cached = token_ids_to_text(output_cached, tokenizer)
    decoded_uncached = token_ids_to_text(output_uncached, tokenizer)
    print("\nWith KV cache:")
    print(f"  Time: {time_cached:.3f} sec")
    print(f"  Tokens/sec: {len(output_cached[0]) / time_cached:.1f}")
    print(f"  Output: {decoded_cached[:100]}...")
    print("\nWithout KV cache:")
    print(f"  Time: {time_uncached:.3f} sec")
    print(f"  Tokens/sec: {len(output_uncached[0]) / time_uncached:.1f}")
    print(f"  Output: {decoded_uncached[:100]}...")
    speedup = time_uncached / time_cached
    print(f"\nSpeedup: {speedup:.2f}x")
    if torch.cuda.is_available():
        max_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Peak GPU memory: {max_mem_gb:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Benchmark KV cache performance")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model: 'model' (local checkpoint) or pretrained name (e.g. 'gpt2-small-124M', 'gpt2-medium-355M').",
    )
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate (default: 100)")
    args = parser.parse_args()

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, config = load_model_and_config(args.model, device)
    tokenizer = tiktoken.get_encoding("gpt2")
    start_context = "Hello, I am"
    encoded_tensor = text_to_token_ids(start_context, tokenizer).to(device)
    print_header(start_context, encoded_tensor, args.max_tokens)

    print("Running with KV cache...")
    if hasattr(model, "reset_kv_cache"):
        model.reset_kv_cache()
    time_cached, output_cached = benchmark_generation(
        model, encoded_tensor, args.max_tokens, config["context_length"], use_cache=True
    )

    print("Running without KV cache...")
    if hasattr(model, "reset_kv_cache"):
        model.reset_kv_cache()
    time_uncached, output_uncached = benchmark_generation(
        model, encoded_tensor, args.max_tokens, config["context_length"], use_cache=False
    )

    print_results(time_cached, output_cached, time_uncached, output_uncached, tokenizer)


if __name__ == "__main__":
    main()
