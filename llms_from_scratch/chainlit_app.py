import sys

import chainlit
import tiktoken
import torch

from llms_from_scratch.inference import generate, text_to_token_ids, token_ids_to_text
from llms_from_scratch.models.gpt2 import GPTModel, get_config, load_weights, GPT_CONFIG
from llms_from_scratch.paths import CHECKPOINTS_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "gpt2-small-124M"


def get_model_and_tokenizer():
    tokenizer = tiktoken.get_encoding("gpt2")
    model_key = MODEL_NAME.replace(".pth", "")
    if model_key == "model":
        model_path = CHECKPOINTS_DIR / "model.pth"
        if not model_path.exists():
            print(
                f"Could not find {model_path}. "
                "Run scripts/pretrain_gpt2.py first"
            )
            sys.exit(1)
        config = GPT_CONFIG.copy()
        model = GPTModel(config)
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        model.eval()
        model.to(device)
    else:
        config = get_config(model_key)
        model = load_weights(model_name=model_key, device=str(device))
    print(f"Loaded model: {model_key}")
    return tokenizer, model, config


tokenizer, model, model_config = get_model_and_tokenizer()


@chainlit.on_message
async def main(message: chainlit.Message):
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(message.content, tokenizer).to(device),
        max_new_tokens=50,
        context_size=model_config["context_length"],
        top_k=50,
        temperature=1.0,
    )
    text = token_ids_to_text(token_ids, tokenizer)
    await chainlit.Message(content=text).send()
