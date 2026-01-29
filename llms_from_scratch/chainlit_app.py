import json
import os
import sys

import chainlit
import tiktoken
import torch

from llms_from_scratch.classification import GPTForClassification, classify_text
from llms_from_scratch.inference import generate, text_to_token_ids, token_ids_to_text
from llms_from_scratch.models.gpt2 import GPT_CONFIG, GPTModel, get_config, load_weights
from llms_from_scratch.paths import CHECKPOINTS_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODE = os.environ.get("LLMSFS_MODE", "generation")
MODEL_NAME = os.environ.get("LLMSFS_MODEL", "gpt2-small-124M")

def get_generation_model_and_tokenizer():
    tokenizer = tiktoken.get_encoding("gpt2")
    model_key = MODEL_NAME.replace(".pth", "")
    if model_key == "model":
        model_path = CHECKPOINTS_DIR / "model.pth"
        if not model_path.exists():
            print(f"Could not find {model_path}. Run scripts/pretrain_gpt2.py first")
            sys.exit(1)
        config = GPT_CONFIG.copy()
        model = GPTModel(config)
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        model.eval()
        model.to(device)
    else:
        config = get_config(model_key)
        model = load_weights(model_name=model_key, device=str(device))
    print(f"Loaded generation model ({model_key})")
    return tokenizer, model, config


def get_classifier_model_and_tokenizer():
    tokenizer = tiktoken.get_encoding("gpt2")
    model_path = CHECKPOINTS_DIR / "spam_classifier.pth"
    metadata_path = CHECKPOINTS_DIR / "spam_classifier_metadata.json"
    if not model_path.exists():
        print(f"Could not find {model_path}. Run scripts/finetune_classifier.py first")
        sys.exit(1)
    if not metadata_path.exists():
        print(f"Could not find {metadata_path}")
        sys.exit(1)
    with metadata_path.open() as f:
        metadata = json.load(f)
    base_model_name = metadata["base_model"]
    max_length = metadata["max_length"]
    num_classes = metadata.get("num_classes", 2)
    config = get_config(base_model_name)
    base_model = GPTModel(config)
    model = GPTForClassification(base_model, num_classes=num_classes, freeze_base=False)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.eval()
    model.to(device)
    print(f"Loaded classifier model (base: {base_model_name})")
    return tokenizer, model, config, max_length


if MODE == "classification":
    tokenizer, model, model_config, max_length = get_classifier_model_and_tokenizer()
else:
    tokenizer, model, model_config = get_generation_model_and_tokenizer()
    max_length = None


@chainlit.on_message
async def main(message: chainlit.Message):
    if MODE == "classification":
        label, confidence = classify_text(
            message.content, model, tokenizer, device, max_length=max_length
        )
        response = f"**Classification:** {label}\n**Confidence:** {confidence*100:.1f}%"
        await chainlit.Message(content=response).send()
    else:
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
