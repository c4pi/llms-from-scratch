import argparse
import json
import sys

import tiktoken
import torch

from llms_from_scratch.classification import GPTForClassification, classify_text
from llms_from_scratch.models.gpt2 import GPTModel, get_config
from llms_from_scratch.paths import CHECKPOINTS_DIR


def main():
    parser = argparse.ArgumentParser(description="Classify text as spam or not spam")
    parser.add_argument(
        "--model",
        type=str,
        default="spam_classifier.pth",
        help="Classifier model filename in checkpoints dir (default: spam_classifier.pth)",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to classify",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    model_path = CHECKPOINTS_DIR / args.model
    metadata_path = CHECKPOINTS_DIR / args.model.replace(".pth", "_metadata.json")

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Run scripts/finetune_classifier.py first to train the model.")
        sys.exit(1)

    if not metadata_path.exists():
        print(f"Error: Metadata not found at {metadata_path}")
        sys.exit(1)

    with metadata_path.open() as f:
        metadata = json.load(f)

    base_model_name = metadata["base_model"]
    max_length = metadata["max_length"]
    num_classes = metadata.get("num_classes", 2)

    config = get_config(base_model_name)
    base_model = GPTModel(config)
    model = GPTForClassification(base_model, num_classes=num_classes, freeze_base=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    label, confidence = classify_text(
        args.text, model, tokenizer, device, max_length=max_length
    )

    print(f"Text: {args.text}")
    print(f"Classification: {label}")
    print(f"Confidence: {confidence*100:.1f}%")


if __name__ == "__main__":
    main()
