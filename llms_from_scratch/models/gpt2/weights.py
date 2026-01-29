from pathlib import Path

import requests
import torch

from llms_from_scratch.models.gpt2.config import PRETRAINED_CONFIGS, get_config
from llms_from_scratch.models.gpt2.model import GPTModel
from llms_from_scratch.paths import CHECKPOINTS_DIR, ensure_dirs

HF_BASE_URL = "https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main"


def download_weights(model_name: str, force: bool = False) -> Path:
    ensure_dirs()
    filename = f"{model_name}.pth"
    file_path = CHECKPOINTS_DIR / filename
    if not file_path.exists() or force:
        url = f"{HF_BASE_URL}/{filename}"
        print(f"Downloading {model_name} from {url}...")
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        with file_path.open("wb") as f:
            f.write(response.content)
        print(f"Downloaded to {file_path}")
    return file_path


def load_weights(
    model_name: str | None = None,
    weights_path: str | None = None,
    config: dict | None = None,
    device: str | None = None,
) -> GPTModel:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if weights_path is not None:
        path = Path(weights_path)
    elif model_name is not None:
        path = CHECKPOINTS_DIR / f"{model_name}.pth"
        if not path.exists():
            download_weights(model_name)
    else:
        raise ValueError("Either model_name or weights_path must be provided")
    if config is None:
        if model_name and model_name in PRETRAINED_CONFIGS:
            config = get_config(model_name)
        else:
            raise ValueError("config must be provided for custom weights")
    model = GPTModel(config)
    state_dict = torch.load(path, weights_only=True, map_location=device)
    # Filter out mask buffers because we compute them dynamically for the kv cache implementation
    state_dict = {k: v for k, v in state_dict.items() if not k.endswith(".att.mask")}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    return model
