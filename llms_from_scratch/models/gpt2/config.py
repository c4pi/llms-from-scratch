GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

TRAIN_SETTINGS = {
    "learning_rate": 5e-4,
    "num_epochs": 20,
    "batch_size": 2,
    "weight_decay": 0.1,
}

PRETRAINED_CONFIGS = {
    "gpt2-small-124M": {"emb_dim": 768, "n_layers": 12, "n_heads": 12, "context_length": 1024, "drop_rate": 0.1},
    "gpt2-medium-355M": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16, "context_length": 1024, "drop_rate": 0.1},
    "gpt2-large-774M": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20, "context_length": 1024, "drop_rate": 0.1},
    "gpt2-xl-1558M": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25, "context_length": 1024, "drop_rate": 0.1},
}


def get_config(model_name: str) -> dict:
    if model_name not in PRETRAINED_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(PRETRAINED_CONFIGS.keys())}")
    return {"vocab_size": 50257, "qkv_bias": True, **PRETRAINED_CONFIGS[model_name]}
