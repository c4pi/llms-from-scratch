import matplotlib.pyplot as plt
import tiktoken
import torch

from llms_from_scratch.data import create_dataloader, load_verdict_text
from llms_from_scratch.models.gpt2 import GPTModel, GPT_CONFIG, TRAIN_SETTINGS
from llms_from_scratch.paths import CHECKPOINTS_DIR, PLOTS_DIR, ensure_dirs
from llms_from_scratch.training import train_model

def main():
    ensure_dirs()
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Model Configuration:")
    for key, value in GPT_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nTraining Settings:")
    for key, value in TRAIN_SETTINGS.items():
        print(f"  {key}: {value}")
    print(f"  device: {device}")

    text_data = load_verdict_text()
    tokenizer = tiktoken.get_encoding("gpt2")
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_loader = create_dataloader(
        text_data[:split_idx],
        tokenizer=tokenizer,
        batch_size=TRAIN_SETTINGS["batch_size"],
        max_length=GPT_CONFIG["context_length"],
        stride=GPT_CONFIG["context_length"],
        drop_last=True,
        shuffle=True,
    )
    val_loader = create_dataloader(
        text_data[split_idx:],
        tokenizer=tokenizer,
        batch_size=TRAIN_SETTINGS["batch_size"],
        max_length=GPT_CONFIG["context_length"],
        stride=GPT_CONFIG["context_length"],
        drop_last=False,
        shuffle=False,
    )

    print(f"\nData:")
    print(f"  Characters: {len(text_data):,}")
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = GPTModel(GPT_CONFIG)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    out_head_params = sum(p.numel() for p in model.out_head.parameters())
    tok_emb_params = sum(p.numel() for p in model.tok_emb.parameters())
    pos_emb_params = sum(p.numel() for p in model.pos_emb.parameters())
    transformer_params = sum(p.numel() for p in model.trf_blocks.parameters())
    norm_params = sum(p.numel() for p in model.final_norm.parameters())
    param_size_mb = total_params * 4 / (1024 ** 2)

    print(f"\nParameter Counts:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {total_params - out_head_params:,}")
    print(f"  Model size (fp32): {param_size_mb:.1f} MB")
    
    print(f"\nParameter Breakdown:")
    print(f"  Token embeddings: {tok_emb_params:,}")
    print(f"  Position embeddings: {pos_emb_params:,}")
    print(f"  Transformer blocks ({GPT_CONFIG['n_layers']}x): {transformer_params:,}")
    print(f"  Final layer norm: {norm_params:,}")
    print(f"  Output head: {out_head_params:,}")
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_SETTINGS["learning_rate"],
        weight_decay=TRAIN_SETTINGS["weight_decay"],
    )
    train_losses, val_losses, tokens_seen = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=TRAIN_SETTINGS["num_epochs"],
        eval_freq=5,
        eval_iter=1,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
    )

    model_path = CHECKPOINTS_DIR / "model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved model to {model_path}")

    epochs_tensor = torch.linspace(0, TRAIN_SETTINGS["num_epochs"], len(train_losses))
    fig, ax1 = plt.subplots()
    ax1.plot(epochs_tensor, train_losses, label="Training loss")
    ax1.plot(epochs_tensor, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plot_path = PLOTS_DIR / "loss.pdf"
    plt.savefig(plot_path)
    print(f"Saved loss plot to {plot_path}")


if __name__ == "__main__":
    main()
