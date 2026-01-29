import argparse
import json

import matplotlib.pyplot as plt
import torch

from llms_from_scratch.classification import (
    GPTForClassification,
    calc_accuracy_loader,
    train_classifier,
)
from llms_from_scratch.data_classification import load_spam_data
from llms_from_scratch.models.gpt2 import load_weights
from llms_from_scratch.paths import CHECKPOINTS_DIR, PLOTS_DIR, ensure_dirs


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 for spam classification")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2-small-124M",
        help="Pretrained model name (default: gpt2-small-124M)",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument(
        "--weight-decay", type=float, default=0.1, help="Weight decay (default: 0.1)"
    )
    parser.add_argument("--eval-freq", type=int, default=50, help="Evaluation frequency (default: 50)")
    parser.add_argument("--eval-iter", type=int, default=5, help="Evaluation iterations (default: 5)")
    return parser.parse_args()


def print_config(args, device):
    print("Fine-tuning Configuration:")
    print(f"  model: {args.model}")
    print(f"  epochs: {args.epochs}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  learning_rate: {args.lr}")
    print(f"  weight_decay: {args.weight_decay}")
    print(f"  device: {device}")
    print()


def load_data_and_model(args, device):
    print("Loading spam dataset...")
    train_loader, val_loader, test_loader, max_length = load_spam_data(
        batch_size=args.batch_size
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Max sequence length: {max_length}")
    print()

    print(f"Loading pretrained model: {args.model}")
    base_model = load_weights(model_name=args.model, device=str(device))
    model = GPTForClassification(base_model, num_classes=2, freeze_base=True)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print()

    return train_loader, val_loader, test_loader, max_length, model


def run_final_eval(train_loader, val_loader, test_loader, model, device):
    print("Final evaluation on full datasets...")
    train_acc = calc_accuracy_loader(train_loader, model, device)
    val_acc = calc_accuracy_loader(val_loader, model, device)
    test_acc = calc_accuracy_loader(test_loader, model, device)
    print(f"  Training accuracy: {train_acc*100:.2f}%")
    print(f"  Validation accuracy: {val_acc*100:.2f}%")
    print(f"  Test accuracy: {test_acc*100:.2f}%")
    print()
    return train_acc, val_acc, test_acc


def save_model_and_metadata(model, args, max_length, test_acc):
    model_path = CHECKPOINTS_DIR / "spam_classifier.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    metadata = {
        "base_model": args.model,
        "max_length": max_length,
        "num_classes": 2,
        "test_accuracy": test_acc,
    }
    metadata_path = CHECKPOINTS_DIR / "spam_classifier_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")


def save_plots(args, train_losses, val_losses, train_accs, val_accs, examples_seen):
    epochs_tensor = torch.linspace(0, args.epochs, len(train_losses))
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_tensor, train_losses, label="Training loss")
    ax1.plot(epochs_tensor, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax2 = ax1.twiny()
    examples_tensor = torch.linspace(0, examples_seen, len(train_losses))
    ax2.plot(examples_tensor, train_losses, alpha=0)
    ax2.set_xlabel("Examples seen")
    fig.tight_layout()
    loss_plot_path = PLOTS_DIR / "classification_loss.pdf"
    plt.savefig(loss_plot_path)
    print(f"Saved loss plot to {loss_plot_path}")

    epochs_acc_tensor = torch.linspace(0, args.epochs, len(train_accs))
    fig2, ax3 = plt.subplots(figsize=(5, 3))
    ax3.plot(epochs_acc_tensor, train_accs, label="Training accuracy")
    ax3.plot(epochs_acc_tensor, val_accs, linestyle="-.", label="Validation accuracy")
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Accuracy")
    ax3.legend()
    ax4 = ax3.twiny()
    examples_acc_tensor = torch.linspace(0, examples_seen, len(train_accs))
    ax4.plot(examples_acc_tensor, train_accs, alpha=0)
    ax4.set_xlabel("Examples seen")
    fig2.tight_layout()
    acc_plot_path = PLOTS_DIR / "classification_accuracy.pdf"
    plt.savefig(acc_plot_path)
    print(f"Saved accuracy plot to {acc_plot_path}")


def main():
    args = parse_args()
    ensure_dirs()
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_config(args, device)
    train_loader, val_loader, test_loader, max_length, model = load_data_and_model(
        args, device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    print("Starting training...")
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
    )
    print()

    _, _, test_acc = run_final_eval(
        train_loader, val_loader, test_loader, model, device
    )
    save_model_and_metadata(model, args, max_length, test_acc)
    save_plots(args, train_losses, val_losses, train_accs, val_accs, examples_seen)


if __name__ == "__main__":
    main()
