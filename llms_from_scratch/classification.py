import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]
        if max_length is None:
            self.max_length = max(len(t) for t in self.encoded_texts)
        else:
            self.max_length = max_length
            self.encoded_texts = [t[: self.max_length] for t in self.encoded_texts]
        self.encoded_texts = [
            t + [pad_token_id] * (self.max_length - len(t)) for t in self.encoded_texts
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encoded = self.encoded_texts[idx]
        label = self.data.iloc[idx]["Label"]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class GPTForClassification(nn.Module):
    def __init__(self, base_model, num_classes=2, freeze_base=True):
        super().__init__()
        self.base_model = base_model
        emb_dim = base_model.out_head.in_features
        self.base_model.out_head = nn.Linear(emb_dim, num_classes)
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            for param in self.base_model.trf_blocks[-1].parameters():
                param.requires_grad = True
            for param in self.base_model.final_norm.parameters():
                param.requires_grad = True
            for param in self.base_model.out_head.parameters():
                param.requires_grad = True

    def forward(self, input_ids):
        logits = self.base_model(input_ids, use_cache=False)
        return logits[:, -1, :]


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct, total = 0, 0
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            inputs = input_batch.to(device)
            targets = target_batch.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total if total > 0 else 0.0


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches


def train_classifier(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.size(0)
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )
        train_acc = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_acc = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_acc*100:.2f}% | Validation accuracy: {val_acc*100:.2f}%")
        train_accs.append(train_acc)
        val_accs.append(val_acc)
    return train_losses, val_losses, train_accs, val_accs, examples_seen


def classify_text(text, model, tokenizer, device, max_length, pad_token_id=50256):
    model.eval()
    input_ids = tokenizer.encode(text)
    input_ids = input_ids[:max_length]
    input_ids = input_ids + [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor)
    pred = torch.argmax(logits, dim=-1).item()
    probs = torch.softmax(logits, dim=-1)
    confidence = probs[0, pred].item()
    return "spam" if pred == 1 else "not spam", confidence
