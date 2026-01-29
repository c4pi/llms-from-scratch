from pathlib import Path

import requests
import torch
from torch.utils.data import DataLoader, Dataset

from llms_from_scratch.paths import RAW_DIR, ensure_dirs

VERDICT_URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
VERDICT_FILENAME = "the-verdict.txt"


def download_verdict_text(force: bool = False) -> Path:
    ensure_dirs()
    file_path = RAW_DIR / VERDICT_FILENAME
    if not file_path.exists() or force:
        response = requests.get(VERDICT_URL, timeout=60)
        response.raise_for_status()
        with file_path.open("w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"Downloaded to {file_path}")
    return file_path


def load_verdict_text() -> str:
    file_path = RAW_DIR / VERDICT_FILENAME
    if not file_path.exists():
        download_verdict_text()
    with file_path.open(encoding="utf-8") as f:
        return f.read()


class GPTDataset(Dataset):
    def __init__(self, txt: str, tokenizer, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    txt: str,
    tokenizer,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
