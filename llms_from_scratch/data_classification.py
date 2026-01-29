from pathlib import Path
import zipfile

import pandas as pd
import requests
import tiktoken
from torch.utils.data import DataLoader

from llms_from_scratch.classification import SpamDataset
from llms_from_scratch.paths import RAW_DIR, ensure_dirs

SPAM_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SPAM_BACKUP_URL = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
SPAM_DIR = "sms_spam_collection"
SPAM_FILENAME = "SMSSpamCollection.tsv"


def download_spam_dataset(force: bool = False) -> Path:
    ensure_dirs()
    data_dir = RAW_DIR / SPAM_DIR
    data_file = data_dir / SPAM_FILENAME
    if data_file.exists() and not force:
        return data_file
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "sms_spam_collection.zip"
    for url in [SPAM_URL, SPAM_BACKUP_URL]:
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            with zip_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            break
        except requests.exceptions.RequestException as e:
            print(f"Failed to download from {url}: {e}")
            continue
    else:
        raise RuntimeError("Failed to download spam dataset from all URLs")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    original_file = data_dir / "SMSSpamCollection"
    if original_file.exists():
        original_file.rename(data_file)
    zip_path.unlink()
    print(f"Downloaded spam dataset to {data_file}")
    return data_file


def create_balanced_dataset(df: pd.DataFrame) -> pd.DataFrame:
    n_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(n_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df


def random_split(df: pd.DataFrame, train_frac: float, val_frac: float):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    val_end = train_end + int(len(df) * val_frac)
    return df[:train_end], df[train_end:val_end], df[val_end:]


def load_spam_data(
    batch_size: int = 8, num_workers: int = 0
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    data_file = download_spam_dataset()
    df = pd.read_csv(data_file, sep="\t", header=None, names=["Label", "Text"])
    balanced_df = create_balanced_dataset(df)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    train_df, val_df, test_df = random_split(balanced_df, 0.7, 0.1)
    ensure_dirs()
    train_csv = RAW_DIR / SPAM_DIR / "train.csv"
    val_csv = RAW_DIR / SPAM_DIR / "validation.csv"
    test_csv = RAW_DIR / SPAM_DIR / "test.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset = SpamDataset(train_csv, tokenizer, max_length=None)
    max_length = train_dataset.max_length
    val_dataset = SpamDataset(val_csv, tokenizer, max_length=max_length)
    test_dataset = SpamDataset(test_csv, tokenizer, max_length=max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader, max_length
