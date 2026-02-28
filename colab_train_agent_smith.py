"""
Google Colab training script for Agent Smith with a custom dataset.

Requirements (Colab cell before running):
!pip install -q transformers
"""

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm


# --- Google Drive Mount (Colab) ---
# Uncomment these lines in Colab.
# from google.colab import drive
# drive.mount("/content/drive")


@dataclass
class TrainConfig:
    dataset_root: str = "/content/drive/MyDrive/dataset/train"
    model_name: str = "bert-base-uncased"
    max_length: int = 256
    batch_size: int = 8
    num_epochs: int = 5
    learning_rate: float = 0.001
    train_split: float = 0.8
    seed: int = 42
    best_model_path: str = "/content/drive/MyDrive/model_weights.pth"


class TextFolderDataset(Dataset):
    """Custom dataset that reads text files from class-named folders and tokenizes them."""

    def __init__(
        self,
        root_dir: str,
        tokenizer: AutoTokenizer,
        max_length: int,
        allowed_extensions: Tuple[str, ...] = (
            ".txt",
            ".md",
            ".json",
            ".csv",
            ".log",
            ".yaml",
            ".yml",
        ),
    ) -> None:
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.allowed_extensions = allowed_extensions
        self.samples: List[Tuple[str, int]] = []
        self.class_to_idx: Dict[str, int] = {}
        self._build_index()

    def _build_index(self) -> None:
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(
                f"Dataset root not found: {self.root_dir}. "
                "Ensure Google Drive is mounted and the path is correct."
            )

        class_names = sorted(
            [
                name
                for name in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, name))
            ]
        )
        if not class_names:
            raise ValueError(
                "No class folders found. Expected one folder per class under the train directory."
            )

        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        for class_name in class_names:
            class_dir = os.path.join(self.root_dir, class_name)
            for root, _, files in os.walk(class_dir):
                for fname in files:
                    if self._is_allowed_file(fname):
                        path = os.path.join(root, fname)
                        self.samples.append((path, self.class_to_idx[class_name]))

        if not self.samples:
            raise ValueError(
                "No supported text files found. "
                "Update allowed_extensions or verify your dataset contents."
            )

    def _is_allowed_file(self, filename: str) -> bool:
        return filename.lower().endswith(self.allowed_extensions)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path, label = self.samples[idx]
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(dataset: Dataset, config: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    train_size = int(config.train_split * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader


def freeze_backbone(model: AutoModelForSequenceClassification) -> None:
    for param in model.base_model.parameters():
        param.requires_grad = False


def evaluate(model: AutoModelForSequenceClassification, data_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def train() -> None:
    config = TrainConfig()
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    dataset = TextFolderDataset(
        root_dir=config.dataset_root,
        tokenizer=tokenizer,
        max_length=config.max_length,
    )

    num_classes = len(dataset.class_to_idx)
    print(f"Detected {num_classes} classes: {dataset.class_to_idx}")

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=num_classes,
    )
    freeze_backbone(model)
    model.to(device)

    train_loader, val_loader = build_dataloaders(dataset, config)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_accuracy = 0.0
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        running_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}")
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            progress.set_postfix(loss=loss.item())

        train_loss = running_loss / max(len(train_loader.dataset), 1)
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), config.best_model_path)
            print(f"Saved new best model to {config.best_model_path}")


if __name__ == "__main__":
    train()
