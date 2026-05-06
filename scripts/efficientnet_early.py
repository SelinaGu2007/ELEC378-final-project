"""
EfficientNet-B0 CNN for ELEC378 final project.

This script implements our improved neural-network model using EfficientNet-B0
trained from scratch. This is the early version of EfficientNet.

Pipeline:
1. 80/20 stratified train-validation split
2. image resizing and normalization
3. training-time data augmentation
4. EfficientNet-B0 initialization from scratch
5. AdamW optimization with weight decay
6. validation-based checkpoint selection / early stopping
7. Kaggle submission generation
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


DATA_DIR = "data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMAGE_DIR = os.path.join("train_images", "train_images")
TEST_IMAGE_DIR = os.path.join("test_images", "test_images")
SAMPLE_SUBMISSION_CSV = os.path.join("submission", "sample_submission.csv")

MODELS_DIR = "models"
SUBMISSION_DIR = "submission"
TRAIN_SPLIT_CSV = os.path.join(DATA_DIR, "train_split.csv")
VAL_SPLIT_CSV = os.path.join(DATA_DIR, "val_split.csv")

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class TrainConfig:
    data_dir: str
    train_csv: str
    train_image_dir: str
    test_image_dir: str
    sample_submission_csv: str
    models_dir: str
    submission_dir: str
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    val_ratio: float
    patience: int
    scheduler_patience: int
    scheduler_factor: float
    min_lr: float
    image_size: int
    use_random_erasing: bool
    erasing_prob: float
    random_seed: int
    num_workers: int

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.models_dir, "efficientnet_b0_best.pth")

    @property
    def label_mapping_path(self) -> str:
        return os.path.join(self.models_dir, "efficientnet_label_mapping.json")

    @property
    def submission_path(self) -> str:
        return os.path.join(self.submission_dir, "submission_efficientnet_b0.csv")

    @property
    def train_split_csv(self) -> str:
        return os.path.join(self.data_dir, "train_split.csv")

    @property
    def val_split_csv(self) -> str:
        return os.path.join(self.data_dir, "val_split.csv")


class ButterflyDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: str,
        transform: transforms.Compose,
        label_to_index: dict[str, int],
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.label_to_index = label_to_index

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.dataframe.iloc[index]
        image_path = os.path.join(self.image_dir, row["file_name"])
        image = Image.open(image_path).convert("RGB")
        label = self.label_to_index[row["TARGET"]]
        return self.transform(image), label


class TestImageDataset(Dataset):
    def __init__(self, file_names: list[str], image_dir: str, transform: transforms.Compose) -> None:
        self.file_names = file_names
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = os.path.join(self.image_dir, self.file_names[index])
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 from scratch for butterfly classification.")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--train-image-dir", default=TRAIN_IMAGE_DIR)
    parser.add_argument("--test-image-dir", default=TEST_IMAGE_DIR)
    parser.add_argument("--sample-submission-csv", default=SAMPLE_SUBMISSION_CSV)
    parser.add_argument("--models-dir", default=MODELS_DIR)
    parser.add_argument("--submission-dir", default=SUBMISSION_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--scheduler-patience", type=int, default=6)
    parser.add_argument("--scheduler-factor", type=float, default=0.3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--use-random-erasing", action="store_true")
    parser.add_argument("--erasing-prob", type=float, default=0.08)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    return TrainConfig(
        data_dir=args.data_dir,
        train_csv=os.path.join(args.data_dir, "train.csv"),
        train_image_dir=args.train_image_dir,
        test_image_dir=args.test_image_dir,
        sample_submission_csv=args.sample_submission_csv,
        models_dir=args.models_dir,
        submission_dir=args.submission_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        patience=args.patience,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        min_lr=args.min_lr,
        image_size=args.image_size,
        use_random_erasing=args.use_random_erasing,
        erasing_prob=args.erasing_prob,
        random_seed=args.random_seed,
        num_workers=args.num_workers,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def require_columns(df: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns.difference(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def load_and_split_data(config: TrainConfig) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, int]]:
    if not os.path.exists(config.train_csv):
        raise FileNotFoundError(f"Training CSV not found: {config.train_csv}")
    if not os.path.isdir(config.train_image_dir):
        raise FileNotFoundError(f"Training image directory not found: {config.train_image_dir}")

    df = pd.read_csv(config.train_csv)
    require_columns(df, {"file_name", "TARGET"}, "Training CSV")

    train_df, val_df = train_test_split(
        df,
        test_size=config.val_ratio,
        stratify=df["TARGET"],
        random_state=config.random_seed,
        shuffle=True,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    os.makedirs(config.data_dir, exist_ok=True)
    train_df.to_csv(config.train_split_csv, index=False)
    val_df.to_csv(config.val_split_csv, index=False)

    class_names = sorted(df["TARGET"].unique().tolist())
    label_to_index = {label: index for index, label in enumerate(class_names)}
    return train_df, val_df, class_names, label_to_index


def build_train_transform(config: TrainConfig) -> transforms.Compose:
    transform_steps: list[object] = [
        transforms.RandomResizedCrop(config.image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if config.use_random_erasing:
        transform_steps.append(transforms.RandomErasing(p=config.erasing_prob, scale=(0.01, 0.04), ratio=(0.5, 2.0)))
    return transforms.Compose(transform_steps)


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_model(num_classes: int) -> nn.Module:
    try:
        model = models.efficientnet_b0(weights=None)
    except TypeError:
        model = models.efficientnet_b0(pretrained=False)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def save_label_mapping(class_names: list[str], path: str) -> None:
    mapping = {
        "class_names": class_names,
        "label_to_index": {label: index for index, label in enumerate(class_names)},
        "index_to_label": {str(index): label for index, label in enumerate(class_names)},
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(mapping, file, indent=2)


def train_model(
    config: TrainConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_names: list[str],
    device: torch.device,
) -> nn.Module:
    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.min_lr,
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print("Training EfficientNet-B0 from scratch (weights=None).")
    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4%} | "
            f"LR: {current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "image_size": config.image_size,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "config": vars(config),
                },
                config.checkpoint_path,
            )
            print(f"Saved best checkpoint to: {config.checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping after {config.patience} epochs without validation-loss improvement.")
                break

    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def test_file_names_from_submission(submission_df: pd.DataFrame, test_image_dir: str) -> list[str]:
    if "file_name" in submission_df.columns:
        return submission_df["file_name"].astype(str).tolist()
    if "ID" not in submission_df.columns:
        raise ValueError("Sample submission must contain either an ID column or a file_name column.")

    file_names = []
    for image_id in submission_df["ID"].astype(str):
        candidate = image_id if os.path.splitext(image_id)[1] else f"{image_id}.jpg"
        if not os.path.exists(os.path.join(test_image_dir, candidate)):
            raise FileNotFoundError(f"Test image not found: {os.path.join(test_image_dir, candidate)}")
        file_names.append(candidate)
    return file_names


@torch.no_grad()
def write_submission(config: TrainConfig, model: nn.Module, class_names: list[str], device: torch.device) -> None:
    if not os.path.isdir(config.test_image_dir):
        print(f"Skipping submission: test image directory not found: {config.test_image_dir}")
        return
    if not os.path.exists(config.sample_submission_csv):
        print(f"Skipping submission: sample submission not found: {config.sample_submission_csv}")
        return

    submission_df = pd.read_csv(config.sample_submission_csv)
    require_columns(submission_df, {"TARGET"}, "Sample submission")
    file_names = test_file_names_from_submission(submission_df, config.test_image_dir)

    dataset = TestImageDataset(file_names, config.test_image_dir, build_eval_transform(config.image_size))
    loader = make_loader(dataset, config.batch_size, shuffle=False, num_workers=config.num_workers, seed=config.random_seed)

    model.eval()
    predictions: list[str] = []
    for images in loader:
        images = images.to(device)
        predicted_indices = model(images).argmax(dim=1).cpu().tolist()
        predictions.extend(class_names[index] for index in predicted_indices)

    os.makedirs(config.submission_dir, exist_ok=True)
    submission_df["TARGET"] = predictions
    submission_df.to_csv(config.submission_path, index=False)
    print(f"Saved submission to: {config.submission_path}")


def main() -> None:
    config = parse_args()
    set_seed(config.random_seed)
    os.makedirs(config.models_dir, exist_ok=True)
    os.makedirs(config.submission_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df, val_df, class_names, label_to_index = load_and_split_data(config)
    save_label_mapping(class_names, config.label_mapping_path)
    print(f"Train images: {len(train_df)} | Validation images: {len(val_df)} | Classes: {len(class_names)}")
    print(f"Saved split CSVs to: {config.train_split_csv} and {config.val_split_csv}")

    train_dataset = ButterflyDataset(train_df, config.train_image_dir, build_train_transform(config), label_to_index)
    val_dataset = ButterflyDataset(val_df, config.train_image_dir, build_eval_transform(config.image_size), label_to_index)
    train_loader = make_loader(train_dataset, config.batch_size, True, config.num_workers, config.random_seed)
    val_loader = make_loader(val_dataset, config.batch_size, False, config.num_workers, config.random_seed)

    model = train_model(config, train_loader, val_loader, class_names, device)

    criterion = nn.CrossEntropyLoss()
    final_val_loss, final_val_accuracy = evaluate(model, val_loader, criterion, device)
    print(f"Best checkpoint validation loss: {final_val_loss:.4f}")
    print(f"Best checkpoint validation accuracy: {final_val_accuracy:.4%}")

    write_submission(config, model, class_names, device)


if __name__ == "__main__":
    main()
