from __future__ import annotations

import json
import os

import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.model_selection import train_test_split


# =========================
# Paths and constants
# =========================

DATA_DIR = "data"

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images", "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images", "test_images")
TRAIN_SPLIT_CSV = os.path.join(DATA_DIR, "train_split.csv")
VAL_SPLIT_CSV = os.path.join(DATA_DIR, "val_split.csv")

MODELS_DIR = "models"
SUBMISSION_DIR = "submission"
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "cnn_resnet18_best.pth")
LABEL_MAPPING_PATH = os.path.join(MODELS_DIR, "cnn_label_mapping.json")


# =========================
# Training settings
# =========================

IMAGE_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 15
NUM_WORKERS = 0
RANDOM_SEED = 42
VAL_RATIO = 0.20

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ButterflyDataset(Dataset):
    """Simple dataset that reads images from train_images/ using file_name."""

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
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Training image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label_index = self.label_to_index[row["TARGET"]]
        return image, label_index


def ensure_file_exists(path: str) -> None:
    """Stop early if a required file or folder is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required path not found: {path}")


def validate_labeled_dataframe(df: pd.DataFrame, name: str) -> None:
    """Check that a label CSV has the columns this script needs."""
    required_columns = {"file_name", "TARGET"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"{name} CSV must contain columns: {sorted(required_columns)}")


def split_train_val(
    input_csv: str = TRAIN_CSV,
    train_output_csv: str = TRAIN_SPLIT_CSV,
    val_output_csv: str = VAL_SPLIT_CSV,
    val_ratio: float = VAL_RATIO,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create reproducible train and validation CSV files from train.csv."""
    ensure_file_exists(input_csv)
    df = pd.read_csv(input_csv)
    validate_labeled_dataframe(df, "Training")

    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        stratify=df["TARGET"],
        random_state=random_seed,
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    os.makedirs(os.path.dirname(train_output_csv), exist_ok=True)
    train_df.to_csv(train_output_csv, index=False)
    val_df.to_csv(val_output_csv, index=False)

    print(f"Total images: {len(df)}")
    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    print(f"Saved training split to: {train_output_csv}")
    print(f"Saved validation split to: {val_output_csv}")

    return train_df, val_df


def build_train_transform() -> transforms.Compose:
    """Data augmentation for the training set."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_val_transform() -> transforms.Compose:
    """Validation and test preprocessing."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_label_mapping(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[dict[str, int], list[str]]:
    """Create one consistent label encoder for both train and validation."""
    all_labels = sorted(set(train_df["TARGET"]).union(set(val_df["TARGET"])))
    label_to_index = {label: index for index, label in enumerate(all_labels)}
    return label_to_index, all_labels


def get_resnet18(num_classes: int) -> nn.Module:
    """Load pretrained ResNet-18 and replace the final classifier."""
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except AttributeError:
        model = models.resnet18(pretrained=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def run_training_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model and return validation loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()

    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = correct_predictions / len(dataloader.dataset)
    return avg_loss, accuracy


def save_label_mapping(class_names: list[str]) -> None:
    """Save class index information for later inference."""
    mapping = {
        "class_names": class_names,
        "label_to_index": {label: index for index, label in enumerate(class_names)},
        "index_to_label": {str(index): label for index, label in enumerate(class_names)},
    }
    with open(LABEL_MAPPING_PATH, "w", encoding="utf-8") as mapping_file:
        json.dump(mapping, mapping_file, indent=2)
    print(f"Saved label mapping to: {LABEL_MAPPING_PATH}")


def validate_required_paths() -> None:
    """Stop early with a helpful error if required files are missing."""
    required_paths = [TRAIN_CSV, TRAIN_IMG_DIR]
    missing_paths = [path for path in required_paths if not os.path.exists(path)]
    if missing_paths:
        missing_text = "\n".join(missing_paths)
        raise FileNotFoundError(f"Missing required project paths:\n{missing_text}")


def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    validate_required_paths()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Creating training and validation split files...")
    train_df, val_df = split_train_val()

    label_to_index, class_names = build_label_mapping(train_df, val_df)
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")

    save_label_mapping(class_names)

    print("Building datasets and dataloaders...")
    train_dataset = ButterflyDataset(
        dataframe=train_df,
        image_dir=TRAIN_IMG_DIR,
        transform=build_train_transform(),
        label_to_index=label_to_index,
    )
    val_dataset = ButterflyDataset(
        dataframe=val_df,
        image_dir=TRAIN_IMG_DIR,
        transform=build_val_transform(),
        label_to_index=label_to_index,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print("Loading pretrained ResNet-18...")
    model = get_resnet18(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = 0.0

    print("Starting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = run_training_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4%}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "class_names": class_names,
                "image_size": IMAGE_SIZE,
            }
            torch.save(checkpoint, CHECKPOINT_PATH)
            print(f"Saved new best model to: {CHECKPOINT_PATH}")

    print("Training finished.")
    print(f"Best validation accuracy: {best_val_accuracy:.4%}")


if __name__ == "__main__":
    main()
