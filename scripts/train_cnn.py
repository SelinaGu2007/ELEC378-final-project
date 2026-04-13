
import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split


# Reproducibility
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



# Paths and constants
DATA_DIR = "data"

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images", "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images", "test_images")

FILENAME_COL = "file_name"
LABEL_COL = "TARGET"
ID_COL = "ID"

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5
IMAGE_SIZE = (128, 128)
SEED = 42



# Dataset
class ButterflyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None, labeled: bool = True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.labeled = labeled

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row[FILENAME_COL])

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.labeled:
            label = int(row["label_idx"])
            return image, label

        return image, row[ID_COL]



# Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# Training / evaluation
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, total_correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, total_correct / total


# Main
if __name__ == "__main__":
    set_seed(SEED)

    # Check files
    required_paths = [TRAIN_CSV, SAMPLE_SUB_CSV, TRAIN_IMG_DIR, TEST_IMG_DIR]
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required path not found: {path}")

    # Load data
    train_df = pd.read_csv(TRAIN_CSV)
    sample_sub_df = pd.read_csv(SAMPLE_SUB_CSV)

    print("Loaded training CSV.")
    print("Training rows:", len(train_df))
    print("Sample submission rows:", len(sample_sub_df))
    print("Training columns:", list(train_df.columns))
    print("Sample submission columns:", list(sample_sub_df.columns))

    # Train / validation split
    train_split_df, val_split_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=SEED,
        stratify=train_df[LABEL_COL],
    )

    # Label encoding
    class_names = sorted(train_df[LABEL_COL].unique())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    train_split_df = train_split_df.copy()
    val_split_df = val_split_df.copy()

    train_split_df["label_idx"] = train_split_df[LABEL_COL].map(class_to_idx)
    val_split_df["label_idx"] = val_split_df[LABEL_COL].map(class_to_idx)

    print("Train split:", len(train_split_df))
    print("Val split:", len(val_split_df))
    print("Number of classes:", len(class_names))

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    # Datasets / loaders
    train_dataset = ButterflyDataset(
        train_split_df,
        TRAIN_IMG_DIR,
        transform=train_transform,
        labeled=True,
    )
    val_dataset = ButterflyDataset(
        val_split_df,
        TRAIN_IMG_DIR,
        transform=val_transform,
        labeled=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Device / model / optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = SimpleCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train
    best_val_acc = 0.0
    best_state = None

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    if best_state is None:
        raise RuntimeError("Training finished without saving a best model state.")

    model.load_state_dict(best_state)
    print("Best validation accuracy:", best_val_acc)

    # Save best model
    torch.save(best_state, "best_cnn_model.pth")
    print("Saved best_cnn_model.pth")

    # Build test dataframe from sample submission IDs
    test_df = sample_sub_df.copy()
    test_df[FILENAME_COL] = test_df[ID_COL] + ".jpg"

    # Test dataset / loader
    test_dataset = ButterflyDataset(
        test_df,
        TEST_IMG_DIR,
        transform=val_transform,
        labeled=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Predict test labels
    model.eval()
    all_ids = []
    all_pred_labels = []

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_ids.extend(ids)
            all_pred_labels.extend([idx_to_class[p] for p in preds])

    print("Predictions made:", len(all_pred_labels))

    # Save submission
    submission_df = pd.DataFrame({
        ID_COL: all_ids,
        LABEL_COL: all_pred_labels,
    })

    submission_df.to_csv("submission_cnn.csv", index=False)
    print("Saved submission_cnn.csv")
    print(submission_df.head())
