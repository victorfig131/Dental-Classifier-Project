import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from torchvision.models import ResNet18_Weights


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.transforms import get_classifier_eval_transforms, get_classifier_train_transforms

DATASET_ROOT = PROJECT_ROOT / "dataset"
MODEL_OUT_PATH = PROJECT_ROOT / "classifier" / "model" / "classifier_3class.pth"


def build_dataloaders(
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Create train/validation dataloaders from dataset/ using an 80/20 split."""
    train_dataset = datasets.ImageFolder(str(DATASET_ROOT), transform=get_classifier_train_transforms())
    val_dataset = datasets.ImageFolder(str(DATASET_ROOT), transform=get_classifier_eval_transforms())

    dataset_size = len(train_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size

    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(dataset_size, generator=generator).tolist()

    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, train_dataset.class_to_idx


def train(
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 32,
    val_split: float = 0.2,
) -> Path:
    """Train ResNet18 for 3-way classification and save classifier weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, class_to_idx = build_dataloaders(
        batch_size=batch_size,
        val_split=val_split,
    )

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / max(1, len(train_loader.dataset))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / max(1, total)
        print(f"Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

    MODEL_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "arch": "resnet18",
        },
        MODEL_OUT_PATH,
    )

    print(f"Saved classifier checkpoint to: {MODEL_OUT_PATH}")
    return MODEL_OUT_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 3-class dental image classifier.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_split=args.val_split,
    )
