# train_danger_model.py

import json
import os
from collections import Counter
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as Ffunc
from torch.utils.data import DataLoader

from src.dataset import DangerVideoDataset, TEMPORAL_WINDOW
from src.model import DangerGRUClassifier


# -----------------------------
# Utilities
# -----------------------------


def load_splits(path: str) -> Dict[str, List[str]]:
    with open(path, "r") as f:
        return json.load(f)


def inspect_class_distribution(dataset: DangerVideoDataset) -> Counter:
    """
    Count occurrences of each class label in the dataset windows.
    """
    counts = Counter()
    for i in range(len(dataset)):
        _, y = dataset[i]
        counts[int(y)] += 1

    print("Class distribution in training windows:")
    for c in range(3):
        print(f"  class {c}: {counts.get(c, 0)}")
    return counts


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with optional per-class alpha weights.

    This is helpful when we have extreme class imbalance and want to
    downweight very easy examples (e.g., the many LOW windows).
    """

    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha  # shape (num_classes,) or None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross-entropy per sample (no reduction)
        ce_loss = Ffunc.cross_entropy(logits, targets, weight=self.alpha, reduction="none")  # (B,)

        # Probabilities for the true class
        probs = Ffunc.softmax(logits, dim=1)  # (B, C)
        pt = probs[torch.arange(len(targets)), targets]  # (B,)

        focal_term = (1.0 - pt) ** self.gamma
        loss = focal_term * ce_loss  # (B,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def create_dataloaders(
    splits_json: str,
    batch_size: int = 64,
    low_stride_train: int = 5,
) -> (DataLoader, DataLoader, DangerVideoDataset, DangerVideoDataset):
    """
    Create train/val DataLoaders.

    We:
      - Use LOW subsampling (low_stride_train) on the training dataset
      - Do NOT subsample LOW in validation (low_stride=1) so that val reflects true distribution
    """
    splits = load_splits(splits_json)

    train_paths = splits["train"]
    val_paths = splits["val"]

    train_ds = DangerVideoDataset(
        train_paths,
        temporal_window=TEMPORAL_WINDOW,
        low_stride=low_stride_train,
        include_labels=True,
    )
    val_ds = DangerVideoDataset(
        val_paths,
        temporal_window=TEMPORAL_WINDOW,
        low_stride=1,  # keep real distribution for validation
        include_labels=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, train_ds, val_ds


# -----------------------------
# Training
# -----------------------------


def train_model(
    splits_json: str = "data/training/splits.json",
    num_epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    low_stride_train: int = 5,
    use_focal_loss: bool = True,
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, val_loader, train_ds, val_ds = create_dataloaders(
        splits_json=splits_json,
        batch_size=batch_size,
        low_stride_train=low_stride_train,
    )

    # Inspect class imbalance on TRAIN windows (after LOW subsampling)
    counts = inspect_class_distribution(train_ds)
    num_classes = 3
    total = sum(counts.values())
    class_counts = [counts.get(c, 1) for c in range(num_classes)]  # avoid div by 0

    # Inverse-frequency weights: more weight for rarer classes
    raw_weights = [total / (num_classes * cc) for cc in class_counts]

    # Convert to tensor
    class_weights = torch.tensor(raw_weights, dtype=torch.float32, device=device)
    print("Class weights (inverse-frequency):", raw_weights)

    # Get feature_dim from one batch
    sample_X, _ = next(iter(train_loader))
    _, T, feat_dim = sample_X.shape
    print(f"Feature dim: {feat_dim}, temporal window: {T}")

    model = DangerGRUClassifier(
        feature_dim=feat_dim,
        hidden_dim=64,
        num_layers=1,
        num_classes=num_classes,
    )
    model.to(device)

    if use_focal_loss:
        criterion = FocalLoss(alpha=class_weights, gamma=2.0, reduction="mean")
        print("Using FocalLoss with class weights.")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using class-weighted CrossEntropyLoss.")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # -----------------------------
    # Epoch loop
    # -----------------------------
    for epoch in range(1, num_epochs + 1):
        # ---- Train ----
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(X)              # (B, C)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += X.size(0)

        train_loss = train_loss_sum / max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        # ---- Validation ----
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)

                logits = model(X)
                loss = criterion(logits, y)

                val_loss_sum += loss.item() * X.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += X.size(0)

        val_loss = val_loss_sum / max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.3f}"
        )

    # Save model
    os.makedirs("models", exist_ok=True)
    out_path = "models/danger_gru_classifier.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    # With extremely skewed data (thousands of LOW, ~10–12 MED/HIGH),
    # you likely want strong LOW subsampling and focal loss.
    train_model(
        splits_json="data/training/splits.json",
        num_epochs=40,
        lr=1e-3,
        batch_size=64,
        low_stride_train=10,   # strong subsampling of LOW windows
        use_focal_loss=True,
    )
