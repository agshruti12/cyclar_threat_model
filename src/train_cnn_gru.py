# train_cnn_gru.py

import glob
import os
from collections import Counter
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as Ffunc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from cnn_video_dataset import SegmentVideoDataset
from cnn_gru_model import CNNGRUDangerModel


def _guess_video_from_label_json(label_json_path: str) -> str:
    """
    Best-effort mapping from '*_scene_labels.json' -> video filename.
    Adjust extensions / directory rules if your dataset differs.
    """
    base = os.path.basename(label_json_path)
    stem = base.replace("_scene_labels.json", "")
    # Common guesses; keep first as a "name" even if path unknown
    return stem  # just the stem; dataset-derived paths are preferred


def extract_video_names_from_dataset(dataset: SegmentVideoDataset) -> List[str]:
    """
    Try to pull video identifiers from dataset.samples if available.
    Falls back to label_json_paths guessing.
    """
    names = set()

    # 1) Try to extract from dataset.samples (most reliable if present)
    sample_keys_to_try = [
        "video_path", "video_file", "video", "mp4_path", "path"
    ]
    if hasattr(dataset, "samples"):
        for s in dataset.samples:
            for k in sample_keys_to_try:
                if k in s and s[k]:
                    names.add(os.path.basename(str(s[k])))
                    break

    # 2) Fallback: guess from label json list stored on dataset (if present)
    if not names:
        for attr in ["label_json_paths", "label_paths", "labels"]:
            if hasattr(dataset, attr):
                for p in getattr(dataset, attr):
                    names.add(_guess_video_from_label_json(str(p)))
                break

    return sorted(names)


def log_training_videos(
    train_files: List[str],
    val_files: List[str],
    train_dataset: SegmentVideoDataset,
    val_dataset: SegmentVideoDataset,
    out_txt_path: str = "models/training_videos_used.txt",
) -> None:
    """
    Print and save which videos (or video-ids) are used for train/val splits.
    Saves both: (a) raw label json filenames and (b) inferred/known video names.
    """
    os.makedirs(os.path.dirname(out_txt_path) or ".", exist_ok=True)

    train_video_names = extract_video_names_from_dataset(train_dataset)
    val_video_names = extract_video_names_from_dataset(val_dataset)

    # Console print
    print("\n=== Videos used (TRAIN split) ===")
    for v in train_video_names:
        print("  ", v)

    print("\n=== Videos used (VAL split) ===")
    for v in val_video_names:
        print("  ", v)

    # Write file
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("TRAIN LABEL JSON FILES:\n")
        for p in train_files:
            f.write(f"{p}\n")

        f.write("\nVAL LABEL JSON FILES:\n")
        for p in val_files:
            f.write(f"{p}\n")

        f.write("\nTRAIN VIDEO NAMES (best-effort):\n")
        for v in train_video_names:
            f.write(f"{v}\n")

        f.write("\nVAL VIDEO NAMES (best-effort):\n")
        for v in val_video_names:
            f.write(f"{v}\n")

    print(f"\nSaved training/val video list to: {out_txt_path}\n")


def find_label_files(labels_dir: str) -> List[str]:
    pattern = os.path.join(labels_dir, "*_scene_labels.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No label JSONs found in {labels_dir}")
    return files


def compute_class_counts(dataset: SegmentVideoDataset) -> Counter:
    counts = Counter()
    for i in range(len(dataset.samples)):
        lbl = dataset.samples[i]["label"]
        counts[int(lbl)] += 1
    print("Class counts (segments):")
    for c in range(3):
        print(f"  class {c}: {counts.get(c, 0)}")
    return counts


def make_weighted_sampler(dataset: SegmentVideoDataset, class_counts: Counter) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler so MED/HIGH segments are sampled more often.
    """
    num_classes = 3
    total = sum(class_counts.values())
    class_counts_list = [class_counts.get(c, 1) for c in range(num_classes)]
    class_weights = [total / (num_classes * c) for c in class_counts_list]

    # Weight per sample
    weights = []
    for s in dataset.samples:
        lbl = int(s["label"])
        weights.append(class_weights[lbl])

    weights_tensor = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights_tensor, num_samples=len(weights_tensor), replacement=True)

    print("Class weights (inverse freq):", class_weights)
    return sampler


def train(
    labels_dir: str = "data/labels",
    frames_per_segment: int = 8,
    img_size: int = 224,
    batch_size: int = 4,
    num_epochs: int = 3,          # keep small at first
    lr: float = 1e-4,
    train_backbone: bool = False,
    device: str = None,
    max_train_segs: int = 200,    # DEBUG: cap how many segments we use
):
    # ---- device selection ----
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Using device: {device}")

    label_files = find_label_files(labels_dir)
    print(f"Found {len(label_files)} label files.")

    # Split by video (label file)
    train_files, val_files = train_test_split(label_files, test_size=0.2, random_state=42)

    # ---- build datasets ----
    train_dataset = SegmentVideoDataset(
        label_json_paths=train_files,
        frames_per_segment=frames_per_segment,
        img_size=img_size,
        train=True,
        drop_unlabeled=True,
    )

    # DEBUG: cap number of training segments so it doesn't fry your laptop
    if max_train_segs is not None and len(train_dataset.samples) > max_train_segs:
        train_dataset.samples = train_dataset.samples[:max_train_segs]
        print(f"[DEBUG] Capped train segments to first {max_train_segs}")

    val_dataset = SegmentVideoDataset(
        label_json_paths=val_files,
        frames_per_segment=frames_per_segment,
        img_size=img_size,
        train=False,
        drop_unlabeled=True,
    )

    print(f"Train segments: {len(train_dataset.samples)}")
    print(f"Val segments:   {len(val_dataset.samples)}")

    # ---- log which videos we trained/validated on ----
    log_training_videos(
        train_files=train_files,
        val_files=val_files,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        out_txt_path="models/training_videos_used.txt",
    )


    # ---- class counts & sampler ----
    class_counts = compute_class_counts(train_dataset)
    sampler = make_weighted_sampler(train_dataset, class_counts)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,  # keep 0 to avoid OpenCV multi-process issues
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Val batches per epoch:   {len(val_loader)}")

    # ---- model ----
    model = CNNGRUDangerModel(
        cnn_feature_dim=512,
        gru_hidden_dim=128,
        gru_layers=1,
        num_classes=3,
        train_backbone=train_backbone,
    )
    model.to(device)

    # ---- loss with class weights ----
    num_classes = 3
    total = sum(class_counts.values())
    counts_list = [class_counts.get(c, 1) for c in range(num_classes)]
    class_weights = [total / (num_classes * c) for c in counts_list]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print("Class weights:", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    print("Starting training...")
    for epoch in range(1, num_epochs + 1):
        # ---- Train ----
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (X, y) in enumerate(train_loader):
            X = X.to(device)   # (B, T, C, H, W)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(X)   # (B, 3)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += X.size(0)

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
                print(
                    f"  Epoch {epoch:02d} | "
                    f"batch {batch_idx+1}/{len(train_loader)} | "
                    f"loss {loss.item():.4f}"
                )

        train_loss = train_loss_sum / max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        # ---- Val ----
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
            f"Epoch {epoch:02d} DONE | "
            f"train loss {train_loss:.4f}, acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.3f}"
        )

    os.makedirs("models", exist_ok=True)
    out_path = "models/cnn_gru_danger_model.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Saved model to {out_path}")



if __name__ == "__main__":
    train(
        labels_dir="data/labels",
        frames_per_segment=4,
        img_size=192,
        batch_size=2,     # small because CNN is heavier
        num_epochs=10,
        lr=1e-4,
        train_backbone=False,   # start with frozen backbone
    )
