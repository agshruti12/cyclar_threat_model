import glob
import json
import os
import numpy as np


def make_splits(npz_dir="data/training", out_path="data/training/splits.json"):
    npz_paths = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    print("Found npz files:")
    for p in npz_paths:
        print(" ", p)

    n = len(npz_paths)
    idxs = np.arange(n)
    np.random.shuffle(idxs)

    train_end = int(0.6 * n)
    val_end = int(0.8 * n)

    train_paths = [npz_paths[i] for i in idxs[:train_end]]
    val_paths = [npz_paths[i] for i in idxs[train_end:val_end]]
    test_paths = [npz_paths[i] for i in idxs[val_end:]]

    splits = {
        "train": train_paths,
        "val": val_paths,
        "test": test_paths,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Saved splits to {out_path}")


if __name__ == "__main__":
    make_splits()
