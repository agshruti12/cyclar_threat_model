# dataset.py

import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Number of frames in each temporal window
TEMPORAL_WINDOW = 15  # adjust if you used a different value during training


class DangerVideoDataset(Dataset):
    """
    Dataset over multiple video .npz files produced by build_frame_data_from_tracks.py.

    Each .npz is expected to contain:
      - frame_features: (N, F)
      - frame_labels:   (N,) int array with values {0,1,2} or -1 for unlabeled
      - fps, frame_count, video_path (not used here)

    Each sample is:
      X: (T, F) float32  (sliding window over T frames)
      y: scalar int64 in {0,1,2} (danger class at last frame in the window)

    We address class imbalance by:
      - Subsampling LOW (class 0) windows: keep only every `low_stride`-th LOW window.
        All MED/HIGH windows are kept.
    """

    def __init__(
        self,
        npz_paths: List[str],
        temporal_window: int = TEMPORAL_WINDOW,
        low_stride: int = 1,
        include_labels: bool = True,
    ):
        """
        Args:
            npz_paths: list of paths to .npz files.
            temporal_window: number of frames in each window (T).
            low_stride: keep every `low_stride`-th LOW (class 0) window; 1 = keep all.
            include_labels: if False, labels are ignored (set to -1). Useful for inference.
        """
        self.temporal_window = temporal_window
        self.low_stride = max(1, low_stride)
        self.videos = []
        self.samples: List[Tuple[int, int]] = []  # (vid_idx, end_frame_idx)

        # Load all videos
        for path in npz_paths:
            data = np.load(path, allow_pickle=True)
            feats = data["frame_features"]  # (N, F)
            labels = data.get("frame_labels", None)

            if labels is None or not include_labels:
                labels = np.full(feats.shape[0], -1, dtype=np.int64)

            self.videos.append(
                {
                    "features": feats,
                    "labels": labels,
                    "path": path,
                }
            )

        # Build sliding-window index with LOW subsampling
        self._build_samples()

        print(
            f"DangerVideoDataset: {len(self.samples)} windows from "
            f"{len(self.videos)} videos (low_stride={self.low_stride})"
        )

    def _build_samples(self) -> None:
        T = self.temporal_window

        for vid_idx, vid in enumerate(self.videos):
            labels = vid["labels"]
            N = labels.shape[0]

            low_counter = 0

            for t in range(T - 1, N):
                window_labels = labels[t - T + 1 : t + 1]

                # require all frames in window to have valid labels
                if np.any(window_labels < 0):
                    continue

                y_t = int(labels[t])
                if y_t < 0:
                    continue

                if y_t == 0:
                    # subsample LOW windows
                    if (low_counter % self.low_stride) != 0:
                        low_counter += 1
                        continue
                    low_counter += 1

                self.samples.append((vid_idx, t))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        vid_idx, t = self.samples[idx]
        vid = self.videos[vid_idx]

        feats = vid["features"]
        labels = vid["labels"]
        T = self.temporal_window

        window_feats = feats[t - T + 1 : t + 1]  # (T, F)
        y = int(labels[t])

        X = torch.from_numpy(window_feats).float()  # (T, F)
        y_tensor = torch.tensor(y, dtype=torch.long)

        return X, y_tensor
