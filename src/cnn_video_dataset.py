# cnn_video_dataset.py

import json
import os
from typing import List, Dict, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_label_file(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def build_frame_indices_for_segment(
    start_frame: int,
    end_frame: int,
    frames_per_segment: int,
) -> np.ndarray:
    """Evenly sample `frames_per_segment` indices between start and end frame."""
    if frames_per_segment <= 1 or end_frame <= start_frame:
        return np.array([start_frame], dtype=int)

    return np.linspace(start_frame, end_frame, num=frames_per_segment).astype(int)


def get_train_transform(img_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.ToPILImage(),

            # --- Spatial augmentations ---
            T.RandomResizedCrop(
                img_size,
                scale=(0.90, 1.0),        # gentle crop, preserves most of scene
                ratio=(0.95, 1.05),       # slight aspect wiggle for camera jitter
            ),

            # Flip horizontally with low probability.
            # Left/right can matter for danger detection, so keep p small.
            T.RandomHorizontalFlip(p=0.25),

            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),

            T.RandomApply(
                [
                    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                ],
                p=0.3,
            ),

            # Mild random rotations / affine to simulate bike movement
            T.RandomApply(
                [
                    T.RandomRotation(degrees=3),
                ],
                p=0.25,
            ),

            # Mild random affine transformations
            T.RandomApply(
                [
                    T.RandomAffine(
                        degrees=0,
                        translate=(0.02, 0.02),   # small camera jiggle
                        scale=(0.98, 1.02),
                        shear=2,
                    )
                ],
                p=0.25
            ),

            # Convert to tensor
            T.ToTensor(),

            # Normalize for ResNet
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )



def get_eval_transform(img_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class SegmentVideoDataset(Dataset):
    """
    Dataset that reads label JSONs (scene-level segments) and loads raw frames
    from the corresponding videos.

    Each sample is:
      X: (T, C, H, W)  -> sequence of T frames
      y: scalar int64  -> 0 (LOW), 1 (MED), 2 (HIGH)
    """

    def __init__(
        self,
        label_json_paths: List[str],
        frames_per_segment: int = 8,
        img_size: int = 224,
        train: bool = True,
        drop_unlabeled: bool = True,
    ):
        """
        Args:
            label_json_paths: list of scene-label json files.
            frames_per_segment: number of frames sampled per segment.
            img_size: CNN input resolution.
            train: if True, uses train augmentations; else eval.
            drop_unlabeled: if True, ignores segments with label None.
        """
        exclude_videos = ["data/labels/sample_bike_ride_scene_labels.json"]
        # self.label_json_paths = label_json_paths

        self.label_json_paths = []
        
        # Filter out JSON files that reference the excluded video paths
        for lbl_path in label_json_paths:
            with open(lbl_path, "r") as f:
                data = json.load(f)
                video_path = data["video_path"]

                if video_path not in exclude_videos:
                    self.label_json_paths.append(lbl_path)
                else:
                    print(f"Skipping video: {video_path}")  # Optionally log the skipped videos

        self.frames_per_segment = frames_per_segment
        self.train = train

        self.transform = (
            get_train_transform(img_size) if train else get_eval_transform(img_size)
        )

        # List of samples: each is a dict with video_path, frame_indices, label
        self.samples: List[Dict] = []

        # Cache video captures to avoid reopening constantly
        self._cap_cache: Dict[str, cv2.VideoCapture] = {}

        self._build_samples(drop_unlabeled=drop_unlabeled)

        print(
            f"SegmentVideoDataset(train={train}): {len(self.samples)} segments from "
            f"{len(self.label_json_paths)} label files"
        )

    def _build_samples(self, drop_unlabeled: bool = True):
        # temp exclude sample_bike_ride
        exclude_videos = ["data/raw/sample_bike_ride.mp4"]

        for lbl_path in self.label_json_paths:
            data = load_label_file(lbl_path)
            video_path = data["video_path"]
            segments = data["segments"]

            # Skip if the video is in the excluded list
            if video_path in exclude_videos:
                print(f"Skipping video: {video_path}")
                continue

            for seg in segments:
                label = seg["label"]
                if drop_unlabeled and (label is None):
                    continue

                # If you collapse to binary later, you can map label here.
                start_f = int(seg["start_frame"])
                end_f = int(seg["end_frame"])

                frame_idxs = build_frame_indices_for_segment(
                    start_frame=start_f,
                    end_frame=end_f,
                    frames_per_segment=self.frames_per_segment,
                )

                self.samples.append(
                    {
                        "video_path": video_path,
                        "frame_indices": frame_idxs,
                        "label": int(label) if label is not None else -1,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def _get_cap(self, video_path: str) -> cv2.VideoCapture:
        if video_path not in self._cap_cache:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video not found: {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")
            self._cap_cache[video_path] = cap
        return self._cap_cache[video_path]

    def _read_frame(self, cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            # If read fails, just reuse the last available frame or a black image
            # Here we return a black image with 3 channels  (we'll resize later).
            # Alternatively, you could raise an error.
            return np.zeros((224, 224, 3), dtype=np.uint8)

        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        video_path = sample["video_path"]
        frame_indices = sample["frame_indices"]
        label = sample["label"]

        cap = self._get_cap(video_path)

        frames = []
        for f_idx in frame_indices:
            img = self._read_frame(cap, int(f_idx))
            img_t = self.transform(img)  # (C, H, W)
            frames.append(img_t)

        # stack into (T, C, H, W)
        X = torch.stack(frames, dim=0)
        y = torch.tensor(label, dtype=torch.long)

        return X, y

    def close(self):
        for cap in self._cap_cache.values():
            cap.release()
        self._cap_cache.clear()
