import json
import os
from typing import Dict, List
from pathlib import Path

import numpy as np


def load_tracks_features(path: str) -> List[Dict]:
    """Load list of track dicts from *_tracks_features.json."""
    with open(path, "r") as f:
        return json.load(f)


def load_scene_labels(path: str) -> Dict:
    """Load scene-level labels JSON (with segments)."""
    with open(path, "r") as f:
        return json.load(f)


def build_frame_labels(scene_labels: Dict, frame_count: int) -> np.ndarray:
    """
    Build frame-level labels array of size (frame_count,).
    frame_labels[f] ∈ {0,1,2} or -1 if unlabeled.
    """
    frame_labels = np.full(frame_count, -1, dtype=np.int64)

    for seg in scene_labels["segments"]:
        label = seg["label"]
        if label is None:
            continue
        start_f = int(seg["start_frame"])
        end_f = int(seg["end_frame"])
        start_f = max(0, start_f)
        end_f = min(frame_count - 1, end_f)

        frame_labels[start_f : end_f + 1] = int(label)

    return frame_labels


def build_frame_features(
    tracks_features: List[Dict],
    frame_count: int,
) -> np.ndarray:
    """
    Aggregate per-track features into per-frame features.

    We will produce a (frame_count, F) array where each row is:

      [closest_area,
       closest_cx_norm,
       closest_cy_norm,
       closest_speed,
       num_tracks,
       max_speed]

    using your *_tracks_features.json structure:

      {
        "track_id": ...,
        "cls_id": ...,
        "frame_indices": [...],
        "areas": [...],
        "centers_norm": [[cx_norm, cy_norm], ...],
        "speed": [...]
      }
    """
    F = 6
    frame_feats = np.zeros((frame_count, F), dtype=np.float32)

    # We will keep track, per frame, of the closest track (largest area)
    # Also the number of tracks and max speed for context.
    closest_area = np.zeros(frame_count, dtype=np.float32)
    closest_cx = np.zeros(frame_count, dtype=np.float32)
    closest_cy = np.zeros(frame_count, dtype=np.float32)
    closest_speed = np.zeros(frame_count, dtype=np.float32)
    num_tracks = np.zeros(frame_count, dtype=np.float32)
    max_speed = np.zeros(frame_count, dtype=np.float32)

    for track in tracks_features:
        frame_idxs = track["frame_indices"]
        areas = track["areas"]
        centers_norm = track["centers_norm"]
        speeds = track["speed"]

        for idx_in_track, f_idx in enumerate(frame_idxs):
            if f_idx < 0 or f_idx >= frame_count:
                continue

            area = float(areas[idx_in_track])
            cx_norm, cy_norm = centers_norm[idx_in_track]
            speed = float(speeds[idx_in_track])

            # Update num_tracks and max_speed
            num_tracks[f_idx] += 1.0
            if abs(speed) > max_speed[f_idx]:
                max_speed[f_idx] = abs(speed)

            # Update closest vehicle (largest area)
            if area > closest_area[f_idx]:
                closest_area[f_idx] = area
                closest_cx[f_idx] = float(cx_norm)
                closest_cy[f_idx] = float(cy_norm)
                closest_speed[f_idx] = speed

    # Normalize area by a constant if you want (optional).
    # For now, we just keep raw area; model can still learn from it.
    frame_feats[:, 0] = closest_area
    frame_feats[:, 1] = closest_cx
    frame_feats[:, 2] = closest_cy
    frame_feats[:, 3] = closest_speed
    frame_feats[:, 4] = num_tracks
    frame_feats[:, 5] = max_speed

    return frame_feats


def process_video(
    tracks_features_path: str,
    scene_labels_path: str,
    out_npz_path: str,
):
    print(f"Processing video:")
    print(f"  tracks_features: {tracks_features_path}")
    print(f"  scene_labels   : {scene_labels_path}")

    tracks_features = load_tracks_features(tracks_features_path)
    scene_labels = load_scene_labels(scene_labels_path)

    frame_count = int(scene_labels["frame_count"])
    fps = float(scene_labels["fps"])
    video_path = scene_labels["video_path"]

    frame_features = build_frame_features(tracks_features, frame_count)
    frame_labels = build_frame_labels(scene_labels, frame_count)

    os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)
    np.savez_compressed(
        out_npz_path,
        frame_features=frame_features,
        frame_labels=frame_labels,
        fps=fps,
        frame_count=frame_count,
        video_path=video_path,
    )
    print(f"  Saved per-frame data to {out_npz_path}")


if __name__ == "__main__":

    raw_dir = Path("data/raw")
    tracks_dir = Path("data/processed")
    labels_dir = Path("data/labels")
    out_dir = Path("data/training")

    out_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(raw_dir.glob("*.mp4"))

    if len(video_files) == 0:
        raise RuntimeError("No .mp4 files found in data/raw")

    print(f"Found {len(video_files)} raw videos")

    for video_path in video_files:
        video_name = video_path.stem  # e.g. "my_video"

        tracks_features_path = tracks_dir / f"{video_name}_tracks_features.json"
        scene_labels_path = labels_dir / f"{video_name}_scene_labels.json"
        out_npz_path = out_dir / f"{video_name}_frame_data.npz"

        if not tracks_features_path.exists():
            print(f"[SKIP] Missing tracks features: {tracks_features_path}")
            continue

        if not scene_labels_path.exists():
            print(f"[SKIP] Missing scene labels: {scene_labels_path}")
            continue

        print(f"Processing video: {video_name}")
        print(f"  Tracks: {tracks_features_path}")
        print(f"  Labels: {scene_labels_path}")
        print(f"  Output: {out_npz_path}")

        process_video(
            str(tracks_features_path),
            str(scene_labels_path),
            str(out_npz_path)
        )

    print("Done processing all videos.")