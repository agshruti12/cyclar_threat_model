# src/build_track_features.py

import json
import os
from typing import List, Dict
import cv2
from src.features import compute_track_features

TRACKS_JSON_PATH = "data/processed/sample_bike_ride_tracks.json"
FEATURES_JSON_PATH = "data/processed/sample_bike_ride_tracks_features.json"
VIDEO_PATH = "data/raw/sample_bike_ride.mp4"


def load_tracks(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


def get_video_size(video_path: str) -> (int, int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def save_features(features_list: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(features_list, f)


if __name__ == "__main__":
    if not os.path.exists(TRACKS_JSON_PATH):
        raise FileNotFoundError(f"Tracks JSON not found: {TRACKS_JSON_PATH}")

    tracks = load_tracks(TRACKS_JSON_PATH)
    frame_width, frame_height = get_video_size(VIDEO_PATH)
    print(f"Video size: {frame_width}x{frame_height}")

    features_per_track: List[Dict] = []
    for t in tracks:
        ft = compute_track_features(t, frame_width, frame_height)
        features_per_track.append(ft)

    save_features(features_per_track, FEATURES_JSON_PATH)
    print(f"Saved features for {len(features_per_track)} tracks to {FEATURES_JSON_PATH}")