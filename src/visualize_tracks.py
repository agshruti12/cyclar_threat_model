import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Paths – adjust if needed
VIDEO_PATH = "data/raw/sample_bike_ride.mp4"
TRACKS_PATH = "data/processed/sample_bike_ride_tracks.json"

BBox = Tuple[float, float, float, float]


def load_tracks(path: str) -> List[dict]:
    with open(path, "r") as f:
        return json.load(f)


def build_frame_to_tracks_index(tracks: List[dict]) -> Dict[int, List[Tuple[int, int]]]:
    """
    Build an index:
      frame_idx -> list of (track_id, index_in_track_bboxes)

    So for a given frame, we know which tracks appear and which bbox to use.
    """
    frame_to_tracks: Dict[int, List[Tuple[int, int]]] = {}
    for t in tracks:
        tid = t["track_id"]
        frame_indices = t["frame_indices"]
        for i, fidx in enumerate(frame_indices):
            frame_to_tracks.setdefault(fidx, []).append((tid, i))
    return frame_to_tracks


def get_color_for_track(track_id: int) -> Tuple[int, int, int]:
    """
    Simple deterministic color by track id.
    """
    # Some distinct-ish colors
    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 255),
        (255, 128, 0),
    ]
    return palette[track_id % len(palette)]


def visualize_tracks(
    video_path: str,
    tracks_path: str,
    max_frames: int = -1,
    show_trajectories: bool = True,
) -> None:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.exists(tracks_path):
        raise FileNotFoundError(f"Tracks JSON not found: {tracks_path}")

    tracks = load_tracks(tracks_path)
    frame_to_tracks = build_frame_to_tracks_index(tracks)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_idx = 0
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames > 0 and frame_idx >= max_frames:
            break

        # Draw tracks visible in this frame
        if frame_idx in frame_to_tracks:
            for (tid, i_in_track) in frame_to_tracks[frame_idx]:
                t = tracks[tid]
                bbox = t["bboxes"][i_in_track]  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, bbox)
                color = get_color_for_track(tid)

                # Draw bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Label with track id
                label = f"ID {tid}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

                if show_trajectories:
                    # Draw trajectory of this track up to current frame
                    pts = []
                    for j in range(0, i_in_track + 1):
                        bx = t["bboxes"][j]
                        cx = int((bx[0] + bx[2]) / 2.0)
                        cy = int((bx[1] + bx[3]) / 2.0)
                        pts.append((cx, cy))
                    for p1, p2 in zip(pts[:-1], pts[1:]):
                        cv2.line(frame, p1, p2, color, 2)

        cv2.imshow("Tracks", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize_tracks(VIDEO_PATH, TRACKS_PATH, max_frames=-1, show_trajectories=True)
