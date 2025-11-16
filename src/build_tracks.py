import json
from typing import List, Dict, Tuple
from src.model_types import Track, BBox
from dataclasses import asdict
import os
from features import compute_track_features


BBox = Tuple[float, float, float, float]

def iou(b1: BBox, b2: BBox) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    if inter == 0:
        return 0.0

    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


IOU_THRESHOLD = 0.3
MAX_MISSES = 10  # frames to keep track alive without match

def load_detections(path: str) -> List[List[Dict]]:
    with open(path, "r") as f:
        return json.load(f)  # list[frame] -> list[det dicts]

def build_tracks(detections_per_frame: List[List[Dict]]) -> List[Track]:
    tracks: List[Track] = []
    active_track_last_bbox: Dict[int, BBox] = {}
    active_track_last_frame: Dict[int, int] = {}
    next_track_id = 0

    for frame_idx, frame_dets in enumerate(detections_per_frame):
        # 1. expire old tracks
        to_deactivate = [
            tid for tid, last_frame in active_track_last_frame.items()
            if frame_idx - last_frame > MAX_MISSES
        ]
        for tid in to_deactivate:
            active_track_last_bbox.pop(tid, None)
            active_track_last_frame.pop(tid, None)

        # 2. mark which tracks are already used this frame
        used_tracks = set()

        # 3. loop over detections in this frame
        for det in frame_dets:
            cls_id = det["cls_id"]
            bbox = tuple(det["bbox"])  # (x1, y1, x2, y2)

            # find best matching active track with same class
            best_iou = 0.0
            best_tid = None
            for tid, tbbox in active_track_last_bbox.items():
                # enforce same class
                if tracks[tid].cls_id != cls_id:
                    continue
                if tid in used_tracks:
                    continue
                iou_val = iou(tbbox, bbox)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_tid = tid

            if best_tid is not None and best_iou >= IOU_THRESHOLD:
                # append to existing track
                tr = tracks[best_tid]
                tr.frame_indices.append(frame_idx)
                tr.bboxes.append(bbox)
                active_track_last_bbox[best_tid] = bbox
                active_track_last_frame[best_tid] = frame_idx
                used_tracks.add(best_tid)
            else:
                # create new track
                tid = next_track_id
                next_track_id += 1
                t = Track(track_id=tid, cls_id=cls_id)
                t.frame_indices.append(frame_idx)
                t.bboxes.append(bbox)
                tracks.append(t)
                active_track_last_bbox[tid] = bbox
                active_track_last_frame[tid] = frame_idx
                used_tracks.add(tid)

    return tracks


def save_tracks(tracks: List[Track], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serializable = []
    for t in tracks:
        serializable.append({
            "track_id": t.track_id,
            "cls_id": t.cls_id,
            "frame_indices": t.frame_indices,
            "bboxes": [list(b) for b in t.bboxes],
        })
    with open(path, "w") as f:
        json.dump(serializable, f)

if __name__ == "__main__":
    dets = load_detections("data/processed/sample_bike_ride_dets.json")
    tracks = build_tracks(dets)
    print(f"Built {len(tracks)} tracks")
    save_tracks(tracks, "data/processed/sample_bike_ride_tracks.json")
    print("Saved tracks to data/processed/sample_bike_ride_tracks.json")