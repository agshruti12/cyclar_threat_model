import numpy as np
from typing import Dict, Any, Tuple, List
from .model_types import BBox

def bbox_area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

def bbox_center(b: BBox) -> Tuple[float, float]:
    cx = (b[0] + b[2]) / 2.0
    cy = (b[1] + b[3]) / 2.0
    return cx, cy

def compute_track_features(track: Dict) -> Dict:
    """
    track: dict with keys track_id, cls_id, frame_indices, bboxes
    Returns dict with same metadata plus per-frame features.
    """
    bboxes: List[BBox] = [tuple(b) for b in track["bboxes"]]
    frame_indices: List[int] = track["frame_indices"]

    areas: List[float] = []
    centers: List[Tuple[float, float]] = []
    delta_areas: List[float] = []
    delta_cx: List[float] = []
    delta_cy: List[float] = []

    prev_area = None
    prev_cx = None
    prev_cy = None

    for b in bboxes:
        a = bbox_area(b)
        cx, cy = bbox_center(b)
        areas.append(a)
        centers.append((cx, cy))

        if prev_area is None:
            delta_areas.append(0.0)
            delta_cx.append(0.0)
            delta_cy.append(0.0)
        else:
            delta_areas.append(a - prev_area)
            delta_cx.append(cx - prev_cx)
            delta_cy.append(cy - prev_cy)

        prev_area = a
        prev_cx = cx
        prev_cy = cy

    # you can add more (normalized center, speed magnitude, etc.)
    return {
        "track_id": track["track_id"],
        "cls_id": track["cls_id"],
        "frame_indices": frame_indices,
        "bboxes": track["bboxes"],
        "areas": areas,
        "centers": centers,
        "delta_areas": delta_areas,
        "delta_cx": delta_cx,
        "delta_cy": delta_cy,
    }


def compute_features_from_bbox_and_depth(
    bbox: BBox,
    depth_map: np.ndarray,
    prev_state: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    """
    Compute per-object features from bbox + depth.

    For now, this is a stub with fake / placeholder values.
    Later you'll implement:
      - distance (median depth within bbox)
      - delta distance
      - bbox area, delta area
      - approximate TTC
      - normalized center coords
    """
    x1, y1, x2, y2 = map(int, bbox)
    # Placeholder: you don't have depth yet
    # Later you'll use depth_map[y1:y2, x1:x2]
    bbox_area = (x2 - x1) * (y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    features = {
        "bbox_area": float(bbox_area),
        "cx_norm": float(cx) / depth_map.shape[1],
        "cy_norm": float(cy) / depth_map.shape[0],
        # TODO: distance, d_dist, TTC once depth is integrated
        "distance": -1.0,  # placeholder
        "delta_distance": 0.0,
        "ttc": 999.0,
    }

    return features
