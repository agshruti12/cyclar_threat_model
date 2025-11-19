# src/heuristic_risk.py

import json
import os
from typing import List, Dict

FEATURES_JSON_PATH = "data/processed/sample_bike_ride_tracks_features.json"
RISK_JSON_PATH = "data/processed/sample_bike_ride_tracks_risk.json"

def load_features(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


def normalize_list(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax - vmin < 1e-6:
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def compute_danger_for_track(track_feat: Dict) -> Dict:
    """
    Adds a 'danger_scores' list aligned with frame_indices.
    Danger is based on:
      - large normalized area
      - area increasing (approach)
      - object close to image center horizontally
    """
    areas = track_feat["areas"]
    centers_norm = track_feat["centers_norm"]  # (cx_norm, cy_norm)
    frame_indices = track_feat["frame_indices"]

    # normalize areas to [0,1] across this track
    areas_norm = normalize_list(areas)

    # compute area deltas
    delta_area = [0.0]
    for i in range(1, len(areas)):
        delta_area.append(areas[i] - areas[i - 1])
    # normalize positive deltas
    delta_area_pos = [max(0.0, da) for da in delta_area]
    delta_area_norm = normalize_list(delta_area_pos)

    danger_scores: List[float] = []

    for i in range(len(frame_indices)):
        area_score = areas_norm[i]
        approach_score = delta_area_norm[i]

        cx_norm, _ = centers_norm[i]
        # how close to horizontal center are we?
        # center at 0.5 is ideal for "in lane"; farther away is less dangerous.
        horiz_offset = abs(cx_norm - 0.5)
        center_score = 1.0 - min(horiz_offset / 0.5, 1.0)  # 1 when at center, 0 at edges

        # combine heuristics:
        # weighted sum: more weight to area + center
        danger = (
            0.5 * area_score +
            0.3 * center_score +
            0.2 * approach_score
        )

        # clamp
        danger = max(0.0, min(1.0, danger))
        danger_scores.append(danger)

    track_feat_with_risk = dict(track_feat)
    track_feat_with_risk["danger_scores"] = danger_scores
    return track_feat_with_risk


def save_risk(tracks_with_risk: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(tracks_with_risk, f)


if __name__ == "__main__":
    if not os.path.exists(FEATURES_JSON_PATH):
        raise FileNotFoundError(f"Features JSON not found: {FEATURES_JSON_PATH}")

    feats = load_features(FEATURES_JSON_PATH)
    tracks_with_risk = [compute_danger_for_track(t) for t in feats]

    save_risk(tracks_with_risk, RISK_JSON_PATH)
    print(f"Saved risk scores for {len(tracks_with_risk)} tracks to {RISK_JSON_PATH}")
