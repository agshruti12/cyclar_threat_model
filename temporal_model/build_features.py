from __future__ import annotations

import csv
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ObjectDetection:
    run_id: str
    video_id: str
    frame_idx: int
    track_id: int
    cls_id: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def aspect_ratio(self) -> float:
        return self.w / self.h if self.h > 1e-6 else 0.0

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


@dataclass
class TrackState:
    queue: Deque[ObjectDetection]
    last_seen_frame: int = -1
    prev_cx: Optional[float] = None
    prev_cy: Optional[float] = None
    prev_area: Optional[float] = None
    prev_h: Optional[float] = None
    ema_dx: float = 0.0
    ema_dy: float = 0.0
    ema_area_growth: float = 0.0
    ema_h_growth: float = 0.0
    mean_conf_ema: float = 0.0
    initialized: bool = False


@dataclass
class QueueConfig:
    queue_len: int = 24
    sample_every: int = 6
    ema_alpha: float = 0.30
    stale_after_frames: int = 30


class ObjectDetectionQueueManager:
    """Maintains per-object queues and recursive state for inference/training."""

    def __init__(self, config: Optional[QueueConfig] = None):
        self.config = config or QueueConfig()
        self._tracks: Dict[Tuple[str, str, int], TrackState] = {}

    def _key(self, det: ObjectDetection) -> Tuple[str, str, int]:
        return (det.run_id, det.video_id, det.track_id)

    def expire_stale(self, current_frame_idx: int) -> None:
        to_delete: List[Tuple[str, str, int]] = []
        for key, state in self._tracks.items():
            if current_frame_idx - state.last_seen_frame > self.config.stale_after_frames:
                to_delete.append(key)
        for key in to_delete:
            del self._tracks[key]

    def update_detection(self, det: ObjectDetection) -> TrackState:
        key = self._key(det)
        state = self._tracks.get(key)
        if state is None:
            state = TrackState(queue=deque(maxlen=self.config.queue_len))
            self._tracks[key] = state

        if state.initialized:
            dx = det.cx - (state.prev_cx if state.prev_cx is not None else det.cx)
            dy = det.cy - (state.prev_cy if state.prev_cy is not None else det.cy)
            area_growth = 0.0
            h_growth = 0.0
            if state.prev_area is not None and state.prev_area > 1e-6:
                area_growth = (det.area - state.prev_area) / state.prev_area
            if state.prev_h is not None and state.prev_h > 1e-6:
                h_growth = (det.h - state.prev_h) / state.prev_h

            a = self.config.ema_alpha
            state.ema_dx = a * dx + (1.0 - a) * state.ema_dx
            state.ema_dy = a * dy + (1.0 - a) * state.ema_dy
            state.ema_area_growth = a * area_growth + (1.0 - a) * state.ema_area_growth
            state.ema_h_growth = a * h_growth + (1.0 - a) * state.ema_h_growth
            state.mean_conf_ema = a * det.conf + (1.0 - a) * state.mean_conf_ema
        else:
            state.mean_conf_ema = det.conf
            state.initialized = True

        state.queue.append(det)
        state.prev_cx = det.cx
        state.prev_cy = det.cy
        state.prev_area = det.area
        state.prev_h = det.h
        state.last_seen_frame = det.frame_idx
        return state

    def get_state(self, run_id: str, video_id: str, track_id: int) -> Optional[TrackState]:
        return self._tracks.get((run_id, video_id, track_id))


FEATURE_NAMES = [
    "cx_norm",
    "cy_norm",
    "h_norm",
    "area_norm",
    "aspect_ratio",
    "conf",
    "queue_len",
    "area_growth_k",
    "h_growth_k",
    "dx_k",
    "dy_k",
    "ema_dx",
    "ema_dy",
    "ema_area_growth",
    "ema_h_growth",
    "mean_conf_ema",
    "dist_to_center",
    "slope_h",
    "slope_cy",
    "slope_dist_to_center",
    "frac_positive_h_growth",
    "frac_toward_center",
    "std_cx",
    "std_delta_h",
]


def _safe_slope(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    x = np.arange(values.size, dtype=np.float32)
    x_mean = float(x.mean())
    y_mean = float(values.mean())
    denom = float(((x - x_mean) ** 2).sum())
    if denom <= 1e-8:
        return 0.0
    numer = float(((x - x_mean) * (values - y_mean)).sum())
    return numer / denom


def compute_object_features(
    state: TrackState,
    frame_w: int,
    frame_h: int,
    config: Optional[QueueConfig] = None,
) -> Dict[str, float]:
    """
    Compute object features for a single active track state.
    Intended for inference-time use after the queue manager has been updated.
    """
    cfg = config or QueueConfig()
    if not state.queue:
        raise ValueError("TrackState queue is empty.")

    q = list(state.queue)
    first = q[0]
    last = q[-1]
    eps = 1e-6

    cx_norm = last.cx / max(frame_w, 1)
    cy_norm = last.cy / max(frame_h, 1)
    h_norm = last.h / max(frame_h, 1)
    area_norm = last.area / max(frame_w * frame_h, 1)
    aspect_ratio = last.aspect_ratio
    conf = last.conf
    queue_len = float(len(q))

    area_growth_k = 0.0
    h_growth_k = 0.0
    if first.area > eps:
        area_growth_k = (last.area - first.area) / first.area
    if first.h > eps:
        h_growth_k = (last.h - first.h) / first.h

    dx_k = last.cx - first.cx
    dy_k = last.cy - first.cy

    dist_to_center = abs(cx_norm - 0.5)

    heights = np.array([d.h for d in q], dtype=np.float32)
    cys = np.array([d.cy for d in q], dtype=np.float32)
    cxs = np.array([d.cx for d in q], dtype=np.float32)
    dists = np.array([abs((d.cx / max(frame_w, 1)) - 0.5) for d in q], dtype=np.float32)

    slope_h = _safe_slope(heights)
    slope_cy = _safe_slope(cys)
    slope_dist_to_center = _safe_slope(dists)

    if len(q) >= 2:
        delta_h = np.diff(heights)
        delta_dist = np.diff(dists)
        frac_positive_h_growth = float((delta_h > 0).mean())
        frac_toward_center = float((delta_dist < 0).mean())
        std_delta_h = float(delta_h.std(ddof=0))
    else:
        frac_positive_h_growth = 0.0
        frac_toward_center = 0.0
        std_delta_h = 0.0

    std_cx = float(cxs.std(ddof=0)) if len(q) >= 2 else 0.0

    return {
        "cx_norm": float(cx_norm),
        "cy_norm": float(cy_norm),
        "h_norm": float(h_norm),
        "area_norm": float(area_norm),
        "aspect_ratio": float(aspect_ratio),
        "conf": float(conf),
        "queue_len": float(queue_len),
        "area_growth_k": float(area_growth_k),
        "h_growth_k": float(h_growth_k),
        "dx_k": float(dx_k),
        "dy_k": float(dy_k),
        "ema_dx": float(state.ema_dx),
        "ema_dy": float(state.ema_dy),
        "ema_area_growth": float(state.ema_area_growth),
        "ema_h_growth": float(state.ema_h_growth),
        "mean_conf_ema": float(state.mean_conf_ema),
        "dist_to_center": float(dist_to_center),
        "slope_h": float(slope_h),
        "slope_cy": float(slope_cy),
        "slope_dist_to_center": float(slope_dist_to_center),
        "frac_positive_h_growth": float(frac_positive_h_growth),
        "frac_toward_center": float(frac_toward_center),
        "std_cx": float(std_cx),
        "std_delta_h": float(std_delta_h),
    }


_DETECTIONS_REQUIRED_COLS = {
    "run_id", "video_id", "frame_idx", "track_id", "cls_id", "conf", "x1", "y1", "x2", "y2"
}

_LABELS_REQUIRED_COLS = {"run_id", "video_id", "frame_idx", "track_id", "label"}


def _read_detections_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = _DETECTIONS_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Detections CSV is missing columns: {sorted(missing)}")
    return df.sort_values(["run_id", "video_id", "frame_idx", "track_id"]).reset_index(drop=True)


def _read_labels_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = _LABELS_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Labels CSV is missing columns: {sorted(missing)}")
    return df.sort_values(["run_id", "video_id", "frame_idx", "track_id"]).reset_index(drop=True)


def compute_object_features_from_csv(
    detections_csv: str | Path,
    labels_csv: Optional[str | Path] = None,
    frame_w: int = 1920,
    frame_h: int = 1080,
    config: Optional[QueueConfig] = None,
) -> pd.DataFrame:
    """
    Training-time feature builder.

    If labels_csv is provided, returns only rows that have labels.
    Otherwise returns features for all active objects on sampled frames.
    """
    cfg = config or QueueConfig()
    det_df = _read_detections_csv(detections_csv)

    labels_lookup: Optional[Dict[Tuple[str, str, int, int], int]] = None
    if labels_csv is not None:
        labels_df = _read_labels_csv(labels_csv)
        labels_lookup = {
            (str(r.run_id), str(r.video_id), int(r.frame_idx), int(r.track_id)): int(r.label)
            for r in labels_df.itertuples(index=False)
        }

    manager = ObjectDetectionQueueManager(cfg)
    rows: List[Dict[str, float]] = []

    current_group: Optional[Tuple[str, str, int]] = None
    frame_dets: List[ObjectDetection] = []

    def flush_frame(group_key: Optional[Tuple[str, str, int]], grouped_dets: List[ObjectDetection]) -> None:
        if group_key is None:
            return
        run_id, video_id, frame_idx = group_key
        manager.expire_stale(frame_idx)

        # Update queues with every detection on this frame.
        for det in grouped_dets:
            manager.update_detection(det)

        sampled_frame = (frame_idx % cfg.sample_every == 0)
        if not sampled_frame and labels_lookup is None:
            return

        # Emit features only for detections active on this frame.
        for det in grouped_dets:
            key = (run_id, video_id, frame_idx, det.track_id)
            if labels_lookup is not None and key not in labels_lookup:
                continue

            state = manager.get_state(run_id, video_id, det.track_id)
            if state is None:
                continue

            feats = compute_object_features(state, frame_w=frame_w, frame_h=frame_h, config=cfg)
            row: Dict[str, float] = {
                "run_id": run_id,
                "video_id": video_id,
                "frame_idx": float(frame_idx),
                "track_id": float(det.track_id),
                "cls_id": float(det.cls_id),
            }
            row.update(feats)
            if labels_lookup is not None:
                row["label"] = float(labels_lookup[key])
            rows.append(row)

    for record in det_df.itertuples(index=False):
        det = ObjectDetection(
            run_id=str(record.run_id),
            video_id=str(record.video_id),
            frame_idx=int(record.frame_idx),
            track_id=int(record.track_id),
            cls_id=int(record.cls_id),
            conf=float(record.conf),
            x1=float(record.x1),
            y1=float(record.y1),
            x2=float(record.x2),
            y2=float(record.y2),
        )
        key = (det.run_id, det.video_id, det.frame_idx)
        if current_group is None:
            current_group = key
        if key != current_group:
            flush_frame(current_group, frame_dets)
            current_group = key
            frame_dets = []
        frame_dets.append(det)

    flush_frame(current_group, frame_dets)
    return pd.DataFrame(rows)


def save_features_dataframe(df: pd.DataFrame, out_csv: str | Path) -> None:
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
