#!/usr/bin/env python3
"""
Dense track annotator.

This tool does two things:
1. Runs YOLO tracking on EVERY frame and saves all tracked detections
2. Creates sparse labeling tasks every Nth frame for active objects

Outputs:
- tracked_detections.csv  (dense, every frame)
- track_labels.csv        (sparse, only sampled-frame tasks that get labeled)

Controls:
    0 = low danger
    1 = medium danger
    2 = high danger
    p = play clip around sampled frame
    s = skip
    b = back
    q = quit

Filtering:
- area threshold unit: pixels^2
- height threshold unit: pixels
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import uuid
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import pandas as pd
from ultralytics import YOLO


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Detection:
    frame_idx: int
    track_id: int
    cls_id: int
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class Task:
    run_id: str
    video_id: str
    frame_idx: int
    track_id: int


# -----------------------------
# File helpers
# -----------------------------

def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_existing_labels(csv_path: str) -> Dict[Tuple[str, str, int, int], int]:
    if not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path)
    required_cols = {"run_id", "video_id", "frame_idx", "track_id", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"{csv_path} exists but is missing required columns: {sorted(required_cols)}"
        )

    labels: Dict[Tuple[str, str, int, int], int] = {}
    for _, row in df.iterrows():
        key = (
            str(row["run_id"]),
            str(row["video_id"]),
            int(row["frame_idx"]),
            int(row["track_id"]),
        )
        labels[key] = int(row["label"])
    return labels


def rewrite_labels_csv(csv_path: str, labels: Dict[Tuple[str, str, int, int], int]) -> None:
    ensure_parent_dir(csv_path)
    rows = [
        [run_id, video_id, frame_idx, track_id, label]
        for (run_id, video_id, frame_idx, track_id), label in labels.items()
    ]
    rows.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "video_id", "frame_idx", "track_id", "label"])
        writer.writerows(rows)


def save_tracked_detections_csv(
    csv_path: str,
    run_id: str,
    video_id: str,
    detections_by_frame: Dict[int, List[Detection]],
) -> None:
    ensure_parent_dir(csv_path)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id", "video_id", "frame_idx", "track_id",
            "cls_id", "conf", "x1", "y1", "x2", "y2"
        ])

        for frame_idx in sorted(detections_by_frame.keys()):
            for det in detections_by_frame[frame_idx]:
                writer.writerow([
                    run_id,
                    video_id,
                    det.frame_idx,
                    det.track_id,
                    det.cls_id,
                    det.conf,
                    det.x1,
                    det.y1,
                    det.x2,
                    det.y2,
                ])


# -----------------------------
# Video/model utilities
# -----------------------------

def class_name_from_model(model: YOLO, cls_id: int) -> str:
    try:
        return str(model.names[int(cls_id)])
    except Exception:
        return str(cls_id)


def load_video_frames(video_path: str) -> List:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames found in video: {video_path}")

    return frames


# -----------------------------
# Tracking
# -----------------------------

def run_tracking(
    model: YOLO,
    frames: List,
    imgsz: int,
    conf: float,
    small_box_area_thresh: int,
    small_box_height_thresh: int,
    allowed_classes: Optional[List[int]],
) -> Dict[int, List[Detection]]:
    detections_by_frame: Dict[int, List[Detection]] = {}

    for frame_idx, frame in enumerate(frames):
        result_list = model.track(
            source=frame,
            persist=True,
            verbose=False,
            conf=conf,
            imgsz=imgsz,
        )

        frame_dets: List[Detection] = []

        if not result_list:
            detections_by_frame[frame_idx] = frame_dets
            continue

        result = result_list[0]
        boxes = result.boxes

        if boxes is None or boxes.xyxy is None or boxes.id is None:
            detections_by_frame[frame_idx] = frame_dets
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        for box, tid, score, cls_id in zip(xyxy, ids, confs, clss):
            if allowed_classes is not None and cls_id not in allowed_classes:
                continue

            x1, y1, x2, y2 = [int(v) for v in box.tolist()]

            det = Detection(
                frame_idx=frame_idx,
                track_id=int(tid),
                cls_id=int(cls_id),
                conf=float(score),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )

            # Filter tiny detections before saving / labeling
            if det.area < small_box_area_thresh or det.height < small_box_height_thresh:
                continue

            frame_dets.append(det)

        detections_by_frame[frame_idx] = frame_dets

    return detections_by_frame


# -----------------------------
# Task creation
# -----------------------------

def build_tasks(
    run_id: str,
    video_id: str,
    detections_by_frame: Dict[int, List[Detection]],
    sample_every: int,
) -> List[Task]:
    tasks: List[Task] = []
    if not detections_by_frame:
        return tasks

    max_frame_idx = max(detections_by_frame.keys())

    for frame_idx in range(0, max_frame_idx + 1, sample_every):
        for det in detections_by_frame.get(frame_idx, []):
            tasks.append(Task(
                run_id=run_id,
                video_id=video_id,
                frame_idx=frame_idx,
                track_id=det.track_id,
            ))

    return tasks


# -----------------------------
# Rendering
# -----------------------------

def draw_box(frame, det: Detection, color: Tuple[int, int, int], text: str, thickness: int = 2):
    cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, thickness)
    label_y = max(18, det.y1 - 8)
    cv2.putText(
        frame,
        text,
        (det.x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        color,
        2,
        cv2.LINE_AA,
    )


def render_frame_for_task(
    raw_frame,
    detections_by_frame: Dict[int, List[Detection]],
    frame_idx: int,
    task: Task,
    model: YOLO,
    task_idx: int,
    total_tasks: int,
    clip_radius: int,
    small_box_area_thresh: int,
    small_box_height_thresh: int,
):
    frame = raw_frame.copy()

    for det in detections_by_frame.get(frame_idx, []):
        is_target = det.track_id == task.track_id
        cls_name = class_name_from_model(model, det.cls_id)

        label = (
            f"id={det.track_id} {cls_name} "
            f"conf={det.conf:.2f} "
            f"h={det.height} "
            f"area={det.area}"
        )

        if is_target:
            draw_box(frame, det, (0, 0, 255), "[TARGET] " + label, thickness=3)
        else:
            draw_box(frame, det, (0, 180, 0), label, thickness=2)

    panel_h = 170
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.68, frame, 0.32, 0)

    lines = [
        f"Task {task_idx + 1}/{total_tasks}    Video: {task.video_id}",
        f"Run ID: {task.run_id}",
        f"Sampled frame: {task.frame_idx}    Target track ID: {task.track_id}    Displayed frame: {frame_idx}",
        "Controls: 0=low  1=medium  2=high  p=play clip  s=skip  b=back  q=quit",
        f"Clip window: +/- {clip_radius} frames",
        f"Filter thresholds: area >= {small_box_area_thresh} px^2, height >= {small_box_height_thresh} px",
    ]

    y0 = 28
    dy = 26
    for i, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (12, y0 + i * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if frame_idx == task.frame_idx:
        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1] - 1, frame.shape[0] - 1),
            (0, 255, 255),
            4,
        )

    return frame


def play_clip(
    window_name: str,
    frames: List,
    detections_by_frame: Dict[int, List[Detection]],
    task: Task,
    model: YOLO,
    task_idx: int,
    total_tasks: int,
    clip_radius: int,
    playback_delay_ms: int,
    small_box_area_thresh: int,
    small_box_height_thresh: int,
):
    start_idx = max(0, task.frame_idx - clip_radius)
    end_idx = min(len(frames) - 1, task.frame_idx + clip_radius)

    for display_frame_idx in range(start_idx, end_idx + 1):
        rendered = render_frame_for_task(
            raw_frame=frames[display_frame_idx],
            detections_by_frame=detections_by_frame,
            frame_idx=display_frame_idx,
            task=task,
            model=model,
            task_idx=task_idx,
            total_tasks=total_tasks,
            clip_radius=clip_radius,
            small_box_area_thresh=small_box_area_thresh,
            small_box_height_thresh=small_box_height_thresh,
        )

        cv2.imshow(window_name, rendered)
        key = cv2.waitKey(playback_delay_ms) & 0xFF

        if key in [ord("0"), ord("1"), ord("2"), ord("s"), ord("b"), ord("q")]:
            return key

    return None


# -----------------------------
# Annotation
# -----------------------------

def annotate_tasks(
    frames: List,
    detections_by_frame: Dict[int, List[Detection]],
    tasks: List[Task],
    model: YOLO,
    labels_csv: str,
    clip_radius: int,
    playback_delay_ms: int,
    small_box_area_thresh: int,
    small_box_height_thresh: int,
):
    if not tasks:
        print("No annotation tasks found.")
        return

    existing_labels = load_existing_labels(labels_csv)

    pending_tasks: List[Task] = []
    for task in tasks:
        key = (task.run_id, task.video_id, task.frame_idx, task.track_id)
        if key not in existing_labels:
            pending_tasks.append(task)

    if not pending_tasks:
        print("All tasks already labeled.")
        return

    label_dict = dict(existing_labels)
    window_name = "Dense Track Annotator"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    task_idx = 0
    while task_idx < len(pending_tasks):
        task = pending_tasks[task_idx]
        action = None

        while action is None:
            rendered = render_frame_for_task(
                raw_frame=frames[task.frame_idx],
                detections_by_frame=detections_by_frame,
                frame_idx=task.frame_idx,
                task=task,
                model=model,
                task_idx=task_idx,
                total_tasks=len(pending_tasks),
                clip_radius=clip_radius,
                small_box_area_thresh=small_box_area_thresh,
                small_box_height_thresh=small_box_height_thresh,
            )
            cv2.imshow(window_name, rendered)
            key = cv2.waitKey(0) & 0xFF

            if key == ord("p"):
                maybe_key = play_clip(
                    window_name=window_name,
                    frames=frames,
                    detections_by_frame=detections_by_frame,
                    task=task,
                    model=model,
                    task_idx=task_idx,
                    total_tasks=len(pending_tasks),
                    clip_radius=clip_radius,
                    playback_delay_ms=playback_delay_ms,
                    small_box_area_thresh=small_box_area_thresh,
                    small_box_height_thresh=small_box_height_thresh,
                )
                if maybe_key is None:
                    continue
                key = maybe_key

            if key in [ord("0"), ord("1"), ord("2")]:
                label = int(chr(key))
                label_dict[(task.run_id, task.video_id, task.frame_idx, task.track_id)] = label
                rewrite_labels_csv(labels_csv, label_dict)
                print(
                    f"Labeled: run_id={task.run_id}, video={task.video_id}, "
                    f"frame={task.frame_idx}, track={task.track_id}, label={label}"
                )
                action = "next"

            elif key == ord("s"):
                print(
                    f"Skipped: run_id={task.run_id}, video={task.video_id}, "
                    f"frame={task.frame_idx}, track={task.track_id}"
                )
                action = "skip"

            elif key == ord("b"):
                action = "back"

            elif key == ord("q"):
                action = "quit"

        if action in ("next", "skip"):
            task_idx += 1

        elif action == "back":
            if task_idx > 0:
                prev_task = pending_tasks[task_idx - 1]
                prev_key = (prev_task.run_id, prev_task.video_id, prev_task.frame_idx, prev_task.track_id)
                if prev_key in label_dict:
                    del label_dict[prev_key]
                    rewrite_labels_csv(labels_csv, label_dict)
                    print(f"Removed previous label for relabeling: {prev_key}")
                task_idx -= 1
            else:
                print("Already at the first task.")

        elif action == "quit":
            print("Quitting. Progress saved.")
            break

    cv2.destroyAllWindows()
    print(f"Done. Labels saved to: {labels_csv}")


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Dense track annotator")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Ultralytics model path/name")
    parser.add_argument("--run_id", type=str, default="", help="Optional run ID. If omitted, auto-generated.")
    parser.add_argument(
        "--detections_csv",
        type=str,
        default="tracked_detections.csv",
        help="Output CSV for dense tracked detections"
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        default="track_labels.csv",
        help="Output CSV for sparse labels"
    )
    parser.add_argument("--sample_every", type=int, default=6, help="Create tasks every Nth frame")
    parser.add_argument(
        "--clip_radius",
        type=int,
        default=6,
        help="When user presses p, play +/- this many frames around sampled frame"
    )
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO image size")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument(
        "--small_box_area_thresh",
        type=int,
        default=1200,
        help="Ignore detections with bbox area below this threshold, in pixels^2"
    )
    parser.add_argument(
        "--small_box_height_thresh",
        type=int,
        default=25,
        help="Ignore detections with bbox height below this threshold, in pixels"
    )
    parser.add_argument("--playback_delay_ms", type=int, default=80, help="Playback delay per frame in milliseconds")
    parser.add_argument(
        "--vehicle_classes",
        type=str,
        default="2,3,5,7",
        help="Comma-separated COCO class IDs to keep. Default: car,motorcycle,bus,truck"
    )
    return parser.parse_args()


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    video_id = os.path.splitext(os.path.basename(args.video))[0]
    run_id = args.run_id.strip() if args.run_id.strip() else uuid.uuid4().hex[:12]

    allowed_classes = None
    if args.vehicle_classes.strip():
        allowed_classes = [int(x.strip()) for x in args.vehicle_classes.split(",") if x.strip()]

    print(f"Using run_id: {run_id}")

    print("Loading video frames...")
    frames = load_video_frames(args.video)
    print(f"Loaded {len(frames)} frames.")

    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)

    print("Running tracking on every frame...")
    detections_by_frame = run_tracking(
        model=model,
        frames=frames,
        imgsz=args.imgsz,
        conf=args.conf,
        small_box_area_thresh=args.small_box_area_thresh,
        small_box_height_thresh=args.small_box_height_thresh,
        allowed_classes=allowed_classes,
    )

    print(f"Saving dense tracked detections to: {args.detections_csv}")
    save_tracked_detections_csv(
        csv_path=args.detections_csv,
        run_id=run_id,
        video_id=video_id,
        detections_by_frame=detections_by_frame,
    )

    print("Building sparse annotation tasks...")
    tasks = build_tasks(
        run_id=run_id,
        video_id=video_id,
        detections_by_frame=detections_by_frame,
        sample_every=args.sample_every,
    )
    print(f"Created {len(tasks)} tasks before resume filtering.")

    annotate_tasks(
        frames=frames,
        detections_by_frame=detections_by_frame,
        tasks=tasks,
        model=model,
        labels_csv=args.labels_csv,
        clip_radius=args.clip_radius,
        playback_delay_ms=args.playback_delay_ms,
        small_box_area_thresh=args.small_box_area_thresh,
        small_box_height_thresh=args.small_box_height_thresh,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)