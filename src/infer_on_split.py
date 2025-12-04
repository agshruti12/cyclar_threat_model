import argparse
import json
import os
import os.path as osp

import cv2
import numpy as np

from infer_and_visualize import run_inference_on_npz, class_to_color
from dataset import TEMPORAL_WINDOW  # to keep things consistent


def visualize_predictions_to_file(
    video_path: str,
    frame_preds: np.ndarray,
    out_path: str,
    fps: float = None,
):
    """
    Writes a new MP4 video with a danger label bar overlaid on each frame.
    No OpenCV window is shown (batch-friendly).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps is None or fps <= 0:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # fallback

    os.makedirs(osp.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    n_preds = len(frame_preds)
    print(f"  Visualizing {min(total_frames, n_preds)} frames to {out_path}")

    frame_idx = 0
    label_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < n_preds:
            cls = int(frame_preds[frame_idx])
        else:
            cls = -1

        color = class_to_color(cls)
        label_str = label_map.get(cls, "UNKNOWN")

        # Draw a top bar
        cv2.rectangle(frame, (0, 0), (width, 60), (0, 0, 0), thickness=-1)
        cv2.putText(
            frame,
            f"Frame {frame_idx}  Danger: {label_str}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run danger GRU inference + video visualization on all test videos from splits.json."
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="data/training/splits.json",
        help="Path to splits.json produced by make_splits.py",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/danger_gru_classifier.pt",
        help="Path to trained GRU model weights.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/vis/test_preds",
        help="Directory to save visualization MP4s.",
    )

    args = parser.parse_args()

    # 1. Load splits
    if not osp.exists(args.splits):
        raise FileNotFoundError(f"Splits file not found: {args.splits}")

    with open(args.splits, "r") as f:
        splits = json.load(f)

    test_npz_paths = splits.get("test", [])
    if not test_npz_paths:
        print("No test NPZ paths found in splits['test']. Nothing to do.")
        return

    print("Test NPZ files:")
    for p in test_npz_paths:
        print("  ", p)

    # 2. For each test npz, run inference + write video
    for npz_path in test_npz_paths:
        if not osp.exists(npz_path):
            print(f"WARNING: test npz not found, skipping: {npz_path}")
            continue

        print("\n==============================================")
        print(f"Processing test NPZ: {npz_path}")

        frame_preds, frame_probs, fps, frame_count, video_path = run_inference_on_npz(
            npz_path=npz_path,
            model_path=args.model,
            temporal_window=TEMPORAL_WINDOW,
        )

        print(f"  Video path: {video_path}")
        print(f"  Frame count (npz): {frame_count}")
        print(f"  First 10 preds: {frame_preds[:10]}")

        # Build an output video path based on npz basename
        base = osp.splitext(osp.basename(npz_path))[0]  # e.g. output_chunk_012_frame_data
        out_name = f"{base}_pred_vis.mp4"
        out_path = osp.join(args.out_dir, out_name)

        visualize_predictions_to_file(
            video_path=video_path,
            frame_preds=frame_preds,
            out_path=out_path,
            fps=fps,
        )

    print("\nAll test videos processed.")


if __name__ == "__main__":
    main()
