import argparse
import json
import os
import os.path as osp

import cv2
import numpy as np
import torch
import torch.nn.functional as Ffunc
from torch.utils.data import DataLoader

from cnn_video_dataset import SegmentVideoDataset, load_label_file
from cnn_gru_model import CNNGRUDangerModel


# ---------- Visualization helpers ----------

def class_to_color(cls: int):
    """
    Map class 0/1/2 to BGR colors for visualization.
    0 = LOW (green), 1 = MED (yellow), 2 = HIGH (red), -1/other = gray
    """
    if cls == 0:
        return (0, 255, 0)      # green
    elif cls == 1:
        return (0, 255, 255)    # yellow
    elif cls == 2:
        return (0, 0, 255)      # red
    else:
        return (128, 128, 128)  # gray


def class_to_str(cls: int):
    label_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    return label_map.get(cls, "UNKNOWN")


# ---------- Inference ----------

def run_inference_on_labels_json(
    labels_json_path: str,
    model_path: str,
    frames_per_segment: int = 8,
    img_size: int = 224,
    batch_size: int = 4,
    device: str = None,
):
    """
    Runs CNN+GRU model on all labeled segments from a label JSON file.

    Returns:
      video_path: str
      frame_pred_classes: np.ndarray of shape (frame_count,) with values {0,1,2} or -1
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load labels JSON to get segments + video_path + frame_count
    label_data = load_label_file(labels_json_path)
    video_path = label_data["video_path"]
    frame_count = int(label_data["frame_count"])
    segments = label_data["segments"]

    print(f"Loaded labels from: {labels_json_path}")
    print(f"Video: {video_path}")
    print(f"Total frames: {frame_count}, num segments: {len(segments)}")

    # Dataset over this single JSON, drop segments with label=None
    dataset = SegmentVideoDataset(
        label_json_paths=[labels_json_path],
        frames_per_segment=frames_per_segment,
        img_size=img_size,
        train=False,
        drop_unlabeled=True,
    )

    if len(dataset) == 0:
        print("No labeled segments found (all labels None). Nothing to infer.")
        return video_path, np.full(frame_count, -1, dtype=np.int64)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Build model, load weights
    model = CNNGRUDangerModel(
        cnn_feature_dim=512,
        gru_hidden_dim=128,
        gru_layers=1,
        num_classes=3,
        train_backbone=False,   # must match training config
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Run inference per segment (in dataset order)
    all_segment_preds = []

    with torch.no_grad():
        for X, _ in loader:
            X = X.to(device)  # (B, T, C, H, W)
            logits = model(X)  # (B, 3)
            probs = Ffunc.softmax(logits, dim=1)  # (B, 3)
            preds = probs.argmax(dim=1)           # (B,)

            all_segment_preds.append(preds.cpu().numpy())

    all_segment_preds = np.concatenate(all_segment_preds, axis=0)  # (num_labeled_segments,)

    print("Predicted segment class distribution:")
    uniq, cnts = np.unique(all_segment_preds, return_counts=True)
    for u, c in zip(uniq, cnts):
        print(f"  class {u}: {c} segments")

    # Map segment-level predictions back to segments array (which may include unlabeled ones)
    seg_pred_classes = []
    pred_idx = 0
    for seg in segments:
        if seg["label"] is None:
            seg_pred_classes.append(-1)  # no prediction for unlabeled segments
        else:
            if pred_idx >= len(all_segment_preds):
                # Should not happen if dataset and segments are aligned
                seg_pred_classes.append(-1)
            else:
                seg_pred_classes.append(int(all_segment_preds[pred_idx]))
                pred_idx += 1

    seg_pred_classes = np.array(seg_pred_classes, dtype=np.int64)

    # Now build per-frame predictions based on segment predictions
    frame_pred_classes = np.full(frame_count, -1, dtype=np.int64)

    for seg, cls in zip(segments, seg_pred_classes):
        start_f = int(seg["start_frame"])
        end_f = int(seg["end_frame"])
        if cls == -1:
            continue
        start_f = max(0, min(frame_count - 1, start_f))
        end_f = max(0, min(frame_count - 1, end_f))
        frame_pred_classes[start_f : end_f + 1] = cls

    # Optional: print frame-level distribution
    uniq_f, cnts_f = np.unique(frame_pred_classes, return_counts=True)
    print("Frame-wise prediction distribution:")
    for u, c in zip(uniq_f, cnts_f):
        print(f"  class {u}: {c} frames")

    dataset.close()

    return video_path, frame_pred_classes


# ---------- Visualization ----------

def visualize_predictions_on_video(
    video_path: str,
    frame_pred_classes: np.ndarray,
    out_video_path: str,
):
    """
    Overlay predicted danger labels on each frame and save to a new MP4.
    """
    if not osp.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback

    os.makedirs(osp.dirname(out_video_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    n_preds = len(frame_pred_classes)
    print(f"Writing visualization to {out_video_path}")
    print(f"Video frames: {total_frames}, preds length: {n_preds}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < n_preds:
            cls = int(frame_pred_classes[frame_idx])
        else:
            cls = -1

        color = class_to_color(cls)
        label_str = class_to_str(cls)

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
    print(f"Saved visualization video to: {out_video_path}")


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="Run CNN+GRU danger model on a labeled video and visualize predictions."
    )
    parser.add_argument(
        "--labels-json",
        type=str,
        required=True,
        help="Path to scene label JSON for a single video (output of label_video_scene.py / label_all_videos).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/cnn_gru_danger_model.pt",
        help="Path to trained CNN+GRU model weights.",
    )
    parser.add_argument(
        "--frames-per-seg",
        type=int,
        default=8,
        help="Number of frames per segment used in training/inference.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Input resolution for CNN (height=width).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--out-video",
        type=str,
        default=None,
        help="Output MP4 path for visualization. If not provided, will create one next to the labels JSON.",
    )

    args = parser.parse_args()

    if args.out_video is None:
        base = osp.splitext(osp.basename(args.labels_json))[0]
        out_dir = "data/vis_cnn_gru"
        os.makedirs(out_dir, exist_ok=True)
        args.out_video = osp.join(out_dir, f"{base}_pred_vis.mp4")

    video_path, frame_pred_classes = run_inference_on_labels_json(
        labels_json_path=args.labels_json,
        model_path=args.model,
        frames_per_segment=args.frames_per_seg,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    visualize_predictions_on_video(
        video_path=video_path,
        frame_pred_classes=frame_pred_classes,
        out_video_path=args.out_video,
    )


if __name__ == "__main__":
    main()
