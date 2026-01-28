import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as Ffunc

from model import DangerGRUClassifier        # from your model.py
from dataset import TEMPORAL_WINDOW         # from your dataset.py

# ---- Tunable thresholds ----
TH_HIGH = 0.75   # minimum p_high to call HIGH
TH_MED  = 0.58   # minimum p_med to call MED
MARGIN_MED_OVER_LOW = 0.07  # how much p_med must beat p_low by
# ----------------------------

def calibrated_class_from_probs(probs_row):
    """
    probs_row: np.array [p_low, p_med, p_high]

    Conservative but not crazy:
      - HIGH if p_high >= TH_HIGH and p_high is max
      - MED if p_med >= TH_MED and p_med >= p_low + MARGIN_MED_OVER_LOW
      - else LOW
    """
    p0, p1, p2 = probs_row  # low, med, high

    # HIGH: fairly confident
    if p2 >= TH_HIGH and p2 > p1 and p2 > p0:
        return 2

    # MED: moderately confident and clearly above LOW
    if p1 >= TH_MED and (p1 - p0) >= MARGIN_MED_OVER_LOW and p1 > p2:
        return 1

    # otherwise, LOW
    return 0



def load_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    frame_features = data["frame_features"]      # (N, F)
    frame_labels = data.get("frame_labels", None)
    fps = float(data["fps"])
    frame_count = int(data["frame_count"])
    video_path = str(data["video_path"])
    return frame_features, frame_labels, fps, frame_count, video_path


def load_model(model_path: str, feature_dim: int, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DangerGRUClassifier(
        feature_dim=feature_dim,
        hidden_dim=64,
        num_layers=1,
        num_classes=3,
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def run_inference_on_npz(
    npz_path: str,
    model_path: str,
    temporal_window: int = TEMPORAL_WINDOW,
):
    """
    Returns:
      frame_preds: (N,) int array in {0,1,2} or -1 for frames before first full window
      frame_probs: (N, 3) float array of softmax probabilities (zeros for early frames)
    """
    frame_features, _, fps, frame_count, video_path = load_npz(npz_path)
    N, feat_dim = frame_features.shape

    model, device = load_model(model_path, feature_dim=feat_dim)

    # Build sliding windows
    xs = []
    idxs = []
    T = temporal_window
    for t in range(T - 1, N):
        window_feats = frame_features[t - T + 1 : t + 1]  # (T, feat_dim)
        xs.append(window_feats)
        idxs.append(t)

    if not xs:
        # video too short
        frame_preds = -np.ones(N, dtype=np.int64)
        frame_probs = np.zeros((N, 3), dtype=np.float32)
        return frame_preds, frame_probs, fps, frame_count, video_path

    X = np.stack(xs, axis=0)   # (num_windows, T, F)
    X = np.asarray(X, dtype=np.float32)      # ensure numeric ndarray
    X = np.ascontiguousarray(X)             # ensure contiguous memory
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

    with torch.no_grad():
        logits = model(X_tensor)           # (num_windows, 3)
        probs = Ffunc.softmax(logits, dim=1)
    probs_np = probs.cpu().numpy()

    # conservative class decisions
    preds_np = np.array(
        [calibrated_class_from_probs(row) for row in probs_np],
        dtype=np.int64,
    )


    # Map to per-frame arrays
    frame_preds = -np.ones(N, dtype=np.int64)
    frame_probs = np.zeros((N, 3), dtype=np.float32)

    for idx, t in enumerate(idxs):
        frame_preds[t] = preds_np[idx]
        frame_probs[t] = probs_np[idx]

    # Optional: fill earlier frames with the first available prediction
    if len(idxs) > 0:
        first_t = idxs[0]
        frame_preds[:first_t] = preds_np[0]
        frame_probs[:first_t] = probs_np[0]

    # After mapping window preds to frame_preds
    unique, counts = np.unique(frame_preds, return_counts=True)
    print("Frame prediction distribution:")
    for u, c in zip(unique, counts):
        print(f"  class {u}: {c} frames")


    return frame_preds, frame_probs, fps, frame_count, video_path


def class_to_color(cls: int):
    """
    Map class 0/1/2 to BGR colors for visualization.
    0 = LOW (green), 1 = MED (yellow), 2 = HIGH (red)
    """
    if cls == 0:
        return (0, 255, 0)      # green
    elif cls == 1:
        return (0, 255, 255)    # yellow
    elif cls == 2:
        return (0, 0, 255)      # red
    else:
        return (128, 128, 128)  # gray for unknown


def visualize_predictions_on_video(
    video_path: str,
    frame_preds: np.ndarray,
    output_video_path: str = None,
    fps: float = None,
):
    """
    Overlays predicted danger levels on the video frames and optionally saves a new video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps is None or fps <= 0:
        fps = cap.get(cv2.CAP_PROP_FPS)

    if output_video_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    else:
        writer = None

    n_preds = len(frame_preds)
    print(f"Visualizing {min(total_frames, n_preds)} frames...")

    frame_idx = 0
    window_name = "Danger Predictions"
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < n_preds:
            cls = int(frame_preds[frame_idx])
        else:
            cls = -1

        color = class_to_color(cls)
        label_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
        label_str = label_map.get(cls, "UNKNOWN")

        # draw a top bar
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

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quit visualization early.")
            break

        if writer is not None:
            writer.write(frame)

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
        print(f"Saved visualization video to: {output_video_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run GRU danger model inference on a video NPZ and visualize predictions."
    )
    parser.add_argument(
        "--npz",
        type=str,
        required=True,
        help="Path to the frame_data .npz file produced by build_frame_data_from_tracks.py",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/danger_gru_classifier.pt",
        help="Path to trained GRU model weights.",
    )
    parser.add_argument(
        "--out-video",
        type=str,
        default=None,
        help="Optional path to save an output visualization video (mp4).",
    )

    args = parser.parse_args()

    frame_preds, frame_probs, fps, frame_count, video_path = run_inference_on_npz(
        npz_path=args.npz,
        model_path=args.model,
        temporal_window=TEMPORAL_WINDOW,
    )

    print(f"Loaded video path from npz: {video_path}")
    print(f"First 20 predictions: {frame_preds[:20]}")

    visualize_predictions_on_video(
        video_path=video_path,
        frame_preds=frame_preds,
        output_video_path=args.out_video,
        fps=fps,
    )


if __name__ == "__main__":
    main()
