import cv2
import json
import math
import os
import argparse
from typing import List, Dict, Optional

# default segment length (in seconds)
SEGMENT_SECONDS = 1.0


def get_video_info(cap) -> Dict:
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }


def make_segments(frame_count: int, fps: float, seg_seconds: float) -> List[Dict]:
    frames_per_seg = int(round(seg_seconds * fps))
    frames_per_seg = max(1, frames_per_seg)

    num_segments = math.ceil(frame_count / frames_per_seg)
    segments: List[Dict] = []

    for seg_idx in range(num_segments):
        start_frame = seg_idx * frames_per_seg
        end_frame = min(frame_count - 1, (seg_idx + 1) * frames_per_seg - 1)

        start_time = start_frame / fps
        end_time = end_frame / fps

        segments.append(
            {
                "segment_idx": seg_idx,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_sec": start_time,
                "end_time_sec": end_time,
                "label": None,  # to be filled with 0/1/2 later
            }
        )

    return segments


def draw_overlay(frame, seg_info: Dict, current_label: Optional[int]):

    text1 = (
        f"Segment {seg_info['segment_idx']}  "
        f"{seg_info['start_time_sec']:.1f}s - {seg_info['end_time_sec']:.1f}s"
    )

    if current_label is None:
        label_str = "Label: [unlabeled]"
    else:
        label_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
        label_str = f"Label: {label_map.get(current_label, '?')} ({current_label})"

    instructions = "Keys: 1=LOW, 2=MED, 3=HIGH, 0=SKIP, b=BACK, q=QUIT+SAVE"

    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), thickness=-1)

    cv2.putText(
        frame,
        text1,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label_str,
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        instructions,
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return frame


def label_single_video(
    video_path: str,
    output_json: str,
    segment_seconds: float = SEGMENT_SECONDS,
):
    """
    Play each ~segment_seconds window, then pause on the last frame for labeling.
    Keys:
      1 = LOW (0)
      2 = MED (1)
      3 = HIGH (2)
      0 = skip / unlabeled
      b = go back one segment
      q = quit + save
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    info = get_video_info(cap)
    fps = info["fps"]
    frame_count = info["frame_count"]

    print(f"\n=== Labeling video: {video_path} ===")
    print(f"FPS={fps:.2f}, frames={frame_count}, size={info['width']}x{info['height']}")
    print(f"Output JSON: {output_json}")

    segments = make_segments(frame_count, fps, segment_seconds)
    print(f"Number of segments: {len(segments)}  (segment length ~{segment_seconds}s)")

    labels = {seg["segment_idx"]: seg.get("label", None) for seg in segments}

    seg_idx = 0
    window_name = "Scene Danger Labeling"
    delay_ms = int(1000 / fps) if fps > 0 else 33  # playback speed for "live" window

    while 0 <= seg_idx < len(segments):
        seg = segments[seg_idx]
        start_f = seg["start_frame"]
        end_f = seg["end_frame"]

        # 1) PLAY THE WHOLE SEGMENT AS A MINI-CLIP
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        last_frame = None

        for frame_idx in range(start_f, end_f + 1):
            ret, frame = cap.read()
            if not ret:
                print(f"Could not read frame {frame_idx}.")
                break

            last_frame = frame.copy()

            # Minimal overlay during playback
            play_vis = last_frame.copy()
            msg = (
                f"Playing segment {seg_idx} "
                f"({seg['start_time_sec']:.1f}-{seg['end_time_sec']:.1f}s)"
            )
            cv2.putText(
                play_vis,
                msg,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, play_vis)

            _ = cv2.waitKey(delay_ms) & 0xFF

        if last_frame is None:
            print(f"Skipping segment {seg_idx} (no frames read).")
            seg_idx += 1
            continue

        # 2) PAUSE ON LAST FRAME FOR LABELING
        current_label = labels[seg_idx]
        vis = last_frame.copy()
        vis = draw_overlay(vis, seg, current_label)
        cv2.imshow(window_name, vis)

        key = cv2.waitKey(0) & 0xFF  # wait for your label input

        if key == ord("q"):
            print("Quitting and saving labels for this video...")
            break
        elif key == ord("b"):
            seg_idx = max(0, seg_idx - 1)
            print(f"Going back to segment {seg_idx}")
            continue
        elif key == ord("0"):
            labels[seg_idx] = None
            print(f"Segment {seg_idx} marked as SKIP/UNLABELED")
            seg_idx += 1
        elif key == ord("1"):
            labels[seg_idx] = 0
            print(f"Segment {seg_idx} labeled LOW (0)")
            seg_idx += 1
        elif key == ord("2"):
            labels[seg_idx] = 1
            print(f"Segment {seg_idx} labeled MEDIUM (1)")
            seg_idx += 1
        elif key == ord("3"):
            labels[seg_idx] = 2
            print(f"Segment {seg_idx} labeled HIGH (2)")
            seg_idx += 1
        else:
            print(f"Ignoring key: {chr(key) if key < 128 else key}")

    cap.release()
    cv2.destroyAllWindows()

    # write labels into segments
    for seg in segments:
        seg_label = labels[seg["segment_idx"]]
        seg["label"] = seg_label  # can be None if skipped

    payload = {
        "video_path": video_path,
        "fps": fps,
        "frame_count": frame_count,
        "segment_seconds": segment_seconds,
        "segments": segments,
    }

    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved labels to {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Scene-level danger labeling for bike videos."
    )
    parser.add_argument("--video", type=str, required=True, help="Path to video file.")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path (optional). If not provided, derived from video name.",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=SEGMENT_SECONDS,
        help="Length of each labeled segment in seconds.",
    )

    args = parser.parse_args()

    video_path = args.video
    if args.out is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_json = os.path.join("data", "labels", f"{base}_scene_labels.json")
    else:
        output_json = args.out

    label_single_video(video_path, output_json, segment_seconds=args.segment_seconds)


if __name__ == "__main__":
    main()
