import os
import csv
import argparse
from typing import Dict, List, Tuple

import cv2


LABEL_NAMES = {
    0: "LOW",
    1: "MED",
    2: "HIGH",
}

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".m4v"}


def get_video_info(cap: cv2.VideoCapture) -> Tuple[float, int]:
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        raise ValueError("Could not determine FPS from video.")
    if frame_count <= 0:
        raise ValueError("Could not determine frame count from video.")

    return fps, frame_count


def build_second_to_frames(frame_count: int, fps: float) -> Dict[int, List[int]]:
    second_to_frames: Dict[int, List[int]] = {}

    for frame_idx in range(frame_count):
        second_idx = int(frame_idx / fps)
        if second_idx not in second_to_frames:
            second_to_frames[second_idx] = []
        second_to_frames[second_idx].append(frame_idx)

    return second_to_frames


def get_representative_frame_index(frame_indices: List[int]) -> int:
    return frame_indices[len(frame_indices) // 2]


def read_frame_at_index(cap: cv2.VideoCapture, frame_idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()
    if not success:
        return None
    return frame


def draw_overlay(
    frame,
    video_name: str,
    second_idx: int,
    side: str,
    left_labels: Dict[int, int],
    right_labels: Dict[int, int],
):
    preview = frame.copy()
    h, w = preview.shape[:2]
    mid_x = w // 2

    dark_overlay = preview.copy()
    dark_overlay[:] = (0, 0, 0)
    preview = cv2.addWeighted(preview, 0.75, dark_overlay, 0.25, 0)

    if side == "Left":
        x1, y1, x2, y2 = 0, 0, mid_x, h
    else:
        x1, y1, x2, y2 = mid_x, 0, w, h

    preview[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

    cv2.rectangle(preview, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 4)
    cv2.line(preview, (mid_x, 0), (mid_x, h), (255, 255, 255), 2)

    lines = [
        f"Video: {video_name}",
        f"Second: {second_idx}",
        f"Pass: {side} side",
        "Press 0 = LOW, 1 = MED, 2 = HIGH",
        "Press b = back, s = skip video, q = quit all",
    ]

    y = 35
    for line in lines:
        cv2.putText(
            preview,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 35

    left_status = LABEL_NAMES[left_labels[second_idx]] if second_idx in left_labels else "UNSET"
    right_status = LABEL_NAMES[right_labels[second_idx]] if second_idx in right_labels else "UNSET"

    status_text = f"Current labels -> Left: {left_status} | Right: {right_status}"
    cv2.putText(
        preview,
        status_text,
        (20, h - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return preview


def wait_for_label_key():
    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord("0"):
            return 0
        if key == ord("1"):
            return 1
        if key == ord("2"):
            return 2
        if key == ord("b"):
            return "back"
        if key == ord("s"):
            return "skip"
        if key == ord("q"):
            return "quit"


def generate_rows_for_second(
    video_stem: str,
    frame_indices: List[int],
    second_idx: int,
    left_label: int,
    right_label: int,
) -> List[Tuple[str, int]]:
    rows = []

    for frame_idx in frame_indices:
        left_frame_id = f"{video_stem}_sec{second_idx:04d}_frame{frame_idx:06d}_Left"
        right_frame_id = f"{video_stem}_sec{second_idx:04d}_frame{frame_idx:06d}_Right"
        rows.append((left_frame_id, left_label))
        rows.append((right_frame_id, right_label))

    return rows


def save_csv(output_csv: str, all_rows: List[Tuple[str, int]]) -> None:
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "danger_label"])
        writer.writerows(all_rows)


def list_video_files(folder_path: str) -> List[str]:
    video_files = []

    for filename in sorted(os.listdir(folder_path)):
        full_path = os.path.join(folder_path, filename)
        if not os.path.isfile(full_path):
            continue

        ext = os.path.splitext(filename)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            video_files.append(full_path)

    return video_files


def label_side_pass(
    cap: cv2.VideoCapture,
    second_indices: List[int],
    second_to_frames: Dict[int, List[int]],
    video_name: str,
    side: str,
    left_labels: Dict[int, int],
    right_labels: Dict[int, int],
) -> str:
    """
    Label one entire side across all seconds.
    Returns:
        'done', 'skip', or 'quit'
    """
    window_name = "CyclAR Labeling"
    current_pos = 0

    while current_pos < len(second_indices):
        second_idx = second_indices[current_pos]
        frame_indices = second_to_frames[second_idx]
        rep_frame_idx = get_representative_frame_index(frame_indices)

        frame = read_frame_at_index(cap, rep_frame_idx)
        if frame is None:
            print(f"Warning: could not read representative frame for second {second_idx}. Skipping.")
            current_pos += 1
            continue

        preview = draw_overlay(
            frame=frame,
            video_name=video_name,
            second_idx=second_idx,
            side=side,
            left_labels=left_labels,
            right_labels=right_labels,
        )
        cv2.imshow(window_name, preview)
        result = wait_for_label_key()

        if result == "quit":
            return "quit"

        if result == "skip":
            return "skip"

        if result == "back":
            if current_pos > 0:
                current_pos -= 1
            else:
                print("Already at the first second.")
            continue

        if side == "Left":
            left_labels[second_idx] = result
            print(f"Left | second {second_idx}: {LABEL_NAMES[result]} ({result})")
        else:
            right_labels[second_idx] = result
            print(f"Right | second {second_idx}: {LABEL_NAMES[result]} ({result})")

        current_pos += 1

    return "done"


def label_single_video(video_path: str, output_csv: str) -> str:
    video_name = os.path.basename(video_path)
    video_stem = os.path.splitext(video_name)[0]

    cap = cv2.VideoCapture(video_path)
    fps, frame_count = get_video_info(cap)
    second_to_frames = build_second_to_frames(frame_count, fps)
    second_indices = sorted(second_to_frames.keys())

    print("\n" + "=" * 80)
    print(f"Labeling video: {video_name}")
    print(f"FPS: {fps:.4f}")
    print(f"Frame count: {frame_count}")
    print(f"Seconds to label: {len(second_indices)}")
    print("Flow:")
    print("  1. Label ALL left-side seconds")
    print("  2. Label ALL right-side seconds")
    print("Controls:")
    print("  0 = LOW")
    print("  1 = MED")
    print("  2 = HIGH")
    print("  b = back")
    print("  s = skip video")
    print("  q = quit all")
    print("=" * 80)

    left_labels: Dict[int, int] = {}
    right_labels: Dict[int, int] = {}

    cv2.namedWindow("CyclAR Labeling", cv2.WINDOW_NORMAL)

    print("\nStarting LEFT-side labeling pass...")
    status = label_side_pass(
        cap=cap,
        second_indices=second_indices,
        second_to_frames=second_to_frames,
        video_name=video_name,
        side="Left",
        left_labels=left_labels,
        right_labels=right_labels,
    )

    if status == "quit":
        cap.release()
        cv2.destroyAllWindows()
        return "quit"

    if status == "skip":
        print(f"Skipped video: {video_name}")
        cap.release()
        cv2.destroyAllWindows()
        return "skip"

    print("\nStarting RIGHT-side labeling pass...")
    status = label_side_pass(
        cap=cap,
        second_indices=second_indices,
        second_to_frames=second_to_frames,
        video_name=video_name,
        side="Right",
        left_labels=left_labels,
        right_labels=right_labels,
    )

    if status == "quit":
        cap.release()
        cv2.destroyAllWindows()
        return "quit"

    if status == "skip":
        print(f"Skipped video: {video_name}")
        cap.release()
        cv2.destroyAllWindows()
        return "skip"

    cap.release()
    cv2.destroyAllWindows()

    if not left_labels or not right_labels:
        print(f"Incomplete labels for {video_name}. CSV not written.")
        return "done"

    all_rows: List[Tuple[str, int]] = []

    for second_idx in second_indices:
        if second_idx not in left_labels or second_idx not in right_labels:
            print(f"Missing labels for second {second_idx}; skipping CSV write.")
            return "done"

        rows = generate_rows_for_second(
            video_stem=video_stem,
            frame_indices=second_to_frames[second_idx],
            second_idx=second_idx,
            left_label=left_labels[second_idx],
            right_label=right_labels[second_idx],
        )
        all_rows.extend(rows)

    save_csv(output_csv, all_rows)
    print(f"Saved labels to: {output_csv}")
    print(f"Total rows: {len(all_rows)}")

    return "done"


def main(input_folder: str, output_folder: str) -> None:
    if not os.path.isdir(input_folder):
        raise NotADirectoryError(f"Input folder not found: {input_folder}")

    os.makedirs(output_folder, exist_ok=True)

    video_files = list_video_files(input_folder)
    if not video_files:
        print("No video files found in the folder.")
        return

    print(f"Found {len(video_files)} video(s) in: {input_folder}")
    print("Each video will produce its own CSV named <video_name>_labels.csv")

    for video_path in video_files:
        video_name = os.path.basename(video_path)
        video_stem = os.path.splitext(video_name)[0]
        output_csv = os.path.join(output_folder, f"{video_stem}_labels.csv")

        status = label_single_video(video_path, output_csv)

        if status == "quit":
            print("Quitting labeling session.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Label all videos in a folder with left-side pass first, then right-side pass."
    )
    parser.add_argument(
        "--input_folder",
        required=True,
        help="Folder containing video files",
    )
    parser.add_argument(
        "--output_folder",
        required=True,
        help="Folder where CSVs will be saved",
    )

    args = parser.parse_args()
    main(args.input_folder, args.output_folder)