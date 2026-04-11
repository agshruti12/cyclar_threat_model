from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_features import QueueConfig, compute_object_features_from_csv, save_features_dataframe


def labels_path_for_detection(det_csv: Path, labels_dir: Path, detections_suffix: str) -> Path:
    """
    Map a detections CSV to its corresponding labels CSV.

    Example:
        det_csv:   detections/ride01_detections.csv
        returns:   labels/ride01_labels.csv

    If the detections filename does not end with the configured suffix,
    falls back to appending '_labels' to the stem.
    """
    stem = det_csv.stem  # e.g. "ride01_detections"

    if stem.endswith(detections_suffix):
        base = stem[: -len(detections_suffix)]  # e.g. "ride01"
    else:
        base = stem

    return labels_dir / f"{base}_labels.csv"


def build_from_folder(
    detections_dir: str,
    labels_dir: str,
    out_csv: str,
    frame_w: int,
    frame_h: int,
    queue_len: int,
    sample_every: int,
    ema_alpha: float,
    stale_after_frames: int,
    detections_suffix: str = "_detections",
    skip_missing_labels: bool = True,
) -> pd.DataFrame:
    det_dir = Path(detections_dir)
    if not det_dir.exists():
        raise FileNotFoundError(f"Detections directory not found: {det_dir}")

    lab_dir = Path(labels_dir)
    if not lab_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {lab_dir}")

    det_paths = sorted(det_dir.glob("*.csv"))
    if not det_paths:
        raise RuntimeError(f"No detections CSV files found in: {det_dir}")

    cfg = QueueConfig(
        queue_len=queue_len,
        sample_every=sample_every,
        ema_alpha=ema_alpha,
        stale_after_frames=stale_after_frames,
    )

    all_parts = []

    for det_csv in det_paths:
        labels_csv = labels_path_for_detection(
            det_csv=det_csv,
            labels_dir=lab_dir,
            detections_suffix=detections_suffix,
        )

        if not labels_csv.exists():
            msg = f"No matching labels file for {det_csv.name}: expected {labels_csv.name}"
            if skip_missing_labels:
                print(f"Skipping. {msg}")
                continue
            raise FileNotFoundError(msg)

        print(f"Building features from:")
        print(f"  detections: {det_csv}")
        print(f"  labels:     {labels_csv}")

        part = compute_object_features_from_csv(
            detections_csv=det_csv,
            labels_csv=str(labels_csv),
            frame_w=frame_w,
            frame_h=frame_h,
            config=cfg,
        )

        if not part.empty:
            all_parts.append(part)

    if all_parts:
        final_df = pd.concat(all_parts, ignore_index=True)
    else:
        final_df = pd.DataFrame()

    save_features_dataframe(final_df, out_csv)
    print(f"Saved training features -> {out_csv}")
    return final_df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build object-level training features from per-video detections/labels CSVs"
    )
    ap.add_argument(
        "--detections_dir",
        required=True,
        help="Folder containing per-video detections CSV files, e.g. ride01_detections.csv",
    )
    ap.add_argument(
        "--labels_dir",
        required=True,
        help="Folder containing per-video labels CSV files, e.g. ride01_labels.csv",
    )
    ap.add_argument("--out", required=True, help="Output CSV for training features")
    ap.add_argument("--frame_w", type=int, default=1920)
    ap.add_argument("--frame_h", type=int, default=1080)
    ap.add_argument("--queue_len", type=int, default=24)
    ap.add_argument("--sample_every", type=int, default=6)
    ap.add_argument("--ema_alpha", type=float, default=0.30)
    ap.add_argument("--stale_after_frames", type=int, default=30)
    ap.add_argument(
        "--detections_suffix",
        default="_detections",
        help="Suffix used in detections filenames before .csv, default: _detections",
    )
    ap.add_argument(
        "--error_on_missing_labels",
        action="store_true",
        help="If set, fail instead of skipping detections files that do not have matching labels",
    )

    args = ap.parse_args()

    build_from_folder(
        detections_dir=args.detections_dir,
        labels_dir=args.labels_dir,
        out_csv=args.out,
        frame_w=args.frame_w,
        frame_h=args.frame_h,
        queue_len=args.queue_len,
        sample_every=args.sample_every,
        ema_alpha=args.ema_alpha,
        stale_after_frames=args.stale_after_frames,
        detections_suffix=args.detections_suffix,
        skip_missing_labels=not args.error_on_missing_labels,
    )


if __name__ == "__main__":
    main()