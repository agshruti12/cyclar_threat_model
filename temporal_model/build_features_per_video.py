from __future__ import annotations

import argparse
from pathlib import Path

from build_features import QueueConfig, compute_object_features_from_csv, save_features_dataframe


def labels_path_for_detection(det_csv: Path, labels_dir: Path, detections_suffix: str) -> Path:
    stem = det_csv.stem

    if stem.endswith(detections_suffix):
        base = stem[: -len(detections_suffix)]
    else:
        base = stem

    return labels_dir / f"{base}_labels.csv"


def output_path_for_detection(det_csv: Path, out_dir: Path, detections_suffix: str) -> Path:
    stem = det_csv.stem

    if stem.endswith(detections_suffix):
        base = stem[: -len(detections_suffix)]
    else:
        base = stem

    return out_dir / f"{base}_features.csv"


def main():
    ap = argparse.ArgumentParser(description="Build per-video feature CSVs")
    ap.add_argument("--detections_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--frame_w", type=int, default=1920)
    ap.add_argument("--frame_h", type=int, default=1080)
    ap.add_argument("--queue_len", type=int, default=24)
    ap.add_argument("--sample_every", type=int, default=6)
    ap.add_argument("--ema_alpha", type=float, default=0.3)
    ap.add_argument("--stale_after_frames", type=int, default=30)

    ap.add_argument("--detections_suffix", default="_detections")
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    det_dir = Path(args.detections_dir)
    lab_dir = Path(args.labels_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = QueueConfig(
        queue_len=args.queue_len,
        sample_every=args.sample_every,
        ema_alpha=args.ema_alpha,
        stale_after_frames=args.stale_after_frames,
    )

    for det_csv in sorted(det_dir.glob("*.csv")):
        labels_csv = labels_path_for_detection(det_csv, lab_dir, args.detections_suffix)

        if not labels_csv.exists():
            print(f"Skipping {det_csv.name} (no labels)")
            continue

        out_csv = output_path_for_detection(det_csv, out_dir, args.detections_suffix)

        if out_csv.exists() and not args.overwrite:
            print(f"Skipping {out_csv.name} (already exists)")
            continue

        print(f"Processing {det_csv.name}")

        df = compute_object_features_from_csv(
            detections_csv=str(det_csv),
            labels_csv=str(labels_csv),
            frame_w=args.frame_w,
            frame_h=args.frame_h,
            config=cfg,
        )

        save_features_dataframe(df, str(out_csv))
        print(f"Saved -> {out_csv}")


if __name__ == "__main__":
    main()