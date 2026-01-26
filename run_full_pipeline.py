import argparse
import glob
import os

from src.first_model.detect_and_store import run_detection_on_video  # YOLO detections
from src.first_model.io_utils import save_detections_json        # to save dets

from src.first_model.build_tracks import load_detections, build_tracks, save_tracks  # tracking
from src.first_model.save_track_features import (                                   # features
    load_tracks as load_tracks_json,
    get_video_size,
    save_features,
)
from src.first_model.features import compute_track_features


def process_single_video(video_path: str, processed_dir: str = "data/processed") -> None:
    """
    Full pipeline for one video:
      1) YOLO detections -> *_dets.json
      2) IoU tracking    -> *_tracks.json
      3) Track features  -> *_tracks_features.json
    """

    os.makedirs(processed_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(video_path))[0]

    dets_json_path = os.path.join(processed_dir, f"{base}_dets.json")
    tracks_json_path = os.path.join(processed_dir, f"{base}_tracks.json")
    features_json_path = os.path.join(processed_dir, f"{base}_tracks_features.json")

    print("\n" + "=" * 70)
    print(f"Processing video: {video_path}")
    print(f"  detections json : {dets_json_path}")
    print(f"  tracks json     : {tracks_json_path}")
    print(f"  features json   : {features_json_path}")

    # ------------------------
    # 1) Run YOLO detections
    # ------------------------
    print("\n[1/3] Running YOLO detections...")
    detections_per_frame = run_detection_on_video(video_path)
    print(f"  YOLO produced detections for {len(detections_per_frame)} frames")

    save_detections_json(detections_per_frame, dets_json_path)
    print(f"  Saved detections to {dets_json_path}")

    # ------------------------
    # 2) Build tracks (IoU)
    # ------------------------
    print("\n[2/3] Building tracks with IoU tracker...")
    dets_loaded = load_detections(dets_json_path)
    tracks = build_tracks(dets_loaded)
    print(f"  Built {len(tracks)} tracks")

    save_tracks(tracks, tracks_json_path)
    print(f"  Saved tracks to {tracks_json_path}")

    # ------------------------
    # 3) Compute track features
    # ------------------------
    print("\n[3/3] Computing track features...")
    # reload tracks as dicts (for compute_track_features)
    tracks_list = load_tracks_json(tracks_json_path)
    frame_width, frame_height = get_video_size(video_path)
    print(f"  Video size: {frame_width}x{frame_height}")

    features_per_track = []
    for t in tracks_list:
        ft = compute_track_features(t, frame_width, frame_height)
        features_per_track.append(ft)

    save_features(features_per_track, features_json_path)
    print(f"  Saved features for {len(features_per_track)} tracks to {features_json_path}")

    print(f"Done processing {video_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run full pipeline (detections -> tracks -> track features) "
            "for one or more videos."
        )
    )

    parser.add_argument(
        "--videos",
        type=str,
        nargs="+",
        help="Explicit list of video paths to process (e.g., data/raw/vid1.mp4 data/raw/vid2.mp4).",
    )

    parser.add_argument(
        "--glob",
        type=str,
        default="data/raw/*.mp4",
        help="Glob pattern for videos if --videos is not provided (default: data/raw/*.mp4).",
    )

    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Directory to store *_dets.json, *_tracks.json, *_tracks_features.json.",
    )

    args = parser.parse_args()

    # Decide which video list to use
    if args.videos is not None and len(args.videos) > 0:
        video_paths = args.videos
        print("Using explicit video list:")
        for vp in video_paths:
            print("  ", vp)
    else:
        video_paths = sorted(glob.glob(args.glob))
        print(f"Using glob pattern: {args.glob}")
        if not video_paths:
            print("No videos found. Exiting.")
            return
        print("Found videos:")
        for vp in video_paths:
            print("  ", vp)

    for vp in video_paths:
        if not os.path.exists(vp):
            print(f"WARNING: video not found, skipping: {vp}")
            continue

        process_single_video(vp, processed_dir=args.processed_dir)

    print("\nAll videos processed.")


if __name__ == "__main__":
    main()
