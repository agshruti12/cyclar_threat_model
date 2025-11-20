import glob
import os

from label_video_scene import label_single_video


def main():
    # Adjust this glob if your videos are somewhere else or not mp4
    video_glob = "data/raw/*.mp4"
    video_paths = sorted(glob.glob(video_glob))

    if not video_paths:
        print(f"No videos found matching {video_glob}")
        return

    print("Found videos:")
    for i, vp in enumerate(video_paths):
        print(f"  [{i}] {vp}")

    for vp in video_paths:
        base = os.path.splitext(os.path.basename(vp))[0]
        out_json = os.path.join("data", "labels", f"{base}_scene_labels.json")

        # Skip if already labeled (so you can resume later)
        if os.path.exists(out_json):
            print(f"\nSkipping {vp} (labels already exist at {out_json})")
            continue

        # Label this video using the "play 1s, then pause" UI
        label_single_video(vp, out_json, segment_seconds=1.0)

        ans = input("Continue to next video? [Y/n]: ").strip().lower()
        if ans == "n":
            print("Stopping multi-video labeling.")
            break


if __name__ == "__main__":
    main()
