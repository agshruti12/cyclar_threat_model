from ultralytics import YOLO
import cv2
import numpy as np
from src.io_utils import save_detections_json
from src.model_types import Detection
import torch


# For now, hardcode a video path
VIDEO_PATH = "data/raw/sample_bike_ride.mp4"


def load_model():
    # Small model for speed; you can change later
    model = YOLO("yolov8n.pt")
    
    return model

def run_detection_on_video(video_path: str):
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

    model = load_model()
    # device = "mps" if torch.backends.mps.is_available() else "cpu"
    # model.to(device)

    frame_idx = 0
    all_detections = []  # list of dicts: one per frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 3 == 0:
        # Run YOLO
            results = model(frame)[0]  # YOLO result for this frame
            # update tracker with fresh detections
        else:
            # Only update tracker prediction, no YOLO call
            pass

        frame_dets = []
        for box in results.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Coco: 2=car, 3=motorcycle, 5=bus, 7=truck
            if cls_id in [2, 3, 5, 7]:
                detection = Detection(
                    frame_idx=frame_idx,
                    cls_id=cls_id,
                    conf=conf,
                    bbox=(x1, y1, x2, y2),
                )
                frame_dets.append(detection)
                

        all_detections.append(frame_dets)

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    return all_detections

def visualize_some_frames(video_path: str, detections_per_frame, num_frames: int = 50):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while frame_idx < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_dets = detections_per_frame[frame_idx]
        for det in frame_dets:
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{det.cls_id}:{det.conf:.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        cv2.imshow("YOLO detections", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    dets = run_detection_on_video(VIDEO_PATH)
    print(f"Processed {len(dets)} frames")

    out_path = "data/processed/sample_bike_ride_dets.json"
    save_detections_json(dets, out_path)
    print(f"Saved detections to {out_path}")

    visualize_some_frames(VIDEO_PATH, dets, num_frames=50)
