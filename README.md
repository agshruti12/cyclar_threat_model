# Bike Threat Detection – Work Log

## Current Status (Day 1)

- ✅ Project repo + conda environment created.
- ✅ YOLO + OpenCV pipeline reads a sample bike video and runs detection on each frame.
- ✅ Basic data structures for detections and tracks defined.
- ✅ Stub for feature extraction from bbox + depth created.

## Next Steps

1. Integrate a tracking library (e.g., ByteTrack or DeepSORT) into `detect_and_store.py`.
2. Add a depth module wrapper for Video Depth Anything:
   - Run depth on each frame (or every Nth frame).
   - Store depth maps (or derived distances) in `data/processed/`.
3. Implement `compute_features_from_bbox_and_depth` to compute:
   - distance, delta_distance, bbox_area change, TTC, normalized centers.
4. Write script to convert tracks + features into sequences suitable for a temporal model (LSTM/Transformer).
5. Prototype a simple danger classifier (e.g., 2-layer BiLSTM) in `src/models/danger_classifier.py`.
