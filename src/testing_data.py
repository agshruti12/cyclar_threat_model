import pandas as pd
import cv2
import os

# Load the CSV
df = pd.read_csv("hf://datasets/smart-dashcam/motorcycle-accident-driving-datasets/train.csv")

print(df.head())
print(df.columns)

# Extract the first video path
video_path = df.iloc[0]['filename']

print(f"Playing video: {video_path}")

# Open the video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video file: {video_path}")

# Create display window
window_name = "First Video"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Play video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break  # end of video
    
    cv2.imshow(window_name, frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.dest
