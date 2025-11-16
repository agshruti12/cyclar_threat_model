import json
from dataclasses import asdict
from typing import List
from src.model_types import Detection  # adjust import path if needed

def save_detections_json(
    detections_per_frame: List[List[Detection]],
    output_path: str,
) -> None:
    """
    Save detections as a list of frames,
    where each frame is a list of detection dicts.
    """
    serializable = [
        [asdict(det) for det in frame_dets]
        for frame_dets in detections_per_frame
    ]
    with open(output_path, "w") as f:
        json.dump(serializable, f)
