from dataclasses import dataclass, field
from typing import List, Tuple

BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2

@dataclass
class Detection:
    frame_idx: int
    cls_id: int
    conf: float
    bbox: BBox

@dataclass
class Track:
    track_id: int
    cls_id: int
    frame_indices: List[int] = field(default_factory=list)
    bboxes: List[BBox] = field(default_factory=list)
    # Later you'll add: distances, TTC, etc.
