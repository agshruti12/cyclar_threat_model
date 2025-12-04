# src/danger_inference.py

import os
from typing import List, Dict, Tuple

import numpy as np
import torch

from train_danger_model import DangerMLP  # or copy class here


MODEL_PATH = "models/danger_mlp.pt"


class DangerScorer:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)

        ckpt = torch.load(model_path, map_location="cpu")
        input_dim = ckpt["input_dim"]

        self.model = DangerMLP(input_dim=input_dim)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def score_scene(self, feats: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        feats: (D,)
        returns: (pred_class, probs_array_of_len_3)
        """
        with torch.no_grad():
            x = torch.from_numpy(feats.astype(np.float32)).unsqueeze(0)  # (1, D)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(probs.argmax())
        return pred, probs
