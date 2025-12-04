import json
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from dataset import DangerVideoDataset, TEMPORAL_WINDOW
from model import DangerGRUClassifier


def load_splits(path="data/training/splits.json"):
    with open(path, "r") as f:
        return json.load(f)


def evaluate(split_name="val", splits_path="data/training/splits.json", model_path="models/danger_gru_classifier.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = load_splits(splits_path)
    npz_paths = splits[split_name]

    ds = DangerVideoDataset(npz_paths, temporal_window=TEMPORAL_WINDOW, low_stride=1, include_labels=True)

    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)

    # get feature dim
    sample_X, _ = next(iter(loader))
    _, T, feat_dim = sample_X.shape

    model = DangerGRUClassifier(feature_dim=feat_dim, hidden_dim=64, num_layers=1, num_classes=3)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_y = []
    all_pred = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            preds = logits.argmax(dim=1)

            all_y.append(y.cpu().numpy())
            all_pred.append(preds.cpu().numpy())

    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)

    print("Confusion matrix:")
    print(confusion_matrix(all_y, all_pred, labels=[0, 1, 2]))

    print("\nClassification report:")
    print(classification_report(all_y, all_pred, labels=[0, 1, 2], target_names=["LOW", "MED", "HIGH"]))


if __name__ == "__main__":
    evaluate(split_name="val")
