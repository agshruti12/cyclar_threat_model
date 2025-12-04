import torch
import torch.nn as nn


class DangerGRUClassifier(nn.Module):
    """
    GRU-based temporal classifier.

    Input:  x of shape (B, T, F)
    Output: logits of shape (B, 3)
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64, num_layers: int = 1, num_classes: int = 3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.gru(x)         # out: (B, T, H)
        last_hidden = out[:, -1, :]  # (B, H)
        logits = self.fc(last_hidden)
        return logits
