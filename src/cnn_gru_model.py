# cnn_gru_model.py

import torch
import torch.nn as nn
import torchvision.models as models


class CNNBackbone(nn.Module):
    """
    Wrap a pretrained ResNet18 and output a feature vector per frame.
    """

    def __init__(self, output_dim: int = 512, train_backbone: bool = False):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Remove the classification head, keep everything up to global pool
        modules = list(resnet.children())[:-1]  # drop fc
        self.backbone = nn.Sequential(*modules)
        self.output_dim = output_dim

        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B*T, 3, H, W)
        Returns: (B*T, F)
        """
        feats = self.backbone(x)  # (B*T, F, 1, 1)
        feats = feats.view(feats.size(0), -1)  # (B*T, F)
        return feats


class CNNGRUDangerModel(nn.Module):
    """
    Model that:
      - Applies CNN backbone frame-wise
      - Feeds per-frame embeddings into GRU
      - Classifies final hidden state into 3 danger classes
    """

    def __init__(
        self,
        cnn_feature_dim: int = 512,
        gru_hidden_dim: int = 128,
        gru_layers: int = 1,
        num_classes: int = 3,
        train_backbone: bool = False,
    ):
        super().__init__()

        self.cnn = CNNBackbone(
            output_dim=cnn_feature_dim,
            train_backbone=train_backbone,
        )

        self.gru = nn.GRU(
            input_size=cnn_feature_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(gru_hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W)
        Returns: logits (B, num_classes)
        """
        B, T, C, H, W = x.shape

        x_flat = x.view(B * T, C, H, W)  # (B*T, C, H, W)
        feats = self.cnn(x_flat)         # (B*T, F)

        Fdim = feats.size(1)
        feats = feats.view(B, T, Fdim)   # (B, T, F)

        # GRU
        out, h_n = self.gru(feats)       # out: (B, T, H), h_n: (num_layers, B, H)
        last_h = h_n[-1]                 # (B, H)

        logits = self.fc(last_h)         # (B, num_classes)
        return logits
