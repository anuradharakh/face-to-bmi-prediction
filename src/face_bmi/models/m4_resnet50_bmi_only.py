import torch
import torch.nn as nn
from torchvision import models


class ResNet50BMIOnly(nn.Module):
    def __init__(
        self,
        hidden_units=(256, 64),
        dropout=0.2,
        frozen=True,
        fine_tune_last_block=True,
    ):
        super().__init__()

        weights = models.ResNet50_Weights.IMAGENET1K_V2
        backbone = models.resnet50(weights=weights)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        if frozen:
            for param in backbone.parameters():
                param.requires_grad = False

        if fine_tune_last_block:
            for param in backbone.layer4.parameters():
                param.requires_grad = True

        self.backbone = backbone

        layers = []
        prev = in_features

        for hidden in hidden_units:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = hidden

        layers.append(nn.Linear(prev, 1))

        self.regressor = nn.Sequential(*layers)

    def forward(self, x):
        features = self.backbone(x)
        bmi = self.regressor(features).squeeze(1)
        return bmi