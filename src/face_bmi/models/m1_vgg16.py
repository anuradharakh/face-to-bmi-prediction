import torch
import torch.nn as nn
from torchvision import models


class VGG16BMIRegressor(nn.Module):
    def __init__(self, hidden_units=(256, 64), dropout=0.3, frozen=True):
        super().__init__()

        weights = models.VGG16_Weights.IMAGENET1K_V1
        self.backbone = models.vgg16(weights=weights)

        if frozen:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        in_features = self.backbone.classifier[0].in_features

        layers = []
        prev = in_features

        for hidden in hidden_units:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = hidden

        layers.append(nn.Linear(prev, 1))

        self.backbone.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.backbone(x).squeeze(1)