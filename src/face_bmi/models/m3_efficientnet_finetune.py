import torch
import torch.nn as nn
from torchvision import models


class EfficientNetBMIRegressor(nn.Module):
    def __init__(
        self,
        hidden_units=(256, 64),
        dropout=0.4,
        unfreeze_last_n_blocks=2,
        backbone_name="efficientnet_b0",
    ):
        super().__init__()

        if backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            backbone = models.efficientnet_b0(weights=weights)
        elif backbone_name == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
            backbone = models.efficientnet_b3(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        for param in backbone.parameters():
            param.requires_grad = False

        if unfreeze_last_n_blocks > 0:
            for block in backbone.features[-unfreeze_last_n_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True

        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

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