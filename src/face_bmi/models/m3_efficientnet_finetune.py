import torch
import torch.nn as nn
from torchvision import models


class EfficientNetBMIRegressor(nn.Module):
    def __init__(
        self,
        hidden_units=(512, 128),
        dropout=0.2,
        unfreeze_last_n_blocks=4,
        backbone_name="efficientnet_b0",
        use_gender=True,
    ):
        super().__init__()

        self.use_gender = use_gender

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

        gender_dim = 16 if use_gender else 0

        if use_gender:
            self.gender_embedding = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
            )

        self.regressor = nn.Sequential(
            nn.Linear(in_features + gender_dim, hidden_units[0]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_units[1], 1),
        )

    def forward(self, x, gender=None):
        features = self.backbone(x)

        if self.use_gender:
            if gender is None:
                raise ValueError("Gender input is required when use_gender=True")

            gender = gender.float().view(-1, 1)
            gender_features = self.gender_embedding(gender)
            features = torch.cat([features, gender_features], dim=1)

        bmi = self.regressor(features).squeeze(1)
        return bmi