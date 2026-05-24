import torch
import torch.nn as nn
from torchvision import models


class ResNet50MultiTaskBMI(nn.Module):
    def __init__(
        self,
        hidden_units=(256, 64),
        dropout=0.3,
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

        shared_layers = []
        prev = in_features

        for hidden in hidden_units:
            shared_layers.append(nn.Linear(prev, hidden))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(dropout))
            prev = hidden

        self.shared_head = nn.Sequential(*shared_layers)

        self.bmi_head = nn.Linear(prev, 1)
        self.gender_head = nn.Linear(prev, 2)

    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared_head(features)

        bmi = self.bmi_head(shared).squeeze(1)
        gender_logits = self.gender_head(shared)

        return {
            "bmi": bmi,
            "gender_logits": gender_logits,
        }