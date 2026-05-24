"""
M2 — ResNet50 Multi-Task Model (Improved)
========================================
Key changes over baseline:
  1. ResNet50 V2 weights  (ImageNet accuracy 80.9% vs 76.1%)
  2. Attention-gated BMI head  (channel-wise SE block on pooled features)
  3. Gender embedding injected into BMI branch  (conditioning)
  4. Separate per-branch BN + Dropout to reduce co-adaptation
  5. Two-phase freeze API (freeze_backbone / unfreeze_layer4)
  6. Huber loss on BMI instead of MSE  (robust to outlier BMI labels)
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import ResNet50_Weights


# ─── Squeeze-Excitation (channel attention) ──────────────────────────────────
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block to re-weight feature channels."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)          # works on [B, C] as [B, C, 1]
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C]
        w = self.fc(x)          # [B, C]
        return x * w


# ─── Main Model ───────────────────────────────────────────────────────────────
class ResNet50MultiTask(nn.Module):
    """
    ResNet50 backbone → shared trunk → two heads:
      • BMI head  : regression, continuous output
      • Gender head: binary classification (male=1 / female=0)

    Gender prediction is injected as a learned embedding into the BMI head,
    acting as a conditioning signal (gender is a known BMI correlate).
    """

    def __init__(
        self,
        dropout_shared: float = 0.3,
        dropout_bmi: float = 0.4,
        dropout_gender: float = 0.3,
        hidden_shared: int = 1024,
        hidden_bmi: int = 512,
        hidden_gender: int = 256,
        gender_embed_dim: int = 16,
    ):
        super().__init__()

        # ── Backbone ────────────────────────────────────────────────────────
        backbone = tv_models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        feature_dim = backbone.fc.in_features          # 2048

        # Remove the original classification head
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        # Output: [B, 2048, 1, 1]  →  flatten to [B, 2048]

        # ── Shared trunk ────────────────────────────────────────────────────
        self.shared_trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, hidden_shared),
            nn.BatchNorm1d(hidden_shared),
            nn.GELU(),
            nn.Dropout(dropout_shared),
        )

        # ── Channel attention on shared features ────────────────────────────
        self.se = SEBlock(hidden_shared, reduction=16)

        # ── Gender head ─────────────────────────────────────────────────────
        self.gender_head = nn.Sequential(
            nn.Linear(hidden_shared, hidden_gender),
            nn.BatchNorm1d(hidden_gender),
            nn.GELU(),
            nn.Dropout(dropout_gender),
            nn.Linear(hidden_gender, 1),        # raw logit for BCEWithLogitsLoss
        )

        # Gender embedding: converts predicted gender prob → learned vector
        # Injected into BMI branch for conditioning
        self.gender_embed = nn.Linear(1, gender_embed_dim)

        # ── BMI head ────────────────────────────────────────────────────────
        self.bmi_head = nn.Sequential(
            nn.Linear(hidden_shared + gender_embed_dim, hidden_bmi),
            nn.BatchNorm1d(hidden_bmi),
            nn.GELU(),
            nn.Dropout(dropout_bmi),
            nn.Linear(hidden_bmi, hidden_bmi // 2),
            nn.BatchNorm1d(hidden_bmi // 2),
            nn.GELU(),
            nn.Dropout(dropout_bmi * 0.5),
            nn.Linear(hidden_bmi // 2, 1),      # raw BMI value
        )

        self._init_heads()

    def _init_heads(self):
        """Kaiming init on all linear layers outside the backbone."""
        for module in [self.shared_trunk, self.gender_head,
                       self.bmi_head, self.gender_embed]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    # ── Two-phase freeze API ─────────────────────────────────────────────────
    def freeze_backbone(self):
        """Phase 1: freeze all backbone weights, train heads only."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_layer4(self):
        """Phase 2: unfreeze ResNet layer4 (the last residual block group)."""
        # backbone children: 0=conv1 1=bn1 2=relu 3=maxpool
        #   4=layer1  5=layer2  6=layer3  7=layer4  8=avgpool
        layer4 = list(self.backbone.children())[7]
        for p in layer4.parameters():
            p.requires_grad = True

    def unfreeze_layer3_and_4(self):
        """Optional Phase 3: also unfreeze layer3 for deeper fine-tuning."""
        for idx in [6, 7]:
            for p in list(self.backbone.children())[idx].parameters():
                p.requires_grad = True

    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    # ── Forward ─────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        # Backbone + flatten
        feat = self.backbone(x)               # [B, 2048, 1, 1]
        feat = self.shared_trunk(feat)        # [B, hidden_shared]
        feat = self.se(feat)                  # [B, hidden_shared]  (attention)

        # Gender logit + soft prediction for conditioning
        gender_logit = self.gender_head(feat).squeeze(-1)   # [B]
        gender_prob = torch.sigmoid(gender_logit).unsqueeze(-1)  # [B, 1]
        gender_emb = self.gender_embed(gender_prob)          # [B, gender_embed_dim]

        # BMI prediction conditioned on gender embedding
        bmi_feat = torch.cat([feat, gender_emb], dim=-1)    # [B, hidden_shared + embed]
        bmi = self.bmi_head(bmi_feat).squeeze(-1)            # [B]

        return bmi, gender_logit


# ─── Loss ─────────────────────────────────────────────────────────────────────
class MultiTaskLoss(nn.Module):
    """
    Combined Huber (BMI) + BCE (gender) loss with learnable task weights.

    Using Kendall et al. (2018) homoscedastic uncertainty weighting:
        L = (1/2σ₁²) * L_bmi + log σ₁
          + (1/2σ₂²) * L_gender + log σ₂
    where log_σ² are learnable parameters.

    Start with fixed lambda for first few epochs, then switch to learned.
    """

    def __init__(self, bmi_lambda: float = 1.0, gender_lambda: float = 0.3,
                 huber_delta: float = 5.0, use_uncertainty_weighting: bool = True):
        super().__init__()
        self.huber = nn.HuberLoss(delta=huber_delta, reduction="mean")
        self.bce   = nn.BCEWithLogitsLoss()
        self.bmi_lambda    = bmi_lambda
        self.gender_lambda = gender_lambda
        self.use_uncertainty_weighting = use_uncertainty_weighting

        # Learnable log(σ²) for each task (Kendall 2018)
        # init near 0 so σ ≈ 1 at start
        self.log_sigma_bmi    = nn.Parameter(torch.zeros(1))
        self.log_sigma_gender = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        bmi_pred: torch.Tensor,
        bmi_true: torch.Tensor,
        gender_logit: torch.Tensor,
        gender_true: torch.Tensor,
    ):
        l_bmi    = self.huber(bmi_pred, bmi_true.float())
        l_gender = self.bce(gender_logit, gender_true.float())

        if self.use_uncertainty_weighting:
            # precision = exp(-log_sigma²)
            prec_bmi    = torch.exp(-self.log_sigma_bmi)
            prec_gender = torch.exp(-self.log_sigma_gender)
            total = (
                prec_bmi    * l_bmi    + self.log_sigma_bmi    +
                prec_gender * l_gender + self.log_sigma_gender
            )
        else:
            total = self.bmi_lambda * l_bmi + self.gender_lambda * l_gender

        return total, l_bmi.detach(), l_gender.detach()
