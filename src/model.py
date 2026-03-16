from __future__ import annotations

import torch
import torch.nn as nn
import timm


class MultiViewClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        pretrained: bool = True,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features
        hidden_dim = max(256, feat_dim // 2)

        self.head = nn.Sequential(
            nn.Linear(feat_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)

    def forward(self, front: torch.Tensor, top: torch.Tensor) -> torch.Tensor:
        front_feat = self.encode(front)
        top_feat = self.encode(top)
        diff_feat = torch.abs(front_feat - top_feat)
        prod_feat = front_feat * top_feat
        fused = torch.cat([front_feat, top_feat, diff_feat, prod_feat], dim=1)
        logits = self.head(fused).squeeze(1)
        return logits

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True
