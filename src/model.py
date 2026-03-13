import torch
import torch.nn as nn
import timm


class MultiViewB3(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )

        feat_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(feat_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(128, 1),
        )

    def forward(self, views):
        f1 = self.backbone(views[0])
        f2 = self.backbone(views[1])
        x = torch.cat([f1, f2], dim=1)
        return self.head(x)