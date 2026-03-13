import torch
import torch.nn as nn
import timm


class MultiViewClassifier(nn.Module):
    def __init__(self, model_name="efficientnet_b3", dropout=0.4):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )

        feature_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(128, 1),
        )

    def forward(self, views):
        front_feat = self.backbone(views[0])
        top_feat = self.backbone(views[1])

        fused = torch.cat([front_feat, top_feat], dim=1)
        logits = self.classifier(fused)
        return logits