import torch
import torch.nn as nn
import timm


class MultiViewEfficientNet(nn.Module):
    def __init__(self, model_name="efficientnet_b0", dropout=0.3, num_classes=1):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg"
        )

        feature_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, views):
        f1 = self.backbone(views[0])
        f2 = self.backbone(views[1])

        combined = torch.cat([f1, f2], dim=1)
        out = self.classifier(combined)
        return out