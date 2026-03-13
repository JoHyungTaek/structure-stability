import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            kernel_size=(x.size(-2), x.size(-1))
        ).pow(1.0 / self.p).flatten(1)


class MultiViewClassifier(nn.Module):
    def __init__(self, model_name="efficientnet_b4", dropout=0.4):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool=""
        )

        self.pool = GeM()
        feature_dim = self.backbone.num_features

        fused_dim = feature_dim * 4  # front, top, abs diff, mul

        self.head = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, 1)
        )

    def encode(self, x):
        feat_map = self.backbone(x)
        feat = self.pool(feat_map)
        return feat

    def forward(self, views):
        front_feat = self.encode(views[0])
        top_feat = self.encode(views[1])

        diff_feat = torch.abs(front_feat - top_feat)
        mul_feat = front_feat * top_feat

        fused = torch.cat(
            [front_feat, top_feat, diff_feat, mul_feat],
            dim=1
        )
        logits = self.head(fused)
        return logits