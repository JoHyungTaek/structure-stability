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

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def unfreeze_last_blocks(self, n_blocks=2):
        for p in self.backbone.parameters():
            p.requires_grad = False

        # efficientnet 계열 기준
        if hasattr(self.backbone, "blocks"):
            blocks = list(self.backbone.blocks)
            for block in blocks[-n_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

        if hasattr(self.backbone, "conv_head"):
            for p in self.backbone.conv_head.parameters():
                p.requires_grad = True

        if hasattr(self.backbone, "bn2"):
            for p in self.backbone.bn2.parameters():
                p.requires_grad = True