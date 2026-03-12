import torch
import torch.nn as nn
from torchvision import models


class MultiViewResNet(nn.Module):
    def __init__(self, num_classes=1):
        super(MultiViewResNet, self).__init__()

        self.backbone = models.resnet18(weights="DEFAULT")
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, views):
        f1 = self.feature_extractor(views[0]).view(views[0].size(0), -1)
        f2 = self.feature_extractor(views[1]).view(views[1].size(0), -1)

        combined = torch.cat((f1, f2), dim=1)
        return self.classifier(combined)
