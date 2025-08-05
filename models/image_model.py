import torch
import torch.nn as nn
from torchvision import models

class ImageModerationModel(nn.Module):
    def __init__(self, num_classes=5):
        super(ImageModerationModel, self).__init__()
        # Use pretrained EfficientNet-B0
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        
        # Replace final classifier
        self.efficientnet.classifier[1] = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.efficientnet.classifier[1].in_features, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.efficientnet(x)