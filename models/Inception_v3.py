import torch
import torch.nn as nn
from torchvision import models

inception_v3 = models.inception_v3(weights="IMAGENET1K_V1")
inception_v3.fc = nn.Identity()  # type: ignore
# inception_v3.eval()


class ImageFeaturesExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = inception_v3

    def forward(self, x):
        out = self.model(x)

        out = out.view(out.size(0), -1)

        return out
