import torch
import torch.nn as nn

inception_v3 = torch.hub.load("pytorch/vision:v0.10.0", "inception_v3", pretrained=True)


class ImageFeaturesExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = torch.nn.Sequential(*(list(inception_v3.children())[:-1]))

    def forward(self, x):
        out = self.model(x)

        out = out.view(-1)

        return out
