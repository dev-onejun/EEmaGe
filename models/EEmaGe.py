import torch
from torch import nn

from models import Base, EEmaGeChannelNet


class EEmaGeClassifier(nn.Module):
    def __init__(self, **kwargs):
        super(EEmaGeClassifier, self).__init__()
        model_type, eeg_exclusion_channel_num, pretrained_model_path = (
            kwargs["model_type"],
            kwargs["eeg_exclusion_channel_num"],
            kwargs["pretrained_model_path"],
        )

        assert model_type in (
            "base",
            "channelnet",
        ), "The type of the model only supports base or channelnet"

        model = (
            Base(128, eeg_exclusion_channel_num, 8)
            if model_type == "Base"
            else EEmaGeChannelNet(eeg_exclusion_channel_num=eeg_exclusion_channel_num)
        )
        model.load_state_dict(torch.load(pretrained_model_path))

        self.eeg_encoder = nn.Sequential(
            model.eeg_feature_extractor,
            model.encoder,
        )

        for name, param in self.eeg_encoder.named_parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(1024, 1996),
            nn.ReLU(),
            nn.Linear(1996, 1996),
        )

    def forward(self, x):
        out = self.eeg_encoder(x)
        out = self.classifier(out)

        return out
