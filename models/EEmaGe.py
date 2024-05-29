import torch
from torch import nn

from .modules import (
    Encoder,
    EEGDecoder,
    ImageDecoder,
    EEGFeatureExtractor,
    ImageFeatureExtractor,
)
from .EEGChannelNet import EEGFeaturesExtractor


class EEmaGeBase(nn.Module):
    def __init__(
        self, eeg_channel_num=128, eeg_exclusion_channel_num=17, feature_num=8
    ):
        super(EEmaGeBase, self).__init__()

        self.ENCODER_INPUT_DIM = 4096

        self.eeg_feature_extractor = EEGFeatureExtractor(
            eeg_channel_num,
            eeg_exclusion_channel_num,
            feature_num,
            self.ENCODER_INPUT_DIM,
        )
        self.image_feature_extractor = ImageFeatureExtractor()

        self.encoder = Encoder(
            self.ENCODER_INPUT_DIM,
            self.ENCODER_INPUT_DIM // 2,
            (self.ENCODER_INPUT_DIM // 2) // 2,
        )

        self.eeg_decoder = EEGDecoder(
            eeg_channel_num,
            eeg_exclusion_channel_num,
            feature_num,
            (self.ENCODER_INPUT_DIM // 2) // 2,
        )
        self.image_decoder = ImageDecoder()

    def forward(self, eeg, image):
        eeg_features = self.eeg_feature_extractor(eeg)  # 4096
        image_features = self.image_feature_extractor(image)  # 4096

        eeg_x = self.encoder(eeg_features)  # 1024
        image_x = self.encoder(image_features)  # 1024

        eeg_out = self.eeg_decoder(eeg_x)  # 440, 128
        image_out = self.image_decoder(image_x)  # 299, 299, 3

        return eeg_out, image_out


class EEmaGeChannelNet(nn.Module):
    def __init__(
        self,
        input_dim=4096,
        hidden_dim1=2048,
        hidden_dim2=1024,
        eeg_exclusion_channel_num=0,
    ):
        super(EEmaGeChannelNet, self).__init__()

        self.eeg_feature_extractor = nn.Sequential(
            EEGFeaturesExtractor(), nn.Linear(500, input_dim), nn.ReLU(True)
        )
        self.image_feature_extractor = ImageFeatureExtractor()

        self.encoder = Encoder(
            input_dim,
            hidden_dim1,
            hidden_dim2,
        )

        self.eeg_decoder = EEGDecoder(
            128,
            eeg_exclusion_channel_num,
            8,
            hidden_dim2,
        )
        self.image_decoder = ImageDecoder()

    def forward(self, eeg, image):
        eeg_features = self.eeg_feature_extractor(eeg)
        image_features = self.image_feature_extractor(image)

        common_eeg_features = self.encoder(eeg_features)
        common_image_features = self.encoder(image_features)

        eeg_out = self.eeg_decoder(common_eeg_features)
        image_out = self.image_decoder(common_image_features)

        return eeg_out, image_out


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
            EEmaGeBase(128, eeg_exclusion_channel_num, 8)
            if model_type == "base"
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
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 40),
        )

    def forward(self, x):
        out = self.eeg_encoder(x)
        out = self.classifier(out)

        return out


class EEmaGeBaseReconstructor(nn.Module):
    def __init__(
        self,
        eeg_channel_num=128,
        eeg_exclusion_channel_num=17,
        feature_num=8,
    ):
        super(EEmaGeBaseReconstructor, self).__init__()

        self.ENCODER_INPUT_DIM = 4096

        self.eeg_feature_extractor = EEGFeatureExtractor(
            eeg_channel_num,
            eeg_exclusion_channel_num,
            feature_num,
            self.ENCODER_INPUT_DIM,
        )

        self.encoder = Encoder(
            self.ENCODER_INPUT_DIM,
            self.ENCODER_INPUT_DIM // 2,
            (self.ENCODER_INPUT_DIM // 2) // 2,
        )

        self.image_decoder = ImageDecoder()

    def forward(self, eeg):
        eeg_features = self.eeg_feature_extractor(eeg)
        latent_vector = self.encoder(eeg_features)
        image_out = self.image_decoder(latent_vector)

        return image_out


class EEmaGeChannelNetReconstructor(nn.Module):
    def __init__(
        self,
        input_dim=4096,
        hidden_dim1=2048,
        hidden_dim2=1024,
    ):
        super(EEmaGeChannelNetReconstructor, self).__init__()

        self.eeg_feature_extractor = nn.Sequential(
            EEGFeaturesExtractor(), nn.Linear(500, input_dim), nn.ReLU(True)
        )

        self.encoder = Encoder(
            input_dim,
            hidden_dim1,
            hidden_dim2,
        )

        self.image_decoder = ImageDecoder()

    def forward(self, eeg):
        eeg_features = self.eeg_feature_extractor(eeg)
        latent_vector = self.encoder(eeg_features)
        image_out = self.image_decoder(latent_vector)

        return image_out
