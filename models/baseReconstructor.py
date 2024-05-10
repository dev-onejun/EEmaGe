from torch import nn
from .base import (
    EEGFeatureExtractor,
    Encoder,
    ImageDecoder,
)


class BaseReconstructor(nn.Module):
    def __init__(
        self,
        eeg_channel_num=128,
        eeg_exclusion_channel_num=17,
        feature_num=8,
    ):
        super(BaseReconstructor, self).__init__()

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
