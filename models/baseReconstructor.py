from torch import nn
from models.base import (
    EEGFeatureExtractor,
    Encoder,
    ImageDecoder,
)


class BaseReconstructor(nn.Module):
    def __init__(
        self,
        input_dim=4096,
        hidden_dim1=2048,
        hidden_dim2=1024,
        eeg_channel_num=128,
        eeg_exclusion_channel_num=17,
        feature_num=8,
    ):
        super(BaseReconstructor, self).__init__()

        self.eeg_feature_extractor = EEGFeatureExtractor(
            eeg_channel_num,
            eeg_exclusion_channel_num,
            feature_num,
            input_dim,
        )

        # XXX: 이게 맞지 않나 싶어요...
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
