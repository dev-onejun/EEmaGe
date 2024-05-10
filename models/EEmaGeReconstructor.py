from torch import nn

from models.EEGChannelNet import EEGFeaturesExtractor
from models.base import Encoder, ImageDecoder


class EEmaGeReconstructor(nn.Module):
    def __init__(
        self,
        input_dim=4096,
        hidden_dim1=2048,
        hidden_dim2=1024,
    ):
        super(EEmaGeReconstructor, self).__init__()

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


if __name__ == "__main__":
    """EEmaGeReconstructor Model"""
