from torch import nn
import torchsummary

from models.EEGChannelNet import EEGFeaturesExtractor
from models.base import ImageFeatureExtractor, Encoder, EEGDecoder, ImageDecoder


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


if __name__ == "__main__":
    """EEmaGeChannelNet Model"""
