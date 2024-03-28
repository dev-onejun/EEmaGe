from torch import nn
import torchsummary

from EEGChannelNet import EEGFeaturesExtractor
from base import ImageFeaturesExtractor, EEGDecoder, ImageDecoder


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.encoder(x)
        return out


class EEmaGeChannelNet(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim1=2048, hidden_dim2=1024):
        super(EEmaGeChannelNet, self).__init__()

        self.eeg_feature_extractor = nn.Sequential(
            EEGFeaturesExtractor(), nn.Linear(500, input_dim), nn.ReLU(True)
        )
        self.image_feature_extractor = nn.Sequential(
            ImageFeaturesExtractor(), nn.Linear(2048, input_dim), nn.ReLU(True)
        )

        self.encoder = Encoder(
            input_dim,
            hidden_dim1,
            hidden_dim2,
        )

        self.eeg_decoder = EEGDecoder(hidden_dim2)
        self.image_decoder = ImageDecoder(hidden_dim2)

    def forward(self, eeg, image):
        eeg_features = self.eeg_feature_extractor(eeg)
        image_features = self.image_feature_extractor(image)

        common_eeg_features = self.encoder(eeg_features)
        common_image_features = self.encoder(image_features)

        eeg_out = self.eeg_decoder(common_eeg_features)
        image_out = self.image_decoder(common_image_features)

        return eeg_out, image_out


if __name__ == "__main__":
    torchsummary.summary(EEmaGe(), [(1, 128, 440), (3, 299, 299)])
