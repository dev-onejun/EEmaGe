from torch import nn
from EEGChannelNet import EEGFeaturesExtractor
from Inception_v3 import ImageFeaturesExtractor


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


class Decoder(nn.Module):
    def __init__(self, hidden_dim2, hidden_dim1, input_dim):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.decoder(x)
        return out


class EEmaGe(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim1=2048, hidden_dim2=1024):
        super(EEmaGe, self).__init__()

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

        self.eeg_decoder = Decoder(
            hidden_dim2,
            hidden_dim1,
            input_dim,
        )
        self.image_decoder = Decoder(
            hidden_dim2,
            hidden_dim1,
            input_dim,
        )

    def forward(self, eeg, image):
        eeg_features = self.eeg_feature_extractor(eeg)
        image_features = self.image_feature_extractor(image)

        common_eeg_features = self.encoder(eeg_features)
        common_image_features = self.encoder(image_features)

        eeg_out = self.eeg_decoder(common_eeg_features)
        image_out = self.image_decoder(common_image_features)

        return eeg_out, image_out
