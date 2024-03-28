from torch import nn
from models.EEGChannelNet import EEGFeaturesExtractor


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
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(EEmaGe, self).__init__()

        self.eeg_feature_extractor = EEGFeaturesExtractor()

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
        image_features = image

        common_eeg_features = self.encoder(eeg_features)
        common_image_features = self.encoder(image_features)

        eeg_out = self.eeg_decoder(common_eeg_features)
        image_out = self.image_decoder(common_image_features)
        return eeg_out, image_out
