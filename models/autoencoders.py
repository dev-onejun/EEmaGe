"""
Experiment Architectures

1. Fully-Connected Networks (class Autoencoder)
2. Conv-DeConv Networks (class ConvAutoencoder)
"""

from torch import nn


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


# https://chioni.github.io/posts/ae/
# https://sanghyu.tistory.com/184


class Autoencoders(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(Autoencoders, self).__init__()

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
        """
        out = x.view(x.size(0), -1)
        out = self.encoder(out)
        out = self.decoder(out)
        out = out.view(x.size())
        """

        eeg_features = self.encoder(eeg)
        image_features = self.encoder(image)

        eeg_out = self.eeg_decoder(eeg_features)
        image_out = self.image_decoder(image_features)
        return eeg_out, image_out


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(3, 5, kernel_size=5),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(5, 3, kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 1, kernel_size=5),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out
