"""
EEmaGe Base Model

* Based on the AE-CDNN model propsed by T. Wen et al. [1, 2].

[1] T. Wen and Z. Zhang, "Deep Convolution Neural Network and Autoencoders-Based Unsupervised Feature Learning of EEG Signals," in IEEE Access, vol. 6, pp. 25399-25410, 2018, doi: 10.1109/ACCESS.2018.2833746.
[2] https://github.com/bruAristimunha/Re-Deep-Convolution-Neural-Network-and-Autoencoders-Based-Unsupervised-Feature-Learning-of-EEG/tree/master, accessed in Mar. 28 2024.
"""

from torch import nn, reshape
from torch import rand


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),  # 4096 -> 2048
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),  # 2048 -> 1024
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.encoder(x)
        return out


class EEGDecoder(nn.Module):
    def __init__(self, input_dim):
        super(EEGDecoder, self).__init__()

        self.linear = nn.Linear(input_dim, 110)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(8192, 4096, 3),  # 55, 4096
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 110, 4096
            nn.ConvTranspose1d(4096, 2048, 3),  # 110, 2048
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 220, 2048
            nn.ConvTranspose1d(2048, 128, 3),  # 220, 128
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 440, 128
            nn.ConvTranspose1d(128, 128, 3),  # 440, 128
            nn.Sigmoid(),
        )

    def forward(self, eeg_x):
        # eeg_out = self.linear(eeg_x)  # 450560
        eeg_out = eeg_x.view(-1, 8192, 55)
        print(eeg_out.shape)

        eeg_out = self.decoder(eeg_out)  # 440, 128
        return eeg_out


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()

        self.eeg_feature_extractor = nn.Sequential(nn.conv1d(128, 2048))
        self.image_feature_extractor = nn.Sequential()

        self.encoder = Encoder(4096, 2048, 1024)

        self.eeg_decoder = EEGDecoder(1024)
        self.image_decoder = ImageDecoder()

    def forward(self, eeg, image):
        """eeg = (440, 128) / image = (299, 299, 3)"""

        eeg_features = self.eeg_preprocessing_block(eeg)  # 4096
        image_features = self.image_preprocessing_block(image)  # 4096

        eeg_x = self.encoder(eeg_features)  # 1024
        image_x = self.encoder(image_features)  # 1024

        eeg_out = self.eeg_decoder(eeg_x)  # 440, 128
        image_out = self.image_decoder(image_x)  # 299, 299, 3

        return eeg_out, image_out

        """
        (440, 128) ->[conv1d] (440, 2048) ->[maxpooling] (220, 2048)
                ->[conv1d] (220, 4096) ->[maxpooling] (110, 4096)
                ->[conv1d] (110, 8192) ->[maxpooling] (55, 8192)
                ->[flatten] (450560) ->[Dense] (4096)

                # Common Encoder
                ->[Dense] (4096) ->[Dense] (2048) ->[Dense] (1024)

                # EEG Decoder
                ->[Dense] (450560) ->[reshape] (55, 8192)

                ->[Deconv] (55, 4096) ->[Upsampling] (110, 4096)
                ->[Deconv] (110, 2048) ->[Upsampling] (220, 2048)
                ->[Deconv] (220, 128) ->[Upsampling] (440, 128)
                ->[Deconv] (440, 128)
        """

        """
        (299, 299, 3) ->[InceptionV3 ] (1, 1, 2048)

                # Common Encoder
                ->[Dense] (4096) ->[Dense] (2048) ->[Dense] (1024)

                # Image Decoder
                ->[Dense] (2048) ->[] ()
                ->[Deconv] (8, 8, 1024)
                ->[Deconv] (16, 16, 512)
                ->[Deconv] (32, 32, 256)
                ->[Deconv] (64, 64, 128)
                ->[Deconv] (128, 128, 64)
                ->[Deconv] (256, 256, 32)
                ->[Deconv] (299, 299, 3)
        """


if __name__ == "__main__":
    decoder = EEGDecoder(4096)

    test_x = rand((1, 1, 450560))

    decoder(test_x)
