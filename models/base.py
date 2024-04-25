"""
EEmaGe Base Model

* Based on the AE-CDNN model propsed by T. Wen et al. [1, 2].

[1] T. Wen and Z. Zhang, "Deep Convolution Neural Network and Autoencoders-Based Unsupervised Feature Learning of EEG Signals," in IEEE Access, vol. 6, pp. 25399-25410, 2018, doi: 10.1109/ACCESS.2018.2833746.
[2] https://github.com/bruAristimunha/Re-Deep-Convolution-Neural-Network-and-Autoencoders-Based-Unsupervised-Feature-Learning-of-EEG/tree/master, accessed in Mar. 28 2024.
"""

from torch import nn, Tensor
from torchvision import models


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),  # 4096 -> 2048
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),  # 2048 -> 1024
        )

    def forward(self, x):
        out = self.encoder(x)
        return out


class EEGDecoder(nn.Module):
    def __init__(
        self, eeg_channel_num, eeg_exclusion_channel_num, feature_num, input_dim
    ):
        super(EEGDecoder, self).__init__()

        target_eeg_channel_num = eeg_channel_num - eeg_exclusion_channel_num

        self.linear = nn.Sequential(
            nn.Linear(
                input_dim, target_eeg_channel_num * feature_num * 2 * 2 * 55, bias=False
            ),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                target_eeg_channel_num * feature_num * 2 * 2,
                target_eeg_channel_num * feature_num * 2,
                1,
            ),  # (55, )
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # (110, )
            nn.ConvTranspose1d(
                target_eeg_channel_num * feature_num * 2,
                target_eeg_channel_num * feature_num,
                1,
            ),  # (110, )
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # (220, )
            nn.ConvTranspose1d(
                target_eeg_channel_num * feature_num,
                target_eeg_channel_num,
                1,
            ),  # (220, 128)
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 440, 128
            nn.ConvTranspose1d(target_eeg_channel_num, eeg_channel_num, 1),  # 440, 128
            nn.Linear(440, 440),
        )

        self.eeg_channel_num = target_eeg_channel_num
        self.feature_num = feature_num

    def forward(self, eeg_x):
        eeg_out = self.linear(eeg_x)
        eeg_out = eeg_out.view(-1, self.eeg_channel_num * self.feature_num * 2 * 2, 55)
        eeg_out = self.decoder(eeg_out)  # 440, 128
        return eeg_out


class ImageDecoder(nn.Module):
    def __init__(self):
        super(ImageDecoder, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(1024, 8 * 8 * 2048, bias=False),
        )
        self.image_decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 1),  # 8, 8, 1024
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 9),  # 16, 16, 512
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 17),  # 32, 32, 256
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 33),  # 64, 64, 128
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 65),  # 128, 128, 64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 129),  # 256, 256, 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 44),  # 299, 299, 3
            nn.Linear(299, 299),
        )

    def forward(self, image_x):
        image_out = self.linear(image_x)  # 2048
        image_out = image_out.view(-1, 2048, 8, 8)  # 8, 8, 2048
        image_out = self.image_decoder(image_out)  # 299, 299, 3
        return image_out


class EEGFeatureExtractor(nn.Module):
    def __init__(
        self, eeg_channel_num, eeg_exclusion_channel_num, feature_num, encoder_input_dim
    ):
        super(EEGFeatureExtractor, self).__init__()

        target_eeg_channel_num = eeg_channel_num - eeg_exclusion_channel_num

        self.embedding_dim = target_eeg_channel_num * feature_num * 2 * 2 * 55

        self.eeg_feature_extractor = nn.Sequential(
            nn.Conv1d(
                eeg_channel_num,
                target_eeg_channel_num * feature_num,
                1,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 220, 2048
            nn.Conv1d(
                target_eeg_channel_num * feature_num,
                target_eeg_channel_num * feature_num * 2,
                1,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 110, 4096
            nn.Conv1d(
                target_eeg_channel_num * feature_num * 2,
                target_eeg_channel_num * feature_num * 2 * 2,
                1,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 55, 8192
        )

        self.linear = nn.Sequential(
            nn.Linear(self.embedding_dim, encoder_input_dim, bias=False),
        )

    def forward(self, eeg):
        eeg_features = self.eeg_feature_extractor(eeg)
        eeg_features = eeg_features.view(-1, self.embedding_dim)
        eeg_features = self.linear(eeg_features)
        return eeg_features


class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super(ImageFeatureExtractor, self).__init__()

        self.image_feature_extractor = models.inception_v3(weights="IMAGENET1K_V1")
        self.image_feature_extractor.fc = nn.Identity()
        self.linear = nn.Sequential(
            nn.Linear(2048, 4096, bias=False),
        )

        for parameters in self.image_feature_extractor.parameters():
            parameters.requires_grad = False

    def forward(self, image):
        image_features = self.image_feature_extractor(image)  # 1, 1, 2048
        if type(image_features) == Tensor:
            image_features = image_features.view(image_features.size(0), -1)
        else:
            image_features = image_features.logits.view(
                image_features.logits.size(0), -1
            )  # 2048
        image_features = self.linear(image_features)
        return image_features


class Base(nn.Module):
    def __init__(
        self, eeg_channel_num=128, eeg_exclusion_channel_num=17, feature_num=8
    ):
        super(Base, self).__init__()

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


if __name__ == "__main__":
    """
    (440, 128) ->[conv1d] (440, 2048) ->[maxpooling] (220, 2048)        [ x 16 해놨는데, x 8로? ]
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
    (440, 128) ->[] (440, 128*8 = 1024) ->[maxpooling] (220, 1024)
            ->[conv1d] (220, 128*8*2 = 2048) ->[] (110, 2048)
            ->[conv1d] (110, 128*8*2*2 = 4096) ->[] (55, 4096)
            ->[flatten] (225280) ->[Dense] (4096)
    """

    """
    (299, 299, 3) ->[InceptionV3 ] (1, 1, 2048)

            # Common Encoder
            ->[Dense] (4096) ->[Dense] (2048) ->[Dense] (1024)

            # Image Decoder
            ->[Dense] (8 * 8 * 2048) ->[Reshape] (8, 8, 2048)
            ->[Deconv] (8, 8, 1024)
            ->[Deconv] (16, 16, 512)
            ->[Deconv] (32, 32, 256)
            ->[Deconv] (64, 64, 128)
            ->[Deconv] (128, 128, 64)
            ->[Deconv] (256, 256, 32)
            ->[Deconv] (299, 299, 3)
    """
