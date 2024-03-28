"""
Based on the AE-CDNN model propsed by T. Wen et al. [1, 2].
    [1] T. Wen and Z. Zhang, "Deep Convolution Neural Network and Autoencoders-Based Unsupervised Feature Learning of EEG Signals," in IEEE Access, vol. 6, pp. 25399-25410, 2018, doi: 10.1109/ACCESS.2018.2833746.
    [2] https://github.com/bruAristimunha/Re-Deep-Convolution-Neural-Network-and-Autoencoders-Based-Unsupervised-Feature-Learning-of-EEG/tree/master, accessed in Mar. 28 2024.   
"""

from torch import nn


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

        eeg_out = self.decoder(eeg_out)  # 440, 128

        return eeg_out
