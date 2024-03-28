from torch import nn


class ImageDecoder(nn.Module):
    def __init__(self, input_dim):
        super(ImageDecoder, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(input_dim, 8 * 8 * 2048),
            nn.ReLU(),
        )
        self.image_decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 1),  # 8, 8, 1024
            nn.ConvTranspose2d(1024, 512, 9),  # 16, 16, 512
            nn.ConvTranspose2d(512, 256, 17),  # 32, 32, 256
            nn.ConvTranspose2d(256, 128, 33),  # 64, 64, 128
            nn.ConvTranspose2d(128, 64, 65),  # 128, 128, 64
            nn.ConvTranspose2d(64, 32, 129),  # 256, 256, 32
            nn.ConvTranspose2d(32, 3, 44),  # 299, 299, 3
        )

    def forward(self, image_x):
        image_out = self.linear(image_x)  # 2048
        image_out = image_out.view(-1, 2048, 8, 8)  # 8, 8, 2048
        image_out = self.image_decoder(image_out)  # 299, 299, 3
        return image_out
