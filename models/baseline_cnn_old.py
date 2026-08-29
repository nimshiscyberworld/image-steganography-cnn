import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Basic convolutional block:
    Conv -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# BASELINE STEGANOGRAPHY ENCODER
# ============================================================

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        # Input:
        # Cover image = 1 channel
        # Secret     = 1 channel
        #
        # Total input = 2 channels

        self.conv1 = ConvBlock(2, 32)

        self.conv2 = ConvBlock(32, 64)

        self.conv3 = ConvBlock(64, 64)

        self.conv4 = ConvBlock(64, 32)

        # Output = 1-channel stego image
        self.output = nn.Conv2d(
            32,
            1,
            kernel_size=3,
            padding=1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, cover, secret):

        # Combine cover image and secret representation
        x = torch.cat(
            [cover, secret],
            dim=1
        )

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        stego = self.output(x)

        stego = self.sigmoid(stego)

        return stego


# ============================================================
# BASELINE STEGANOGRAPHY DECODER
# ============================================================

class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = ConvBlock(1, 32)

        self.conv2 = ConvBlock(32, 64)

        self.conv3 = ConvBlock(64, 64)

        self.conv4 = ConvBlock(64, 32)

        # Output secret representation
        self.output = nn.Conv2d(
            32,
            1,
            kernel_size=3,
            padding=1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, stego):

        x = self.conv1(stego)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        secret = self.output(x)

        secret = self.sigmoid(secret)

        return secret


# ============================================================
# COMPLETE BASELINE MODEL
# ============================================================

class BaselineSteganography(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = Encoder()

        self.decoder = Decoder()

    def forward(self, cover, secret):

        stego = self.encoder(
            cover,
            secret
        )

        recovered_secret = self.decoder(
            stego
        )

        return stego, recovered_secret


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    print("Testing Baseline CNN...")

    model = BaselineSteganography()

    # Example batch
    batch_size = 2

    cover = torch.rand(
        batch_size,
        1,
        256,
        256
    )

    secret = torch.rand(
        batch_size,
        1,
        256,
        256
    )

    stego, recovered = model(
        cover,
        secret
    )

    print("\nModel test successful!")

    print("Cover shape          :", cover.shape)
    print("Secret shape         :", secret.shape)
    print("Stego shape          :", stego.shape)
    print("Recovered shape      :", recovered.shape)

    print("\nTotal parameters:")

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(total_params)