import torch
import torch.nn as nn


# ============================================================
# MESSAGE EMBEDDER
# ============================================================

class MessageEmbedder(nn.Module):
    """
    Converts a 256-bit secret message into a spatial feature map
    that can be combined with the cover image.
    """

    def __init__(self, message_bits=256):
        super().__init__()

        self.message_bits = message_bits

        self.fc = nn.Sequential(
            nn.Linear(message_bits, 128 * 16 * 16),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.upsample = nn.Upsample(
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        )

    def forward(self, message):

        x = self.fc(message)

        x = x.view(
            message.size(0),
            128,
            16,
            16
        )

        x = self.conv(x)

        x = self.upsample(x)

        return x


# ============================================================
# CNN BLOCK
# ============================================================

class ConvBlock(nn.Module):

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
# ENCODER
# ============================================================

class Encoder(nn.Module):

    def __init__(self):

        super().__init__()

        # Cover = 1 channel
        # Secret feature = 1 channel
        # Combined = 2 channels

        self.conv1 = ConvBlock(2, 32)

        self.conv2 = ConvBlock(32, 64)

        self.conv3 = ConvBlock(64, 64)

        self.conv4 = ConvBlock(64, 32)

        self.output = nn.Conv2d(
            32,
            1,
            kernel_size=3,
            padding=1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, cover, secret_feature):

        x = torch.cat(
            [cover, secret_feature],
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
# DECODER
# ============================================================

class Decoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = ConvBlock(1, 32)

        self.conv2 = ConvBlock(32, 64)

        self.conv3 = ConvBlock(64, 64)

        self.conv4 = ConvBlock(64, 32)

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

        secret_feature = self.output(x)

        secret_feature = self.sigmoid(
            secret_feature
        )

        return secret_feature


# ============================================================
# MESSAGE DECODER
# ============================================================

class MessageDecoder(nn.Module):
    """
    Converts the decoded spatial representation into
    256 recovered bits.
    """

    def __init__(self, message_bits=256):

        super().__init__()

        self.message_bits = message_bits

        self.pool = nn.AdaptiveAvgPool2d(
            (8, 8)
        )

        self.fc = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                1 * 8 * 8,
                256
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                256,
                message_bits
            ),

            nn.Sigmoid()
        )

    def forward(self, secret_feature):

        x = self.pool(secret_feature)

        message = self.fc(x)

        return message


# ============================================================
# COMPLETE BASELINE STEGANOGRAPHY MODEL
# ============================================================

class BaselineSteganography(nn.Module):

    def __init__(self, message_bits=256):

        super().__init__()

        self.message_embedder = MessageEmbedder(
            message_bits
        )

        self.encoder = Encoder()

        self.decoder = Decoder()

        self.message_decoder = MessageDecoder(
            message_bits
        )

    def forward(self, cover, message):

        # ----------------------------------------------------
        # 1. Convert message bits → spatial representation
        # ----------------------------------------------------

        secret_feature = self.message_embedder(
            message
        )

        # ----------------------------------------------------
        # 2. Create stego image
        # ----------------------------------------------------

        stego = self.encoder(
            cover,
            secret_feature
        )

        # ----------------------------------------------------
        # 3. Recover secret representation
        # ----------------------------------------------------

        recovered_feature = self.decoder(
            stego
        )

        # ----------------------------------------------------
        # 4. Recover original message bits
        # ----------------------------------------------------

        recovered_message = self.message_decoder(
            recovered_feature
        )

        return (
            stego,
            recovered_message
        )


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("BASELINE STEGANOGRAPHY MODEL TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = BaselineSteganography(
        message_bits=256
    )

    # --------------------------------------------------------
    # Dummy data
    # --------------------------------------------------------

    batch_size = 2

    cover = torch.rand(
        batch_size,
        1,
        256,
        256
    )

    message = torch.randint(
        0,
        2,
        (
            batch_size,
            256
        )
    ).float()

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    stego, recovered_message = model(
        cover,
        message
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\nInput:")
    print("Cover shape       :", cover.shape)
    print("Message shape     :", message.shape)

    print("\nOutput:")
    print("Stego shape       :", stego.shape)
    print(
        "Recovered message :",
        recovered_message.shape
    )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("\nTotal parameters:")
    print(total_params)

    print("\nModel test successful!")

    print("=" * 60)