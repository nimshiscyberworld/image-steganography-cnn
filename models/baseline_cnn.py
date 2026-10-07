import torch
import torch.nn as nn


# ============================================================
# MESSAGE EMBEDDER
# ============================================================
# ============================================================
# MESSAGE EMBEDDER
# HiDDeN-inspired spatial message representation
# ============================================================

class MessageEmbedder(nn.Module):
    def __init__(self, message_bits=256, feature_channels=64):
        super().__init__()

        self.feature_channels = feature_channels

        self.fc = nn.Sequential(
            nn.Linear(message_bits, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, feature_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, message):

        # [B, 256] -> [B, 64]
        x = self.fc(message)

        # [B, 64, 1, 1]
        x = x.unsqueeze(-1).unsqueeze(-1)

        # [B, 64, 256, 256]
        x = x.expand(-1, -1, 256, 256)

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

# ============================================================
# ENCODER
# ============================================================

class Encoder(nn.Module):

    def __init__(self):

        super().__init__()

        # Cover = 1 channel
        # Secret feature = 32 channels
        # Combined = 33 channels

        self.conv1 = ConvBlock(33, 32)

        self.conv2 = ConvBlock(32, 64)

        self.conv3 = ConvBlock(64, 64)

        self.conv4 = ConvBlock(64, 32)

        # Stego image must be 1 channel

        self.output = nn.Conv2d(
            32,
            1,
            kernel_size=3,
            padding=1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, cover, secret_feature):

        # Combine cover image and message feature

        x = torch.cat(
            [cover, secret_feature],
            dim=1
        )

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x = self.conv4(x)

        # Generate stego image

        stego = self.output(x)

        stego = self.sigmoid(stego)

        return stego

# ============================================================
# DECODER
# ============================================================

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

        # Output 32-channel recovered feature

        self.output = nn.Conv2d(
            32,
            32,
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
# ============================================================
# ============================================================
# MESSAGE DECODER
# HiDDeN-inspired message recovery
# ============================================================

class MessageDecoder(nn.Module):
    """
    Converts the decoded spatial feature representation
    into the original 256-bit message.

    Keeps more spatial information before message reconstruction.
    """

    def __init__(self, message_bits=256):
        super().__init__()

        # Reduce 256x256 feature map to 8x8
        self.pool = nn.AdaptiveAvgPool2d((8, 8))

        # 32 channels x 8 x 8 = 2048 features
        self.fc = nn.Sequential(
            nn.Linear(32 * 8 * 8, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, message_bits),
            nn.Sigmoid()
        )

    def forward(self, secret_feature):

        # Input: [B, 32, 256, 256]
        x = self.pool(secret_feature)

        # [B, 32, 8, 8] -> [B, 2048]
        x = x.view(x.size(0), -1)

        # [B, 256]
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
