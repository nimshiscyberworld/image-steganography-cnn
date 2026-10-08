import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. MESSAGE PREPARATION NETWORK
# ============================================================

class MessageEmbedder(nn.Module):
    """
    Converts a 256-bit message into a spatial feature map.

    Input:
        [B, 256]

    Output:
        [B, 32, 256, 256]
    """

    def __init__(self, message_bits=256, feature_channels=32):
        super().__init__()

        self.message_bits = message_bits

        # 256 bits -> 16 x 16 spatial representation
        self.fc = nn.Linear(
            message_bits,
            16 * 16
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, message):

        # [B, 256] -> [B, 256]
        x = self.fc(message)

        # [B, 256] -> [B, 1, 16, 16]
        x = x.view(
            x.size(0),
            1,
            16,
            16
        )

        # 16 -> 32
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        x = self.conv1(x)

        # 32 -> 64
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        x = self.conv2(x)

        # 64 -> 128
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        x = self.conv3(x)

        # 128 -> 256
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        x = self.conv4(x)

        return x


# ============================================================
# 2. HIDING NETWORK / ENCODER
# ============================================================

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        # Cover = 1 channel
        # Message features = 32 channels
        # Total = 33 channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(33, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Produces the stego image
        self.output = nn.Conv2d(
            32,
            1,
            3,
            padding=1
        )

    def forward(self, cover, message_feature):

        # [B,1,256,256] + [B,32,256,256]
        # -> [B,33,256,256]

        x = torch.cat(
            [cover, message_feature],
            dim=1
        )

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Predict hidden residual
        residual = self.output(x)

        # Keep modification bounded
        residual = 0.05 * torch.tanh(residual)

        # Add small message-dependent residual to cover
        stego = cover + residual

        # Keep image in valid [0,1] range
        stego = torch.clamp(
            stego,
            0.0,
            1.0
        )

        return stego


# ============================================================
# 3. REVEAL / DECODER NETWORK
# ============================================================

class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, stego):

        x = self.conv1(stego)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x


# ============================================================
# 4. MESSAGE DECODER
# ============================================================

class MessageDecoder(nn.Module):

    def __init__(self, message_bits=256):

        super().__init__()

        # Decoder output:
        # [B,128,16,16]
        #
        # Flatten:
        # 128 * 16 * 16 = 32768

        self.fc = nn.Sequential(

            nn.Linear(
                128 * 16 * 16,
                1024
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                1024,
                message_bits
            )

            # IMPORTANT:
            # No Sigmoid here.
            #
            # We will use:
            # BCEWithLogitsLoss()
        )

    def forward(self, x):

        x = x.view(
            x.size(0),
            -1
        )

        logits = self.fc(x)

        return logits


# ============================================================
# 5. COMPLETE MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CNN IMAGE STEGANOGRAPHY - NEW ARCHITECTURE TEST")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)

    # --------------------------------------------------------
    # Create models
    # --------------------------------------------------------

    embedder = MessageEmbedder().to(device)
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    message_decoder = MessageDecoder().to(device)

    # --------------------------------------------------------
    # Dummy input
    # --------------------------------------------------------

    batch_size = 2

    cover = torch.rand(
        batch_size,
        1,
        256,
        256,
        device=device
    )

    message = torch.randint(
        0,
        2,
        (batch_size, 256),
        device=device
    ).float()

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    message_feature = embedder(message)

    stego = encoder(
        cover,
        message_feature
    )

    decoder_feature = decoder(
        stego
    )

    recovered_logits = message_decoder(
        decoder_feature
    )

    # --------------------------------------------------------
    # Print shapes
    # --------------------------------------------------------

    print("\nShapes:")
    print("----------------------------------------")

    print(
        "Cover:",
        cover.shape
    )

    print(
        "Message:",
        message.shape
    )

    print(
        "Message feature:",
        message_feature.shape
    )

    print(
        "Stego:",
        stego.shape
    )

    print(
        "Decoder feature:",
        decoder_feature.shape
    )

    print(
        "Recovered logits:",
        recovered_logits.shape
    )

    # --------------------------------------------------------
    # Convert logits to probabilities
    # --------------------------------------------------------

    recovered_probability = torch.sigmoid(
        recovered_logits
    )

    print(
        "Recovered probability:",
        recovered_probability.shape
    )

    # --------------------------------------------------------
    # Test BCEWithLogitsLoss
    # --------------------------------------------------------

    loss = nn.BCEWithLogitsLoss()(
        recovered_logits,
        message
    )

    print(
        "\nInitial BCEWithLogitsLoss:",
        loss.item()
    )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    total_parameters = sum(
        p.numel()
        for model in [
            embedder,
            encoder,
            decoder,
            message_decoder
        ]
        for p in model.parameters()
    )

    print(
        "\nTotal parameters:",
        f"{total_parameters:,}"
    )

    print("\n" + "=" * 70)
    print("MODEL TEST SUCCESSFUL")
    print("=" * 70)
