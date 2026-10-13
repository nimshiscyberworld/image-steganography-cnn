
import torch
import torch.nn as nn
import torch.nn.functional as F


class MessageEmbedder(nn.Module):
    """Convert a 256-bit secret message into spatial features."""

    def __init__(self, message_bits=256, feature_channels=32):
        super().__init__()
        self.fc = nn.Linear(message_bits, 16 * 16)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, feature_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.feature_channels = feature_channels

    def forward(self, message):
        x = self.fc(message.float())
        x = x.view(-1, 1, 16, 16)
        x = F.interpolate(
            x, size=(256, 256), mode="bilinear",
            align_corners=False
        )
        return self.conv(x)


class CannyGuidedEncoder(nn.Module):
    """Embed message features while providing a Canny edge map."""

    def __init__(self, message_channels=32):
        super().__init__()

        # 1 cover + 1 edge map + 32 message feature channels
        self.network = nn.Sequential(
            nn.Conv2d(34, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, cover, message_features, edge_map):
        x = torch.cat(
            [cover, edge_map, message_features], dim=1
        )

        # Small residual helps preserve the cover image.
        residual = 0.05 * torch.tanh(self.network(x))
        stego = torch.clamp(cover + residual, 0.0, 1.0)

        return stego


class StegoDecoder(nn.Module):
    """Extract spatial features from the stego image."""

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, stego):
        return self.network(stego)


class MessageDecoder(nn.Module):
    """Recover the secret message as 256 logits."""

    def __init__(self, message_bits=256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, message_bits)
        )

    def forward(self, features):
        return self.network(features)


class CannyGuidedSteganography(nn.Module):
    """Complete Canny-guided image steganography model."""

    def __init__(self, message_bits=256):
        super().__init__()

        self.message_embedder = MessageEmbedder(message_bits, 32)
        self.encoder = CannyGuidedEncoder(32)
        self.decoder = StegoDecoder()
        self.message_decoder = MessageDecoder(message_bits)

    def forward(self, cover, message, edge_map):
        message_features = self.message_embedder(message)
        stego = self.encoder(cover, message_features, edge_map)
        decoded_features = self.decoder(stego)
        recovered_logits = self.message_decoder(decoded_features)

        return stego, recovered_logits
