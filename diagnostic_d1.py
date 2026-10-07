import torch

from models.baseline_cnn import (
    MessageEmbedder,
    Encoder,
    Decoder,
    MessageDecoder
)


# ============================================================
# D1 - MESSAGE INFLUENCE ON STEGO IMAGE
# ============================================================

print("=" * 70)
print("D1 - MESSAGE INFLUENCE ON STEGO IMAGE")
print("=" * 70)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDevice:", device)


IMAGE_SIZE = 256
MESSAGE_BITS = 256


# ------------------------------------------------------------
# Create models
# ------------------------------------------------------------

embedder = MessageEmbedder(
    message_bits=MESSAGE_BITS,
    feature_channels=64
).to(device)

encoder = Encoder().to(device)

decoder = Decoder().to(device)

message_decoder = MessageDecoder(
    message_bits=MESSAGE_BITS
).to(device)


# ------------------------------------------------------------
# Evaluation mode
# ------------------------------------------------------------

embedder.eval()
encoder.eval()
decoder.eval()
message_decoder.eval()


# ------------------------------------------------------------
# Create ONE fixed cover image
# ------------------------------------------------------------

cover = torch.rand(
    1,
    1,
    IMAGE_SIZE,
    IMAGE_SIZE,
    device=device
)


# ------------------------------------------------------------
# Message A: all zeros
# ------------------------------------------------------------

message_a = torch.zeros(
    1,
    MESSAGE_BITS,
    device=device
)


# ------------------------------------------------------------
# Message B: all ones
# ------------------------------------------------------------

message_b = torch.ones(
    1,
    MESSAGE_BITS,
    device=device
)


# ------------------------------------------------------------
# Generate stego A
# ------------------------------------------------------------

with torch.no_grad():

    feature_a = embedder(message_a)

    stego_a = encoder(
        cover,
        feature_a
    )


# ------------------------------------------------------------
# Generate stego B
# ------------------------------------------------------------

with torch.no_grad():

    feature_b = embedder(message_b)

    stego_b = encoder(
        cover,
        feature_b
    )


# ------------------------------------------------------------
# Compare message features
# ------------------------------------------------------------

feature_difference = (
    feature_a - feature_b
).abs().mean().item()


# ------------------------------------------------------------
# Compare stego images
# ------------------------------------------------------------

stego_difference = (
    stego_a - stego_b
).abs().mean().item()


stego_rms = torch.sqrt(
    ((stego_a - stego_b) ** 2).mean()
).item()


# Difference between cover and stego
cover_stego_a = torch.sqrt(
    ((stego_a - cover) ** 2).mean()
).item()

cover_stego_b = torch.sqrt(
    ((stego_b - cover) ** 2).mean()
).item()


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print("\nMessage A:")
print("Zeros:", (message_a == 0).sum().item())
print("Ones :", (message_a == 1).sum().item())


print("\nMessage B:")
print("Zeros:", (message_b == 0).sum().item())
print("Ones :", (message_b == 1).sum().item())


print("\nMessage feature difference:")
print(feature_difference)


print("\nStego A shape:")
print(stego_a.shape)


print("\nStego B shape:")
print(stego_b.shape)


print("\nMean absolute difference between Stego A and Stego B:")
print(stego_difference)


print("\nRMS difference between Stego A and Stego B:")
print(stego_rms)


print("\nRMS difference between Cover and Stego A:")
print(cover_stego_a)


print("\nRMS difference between Cover and Stego B:")
print(cover_stego_b)


# ------------------------------------------------------------
# Flip ONE bit test
# ------------------------------------------------------------

message_c = torch.zeros(
    1,
    MESSAGE_BITS,
    device=device
)

message_d = message_c.clone()

# Flip only bit 0
message_d[0, 0] = 1.0


with torch.no_grad():

    feature_c = embedder(message_c)
    feature_d = embedder(message_d)

    stego_c = encoder(
        cover,
        feature_c
    )

    stego_d = encoder(
        cover,
        feature_d
    )


one_bit_difference = torch.sqrt(
    ((stego_c - stego_d) ** 2).mean()
).item()


print("\n" + "=" * 70)
print("ONE-BIT FLIP TEST")
print("=" * 70)

print("Only bit 0 was changed.")

print("\nRMS stego difference:")
print(one_bit_difference)


print("\n" + "=" * 70)
print("D1 COMPLETE")
print("=" * 70)