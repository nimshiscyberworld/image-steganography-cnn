import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.baseline_cnn import (
    MessageEmbedder,
    Encoder,
    Decoder,
    MessageDecoder
)

# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 256
MESSAGE_BITS = 256

STEPS = 1000
LR = 1e-4

print("Device:", DEVICE)

# ============================================================
# CREATE MODEL
# ============================================================

message_embedder = MessageEmbedder(
    message_bits=MESSAGE_BITS,
    feature_channels=32
).to(DEVICE)

encoder = Encoder().to(DEVICE)
decoder = Decoder().to(DEVICE)

message_decoder = MessageDecoder(
    message_bits=MESSAGE_BITS
).to(DEVICE)

# ============================================================
# COMBINE ALL PARAMETERS
# ============================================================

parameters = list(message_embedder.parameters()) \
           + list(encoder.parameters()) \
           + list(decoder.parameters()) \
           + list(message_decoder.parameters())

optimizer = optim.Adam(parameters, lr=LR)

criterion = nn.BCEWithLogitsLoss()

# ============================================================
# ONE FIXED COVER IMAGE
# ============================================================

torch.manual_seed(42)

cover = torch.rand(
    1,
    1,
    IMAGE_SIZE,
    IMAGE_SIZE,
    device=DEVICE
)

# ============================================================
# ONE FIXED 256-BIT MESSAGE
# ============================================================

message = torch.randint(
    0,
    2,
    (1, MESSAGE_BITS),
    device=DEVICE
).float()

print("Cover shape:", cover.shape)
print("Message shape:", message.shape)

print("Original message:")
print(message[0].int().tolist())

# ============================================================
# TRAIN
# ============================================================

for step in range(1, STEPS + 1):

    optimizer.zero_grad()

    # Message -> spatial feature
    message_feature = message_embedder(message)

    # Hide message
    stego = encoder(
        cover,
        message_feature
    )

    # Decode stego
    secret_feature = decoder(stego)

    # Recover message
    recovered_logits = message_decoder(secret_feature)

    # Message loss ONLY
    loss = criterion(
        recovered_logits,
        message
    )

    loss.backward()

    optimizer.step()

    # Calculate accuracy
    with torch.no_grad():

        recovered_prob = torch.sigmoid(
            recovered_logits
        )

        recovered_bits = (
            recovered_prob >= 0.5
        ).float()

        bit_accuracy = (
            recovered_bits == message
        ).float().mean().item() * 100

    if step == 1 or step % 50 == 0:

        print(
            f"Step {step:4d} | "
            f"Loss: {loss.item():.6f} | "
            f"Bit Accuracy: {bit_accuracy:.2f}%"
        )

# ============================================================
# FINAL TEST
# ============================================================

with torch.no_grad():

    message_feature = message_embedder(message)

    stego = encoder(
        cover,
        message_feature
    )

    secret_feature = decoder(stego)

    recovered_logits = message_decoder(
        secret_feature
    )

    recovered_prob = torch.sigmoid(
        recovered_logits
    )

    recovered_bits = (
        recovered_prob >= 0.5
    ).float()

    final_accuracy = (
        recovered_bits == message
    ).float().mean().item() * 100

    exact_match = torch.equal(
        recovered_bits,
        message
    )

print("\n==============================")
print("FINAL RESULT")
print("==============================")

print(
    f"Final Bit Accuracy : "
    f"{final_accuracy:.2f}%"
)

print(
    f"Exact Message Match: "
    f"{exact_match}"
)

print("\nOriginal:")
print(message[0].int().tolist())

print("\nRecovered:")
print(recovered_bits[0].int().tolist())

print("\nD6 TEST COMPLETE")