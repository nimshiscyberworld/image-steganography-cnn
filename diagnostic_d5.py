import torch
import torch.nn.functional as F

from models.baseline_cnn import (
    MessageEmbedder,
    Encoder,
    Decoder,
    MessageDecoder
)


# ============================================================
# D5 - MESSAGE ONLY TRAINING
# ============================================================

print("=" * 70)
print("D5 - MESSAGE ONLY TRAINING")
print("=" * 70)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

BATCH_SIZE = 8
IMAGE_SIZE = 256
MESSAGE_BITS = 256

STEPS = 2000
LEARNING_RATE = 1e-4


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
# Optimizer
# ------------------------------------------------------------

optimizer = torch.optim.Adam(
    list(embedder.parameters())
    + list(encoder.parameters())
    + list(decoder.parameters())
    + list(message_decoder.parameters()),
    lr=LEARNING_RATE
)


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

print("\nStarting message-only training...")
print("Image loss = 0")
print("Message loss = 1")


for step in range(1, STEPS + 1):

    # --------------------------------------------------------
    # Random cover
    # --------------------------------------------------------

    cover = torch.rand(
        BATCH_SIZE,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device
    )


    # --------------------------------------------------------
    # Fresh random 256-bit message
    # --------------------------------------------------------

    message = torch.randint(
        0,
        2,
        (BATCH_SIZE, MESSAGE_BITS),
        device=device
    ).float()


    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    message_feature = embedder(message)

    stego = encoder(
        cover,
        message_feature
    )

    secret_feature = decoder(stego)

    recovered = message_decoder(secret_feature)


    # --------------------------------------------------------
    # MESSAGE LOSS ONLY
    # --------------------------------------------------------

    loss = F.binary_cross_entropy(
        recovered,
        message
    )


    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    # --------------------------------------------------------
    # Calculate bit accuracy
    # --------------------------------------------------------

    with torch.no_grad():

        predicted_bits = (
            recovered >= 0.5
        ).float()

        bit_accuracy = (
            predicted_bits == message
        ).float().mean().item() * 100


    # --------------------------------------------------------
    # Print progress
    # --------------------------------------------------------

    if step == 1 or step % 100 == 0:

        print(
            f"Step [{step:4d}/{STEPS}] | "
            f"Loss: {loss.item():.6f} | "
            f"Bit Accuracy: {bit_accuracy:.2f}%"
        )


# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 70)
print("FINAL D5 TEST")
print("=" * 70)


embedder.eval()
encoder.eval()
decoder.eval()
message_decoder.eval()


with torch.no_grad():

    cover = torch.rand(
        BATCH_SIZE,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device
    )

    message = torch.randint(
        0,
        2,
        (BATCH_SIZE, MESSAGE_BITS),
        device=device
    ).float()

    message_feature = embedder(message)

    stego = encoder(
        cover,
        message_feature
    )

    secret_feature = decoder(stego)

    recovered = message_decoder(secret_feature)

    final_loss = F.binary_cross_entropy(
        recovered,
        message
    )

    predicted_bits = (
        recovered >= 0.5
    ).float()

    final_accuracy = (
        predicted_bits == message
    ).float().mean().item() * 100


print("Final BCE Loss     :", final_loss.item())
print("Final Bit Accuracy :", final_accuracy, "%")


# ------------------------------------------------------------
# Exact message match
# ------------------------------------------------------------

exact_matches = (
    (predicted_bits == message)
    .all(dim=1)
    .float()
    .mean()
    .item() * 100
)

print("Exact Message Match:", exact_matches, "%")


print("\n" + "=" * 70)
print("D5 COMPLETE")
print("=" * 70)