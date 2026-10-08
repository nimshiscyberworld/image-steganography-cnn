import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# ============================================================
# PROJECT ROOT
# ============================================================

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from models.baseline_cnn import (
    MessageEmbedder,
    Encoder,
    Decoder,
    MessageDecoder
)

# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 256
MESSAGE_BITS = 256

NUM_MESSAGES = 64
STEPS = 3000
BATCH_SIZE = 8

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
# OPTIMIZER
# ============================================================

parameters = (
    list(message_embedder.parameters())
    + list(encoder.parameters())
    + list(decoder.parameters())
    + list(message_decoder.parameters())
)

optimizer = optim.Adam(
    parameters,
    lr=LR
)

criterion = nn.BCEWithLogitsLoss()

# ============================================================
# FIXED COVER
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
# 64 FIXED RANDOM MESSAGES
# ============================================================

messages = torch.randint(
    0,
    2,
    (
        NUM_MESSAGES,
        MESSAGE_BITS
    ),
    device=DEVICE
).float()

print("Cover:", cover.shape)
print("Messages:", messages.shape)

# ============================================================
# TRAIN
# ============================================================

for step in range(1, STEPS + 1):

    # --------------------------------------------------------
    # Randomly select messages from the fixed 64
    # --------------------------------------------------------

    indices = torch.randint(
        0,
        NUM_MESSAGES,
        (BATCH_SIZE,),
        device=DEVICE
    )

    batch_messages = messages[indices]

    # Repeat SAME cover for batch
    batch_covers = cover.repeat(
        BATCH_SIZE,
        1,
        1,
        1
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    optimizer.zero_grad()

    message_feature = message_embedder(
        batch_messages
    )

    stego = encoder(
        batch_covers,
        message_feature
    )

    secret_feature = decoder(
        stego
    )

    recovered_logits = message_decoder(
        secret_feature
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss = criterion(
        recovered_logits,
        batch_messages
    )

    loss.backward()

    optimizer.step()

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    with torch.no_grad():

        recovered_prob = torch.sigmoid(
            recovered_logits
        )

        recovered_bits = (
            recovered_prob >= 0.5
        ).float()

        bit_accuracy = (
            recovered_bits == batch_messages
        ).float().mean().item() * 100

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    if step == 1 or step % 100 == 0:

        print(
            f"Step {step:4d} | "
            f"Loss: {loss.item():.6f} | "
            f"Bit Accuracy: {bit_accuracy:.2f}%"
        )

# ============================================================
# FINAL EVALUATION ON ALL 64 MESSAGES
# ============================================================

print("\n==============================")
print("FINAL EVALUATION")
print("==============================")

correct_bits = 0
total_bits = 0

exact_matches = 0

with torch.no_grad():

    batch_covers = cover.repeat(
        NUM_MESSAGES,
        1,
        1,
        1
    )

    message_feature = message_embedder(
        messages
    )

    stego = encoder(
        batch_covers,
        message_feature
    )

    secret_feature = decoder(
        stego
    )

    recovered_logits = message_decoder(
        secret_feature
    )

    recovered_prob = torch.sigmoid(
        recovered_logits
    )

    recovered_bits = (
        recovered_prob >= 0.5
    ).float()

    # Bit accuracy

    correct_bits = (
        recovered_bits == messages
    ).sum().item()

    total_bits = messages.numel()

    final_accuracy = (
        correct_bits / total_bits
    ) * 100

    # Exact message matches

    for i in range(NUM_MESSAGES):

        if torch.equal(
            recovered_bits[i],
            messages[i]
        ):
            exact_matches += 1

print(
    f"Final Bit Accuracy : "
    f"{final_accuracy:.2f}%"
)

print(
    f"Exact Message Match: "
    f"{exact_matches}/{NUM_MESSAGES}"
)

print(
    f"Exact Match Rate    : "
    f"{(exact_matches / NUM_MESSAGES) * 100:.2f}%"
)

print("\nD6-B TEST COMPLETE")