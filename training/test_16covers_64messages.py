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

NUM_COVERS = 16
NUM_MESSAGES = 64

STEPS = 5000
BATCH_SIZE = 16

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
# CREATE 16 FIXED COVERS
# ============================================================

torch.manual_seed(42)

covers = torch.rand(
    NUM_COVERS,
    1,
    IMAGE_SIZE,
    IMAGE_SIZE,
    device=DEVICE
)

# ============================================================
# CREATE 64 FIXED MESSAGES
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

print("Covers:", covers.shape)
print("Messages:", messages.shape)

# ============================================================
# TRAIN
# ============================================================

for step in range(1, STEPS + 1):

    # --------------------------------------------------------
    # Randomly choose covers and messages
    # --------------------------------------------------------

    cover_indices = torch.randint(
        0,
        NUM_COVERS,
        (BATCH_SIZE,),
        device=DEVICE
    )

    message_indices = torch.randint(
        0,
        NUM_MESSAGES,
        (BATCH_SIZE,),
        device=DEVICE
    )

    batch_covers = covers[cover_indices]

    batch_messages = messages[message_indices]

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
    # MESSAGE LOSS
    # --------------------------------------------------------

    loss = criterion(
        recovered_logits,
        batch_messages
    )

    loss.backward()

    optimizer.step()

    # --------------------------------------------------------
    # BIT ACCURACY
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
    # PRINT
    # --------------------------------------------------------

    if step == 1 or step % 100 == 0:

        print(
            f"Step {step:4d} | "
            f"Loss: {loss.item():.6f} | "
            f"Bit Accuracy: {bit_accuracy:.2f}%"
        )

# ============================================================
# FINAL EVALUATION
# ALL 16 × 64 = 1024 COMBINATIONS
# ============================================================

print("\n==============================")
print("FINAL EVALUATION")
print("==============================")

correct_bits = 0
total_bits = 0

exact_matches = 0
total_combinations = (
    NUM_COVERS * NUM_MESSAGES
)

with torch.no_grad():

    for cover_idx in range(NUM_COVERS):

        cover = covers[
            cover_idx:cover_idx + 1
        ]

        # Repeat cover for all 64 messages

        batch_covers = cover.repeat(
            NUM_MESSAGES,
            1,
            1,
            1
        )

        # All 64 messages

        batch_messages = messages

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

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

        recovered_prob = torch.sigmoid(
            recovered_logits
        )

        recovered_bits = (
            recovered_prob >= 0.5
        ).float()

        # ----------------------------------------------------
        # BIT ACCURACY
        # ----------------------------------------------------

        correct_bits += (
            recovered_bits == batch_messages
        ).sum().item()

        total_bits += batch_messages.numel()

        # ----------------------------------------------------
        # EXACT MATCH
        # ----------------------------------------------------

        for i in range(NUM_MESSAGES):

            if torch.equal(
                recovered_bits[i],
                batch_messages[i]
            ):
                exact_matches += 1

# ============================================================
# FINAL METRICS
# ============================================================

final_accuracy = (
    correct_bits / total_bits
) * 100

exact_match_rate = (
    exact_matches /
    total_combinations
) * 100

print(
    f"Final Bit Accuracy : "
    f"{final_accuracy:.2f}%"
)

print(
    f"Exact Message Match: "
    f"{exact_matches}/{total_combinations}"
)

print(
    f"Exact Match Rate    : "
    f"{exact_match_rate:.2f}%"
)

print("\nD6-C TEST COMPLETE")