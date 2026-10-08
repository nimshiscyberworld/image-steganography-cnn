import os
import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from PIL import Image

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)

from models.baseline_cnn import (
    MessageEmbedder,
    Encoder,
    Decoder,
    MessageDecoder
)

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 256
MESSAGE_BITS = 256

BATCH_SIZE = 8

EPOCHS = 10

LEARNING_RATE = 1e-4

NUM_WORKERS = 2

# ------------------------------------------------------------
# Kaggle dataset
# ------------------------------------------------------------

DATASET_ROOT = (
    "/kaggle/input/datasets/"
    "nimshipaul/image-steganography-processed"
)

TRAIN_DIR = os.path.join(
    DATASET_ROOT,
    "train"
)

VAL_DIR = os.path.join(
    DATASET_ROOT,
    "validation"
)

# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "d7_real_data"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

BEST_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "d7_best_model.pth"
)

LAST_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "d7_last_model.pth"
)

# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# DATASET
# ============================================================

class ImageDataset(Dataset):

    def __init__(self, folder):

        self.folder = folder

        valid_extensions = (
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".pgm"
        )

        self.files = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(
                valid_extensions
            )
        ])

        if len(self.files) == 0:

            raise RuntimeError(
                f"No images found in:\n{folder}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        image_path = self.files[index]

        image = Image.open(
            image_path
        ).convert("L")

        image = image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            Image.Resampling.BILINEAR
        )

        image = np.array(
            image,
            dtype=np.float32
        ) / 255.0

        image = torch.from_numpy(
            image
        ).unsqueeze(0)

        return image


# ============================================================
# LOAD DATA
# ============================================================

train_dataset = ImageDataset(
    TRAIN_DIR
)

val_dataset = ImageDataset(
    VAL_DIR
)

print("======================================")
print("D7 REAL DATA TRAINING")
print("======================================")

print("Device:", DEVICE)

print(
    "Training images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(val_dataset)
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Message bits:",
    MESSAGE_BITS
)

# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=False
)

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
# PARAMETER COUNT
# ============================================================

total_params = sum(
    p.numel()
    for p in message_embedder.parameters()
)

total_params += sum(
    p.numel()
    for p in encoder.parameters()
)

total_params += sum(
    p.numel()
    for p in decoder.parameters()
)

total_params += sum(
    p.numel()
    for p in message_decoder.parameters()
)

print(
    "Total parameters:",
    f"{total_params:,}"
)

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
    lr=LEARNING_RATE
)

# ============================================================
# LOSS
# ============================================================

criterion = nn.BCEWithLogitsLoss()

# ============================================================
# HELPER FUNCTION
# ============================================================

def calculate_metrics(
    logits,
    messages
):

    probabilities = torch.sigmoid(
        logits
    )

    predicted_bits = (
        probabilities >= 0.5
    ).float()

    correct_bits = (
        predicted_bits == messages
    ).sum().item()

    total_bits = messages.numel()

    bit_accuracy = (
        correct_bits /
        total_bits
    ) * 100.0

    ber = 1.0 - (
        correct_bits /
        total_bits
    )

    exact_matches = (
        predicted_bits == messages
    ).all(dim=1).sum().item()

    total_messages = messages.size(0)

    exact_match_rate = (
        exact_matches /
        total_messages
    ) * 100.0

    return (
        bit_accuracy,
        ber,
        exact_match_rate
    )


# ============================================================
# BEST MODEL TRACKING
# ============================================================

best_val_accuracy = 0.0

best_val_loss = float("inf")

# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(
    1,
    EPOCHS + 1
):

    # ========================================================
    # TRAIN
    # ========================================================

    message_embedder.train()
    encoder.train()
    decoder.train()
    message_decoder.train()

    train_loss_total = 0.0

    train_correct_bits = 0
    train_total_bits = 0

    train_exact_matches = 0
    train_total_messages = 0

    for covers in train_loader:

        covers = covers.to(
            DEVICE,
            non_blocking=True
        )

        batch_size = covers.size(0)

        # ----------------------------------------------------
        # Fresh random message
        # ----------------------------------------------------

        messages = torch.randint(
            0,
            2,
            (
                batch_size,
                MESSAGE_BITS
            ),
            device=DEVICE
        ).float()

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        message_feature = (
            message_embedder(
                messages
            )
        )

        stego = encoder(
            covers,
            message_feature
        )

        secret_feature = decoder(
            stego
        )

        recovered_logits = (
            message_decoder(
                secret_feature
            )
        )

        # ----------------------------------------------------
        # Message loss
        # ----------------------------------------------------

        loss = criterion(
            recovered_logits,
            messages
        )

        # ----------------------------------------------------
        # Backprop
        # ----------------------------------------------------

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        train_loss_total += (
            loss.item()
            * batch_size
        )

        with torch.no_grad():

            predicted_bits = (
                torch.sigmoid(
                    recovered_logits
                ) >= 0.5
            ).float()

            train_correct_bits += (
                predicted_bits == messages
            ).sum().item()

            train_total_bits += (
                messages.numel()
            )

            train_exact_matches += (
                (
                    predicted_bits == messages
                )
                .all(dim=1)
                .sum()
                .item()
            )

            train_total_messages += (
                batch_size
            )

    # ========================================================
    # TRAIN METRICS
    # ========================================================

    train_loss = (
        train_loss_total /
        len(train_dataset)
    )

    train_accuracy = (
        train_correct_bits /
        train_total_bits
    ) * 100.0

    train_ber = 1.0 - (
        train_correct_bits /
        train_total_bits
    )

    train_exact_rate = (
        train_exact_matches /
        train_total_messages
    ) * 100.0

    # ========================================================
    # VALIDATION
    # ========================================================

    message_embedder.eval()
    encoder.eval()
    decoder.eval()
    message_decoder.eval()

    val_loss_total = 0.0

    val_correct_bits = 0
    val_total_bits = 0

    val_exact_matches = 0
    val_total_messages = 0

    with torch.no_grad():

        for covers in val_loader:

            covers = covers.to(
                DEVICE,
                non_blocking=True
            )

            batch_size = covers.size(0)

            # ------------------------------------------------
            # IMPORTANT:
            # Fresh random messages for validation too
            # ------------------------------------------------

            messages = torch.randint(
                0,
                2,
                (
                    batch_size,
                    MESSAGE_BITS
                ),
                device=DEVICE
            ).float()

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            message_feature = (
                message_embedder(
                    messages
                )
            )

            stego = encoder(
                covers,
                message_feature
            )

            secret_feature = decoder(
                stego
            )

            recovered_logits = (
                message_decoder(
                    secret_feature
                )
            )

            loss = criterion(
                recovered_logits,
                messages
            )

            val_loss_total += (
                loss.item()
                * batch_size
            )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            predicted_bits = (
                torch.sigmoid(
                    recovered_logits
                ) >= 0.5
            ).float()

            val_correct_bits += (
                predicted_bits == messages
            ).sum().item()

            val_total_bits += (
                messages.numel()
            )

            val_exact_matches += (
                (
                    predicted_bits == messages
                )
                .all(dim=1)
                .sum()
                .item()
            )

            val_total_messages += (
                batch_size
            )

    # ========================================================
    # VALIDATION METRICS
    # ========================================================

    val_loss = (
        val_loss_total /
        len(val_dataset)
    )

    val_accuracy = (
        val_correct_bits /
        val_total_bits
    ) * 100.0

    val_ber = 1.0 - (
        val_correct_bits /
        val_total_bits
    )

    val_exact_rate = (
        val_exact_matches /
        val_total_messages
    ) * 100.0

    # ========================================================
    # PRINT
    # ========================================================

    print("\n")
    print("======================================")
    print(
        f"Epoch [{epoch}/{EPOCHS}]"
    )
    print("======================================")

    print(
        f"Train Loss           : "
        f"{train_loss:.6f}"
    )

    print(
        f"Train Bit Accuracy   : "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Train BER            : "
        f"{train_ber:.6f}"
    )

    print(
        f"Train Exact Match    : "
        f"{train_exact_rate:.2f}%"
    )

    print(
        f"Validation Loss      : "
        f"{val_loss:.6f}"
    )

    print(
        f"Validation Bit Acc.  : "
        f"{val_accuracy:.2f}%"
    )

    print(
        f"Validation BER       : "
        f"{val_ber:.6f}"
    )

    print(
        f"Validation Exact     : "
        f"{val_exact_rate:.2f}%"
    )

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if (
        val_accuracy > best_val_accuracy
        or (
            val_accuracy == best_val_accuracy
            and val_loss < best_val_loss
        )
    ):

        best_val_accuracy = val_accuracy

        best_val_loss = val_loss

        checkpoint = {

            "epoch": epoch,

            "message_embedder":
                message_embedder.state_dict(),

            "encoder":
                encoder.state_dict(),

            "decoder":
                decoder.state_dict(),

            "message_decoder":
                message_decoder.state_dict(),

            "optimizer":
                optimizer.state_dict(),

            "val_loss":
                val_loss,

            "val_accuracy":
                val_accuracy,

            "val_ber":
                val_ber,

            "val_exact_rate":
                val_exact_rate
        }

        torch.save(
            checkpoint,
            BEST_MODEL_PATH
        )

        print(
            "\n*** BEST MODEL SAVED ***"
        )

        print(
            BEST_MODEL_PATH
        )

    # ========================================================
    # SAVE LAST MODEL
    # ========================================================

    checkpoint = {

        "epoch": epoch,

        "message_embedder":
            message_embedder.state_dict(),

        "encoder":
            encoder.state_dict(),

        "decoder":
            decoder.state_dict(),

        "message_decoder":
            message_decoder.state_dict(),

        "optimizer":
            optimizer.state_dict(),

        "val_loss":
            val_loss,

        "val_accuracy":
            val_accuracy
    }

    torch.save(
        checkpoint,
        LAST_MODEL_PATH
    )

# ============================================================
# FINAL
# ============================================================

print("\n")
print("======================================")
print("D7 TRAINING COMPLETE")
print("======================================")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Best Validation Loss: "
    f"{best_val_loss:.6f}"
)

print(
    "Best checkpoint:"
)

print(
    BEST_MODEL_PATH
)
