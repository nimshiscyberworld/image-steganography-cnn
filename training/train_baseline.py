import os
import sys
import glob
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

# Kaggle dataset path
DATASET_ROOT = "/kaggle/input/datasets/nimshipaul/image-steganography-processed"

TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VALIDATION_DIR = os.path.join(DATASET_ROOT, "validation")

# Where the trained model will be saved
OUTPUT_DIR = "/kaggle/working/baseline_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(
    OUTPUT_DIR,
    "baseline_best.pth"
)

# Training parameters
IMAGE_SIZE = 256
MESSAGE_BITS = 256

BATCH_SIZE = 16
NUM_EPOCHS = 10

LEARNING_RATE = 1e-4

# Image loss weight
LAMBDA_IMAGE = 1.0

# Message loss weight
LAMBDA_MESSAGE = 1.0

RANDOM_SEED = 42

# DataLoader workers
# 0 is safest for Kaggle/Windows compatibility.
NUM_WORKERS = 0


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 70)
print("BASELINE CNN TRAINING")
print("=" * 70)

print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())
print("Device          :", device)

if torch.cuda.is_available():

    print("GPU             :", torch.cuda.get_device_name(0))
    print("GPU count       :", torch.cuda.device_count())

print("=" * 70)


# ============================================================
# DATASET
# ============================================================

class SteganographyDataset(Dataset):

    def __init__(self, image_paths):

        self.image_paths = image_paths

        self.transform = transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.ToTensor()
        ])

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        try:

            image = Image.open(
                image_path
            ).convert("L")

        except Exception as e:

            raise RuntimeError(
                f"Could not read image: {image_path}\n"
                f"Error: {e}"
            )

        image = self.transform(image)

        # Generate random binary secret message
        message = torch.randint(
            0,
            2,
            (MESSAGE_BITS,),
            dtype=torch.float32
        )

        return image, message


# ============================================================
# FIND DATASET
# ============================================================

print("\nDataset information")
print("-" * 70)

print("Dataset root      :", DATASET_ROOT)
print("Training directory:", TRAIN_DIR)
print("Validation dir    :", VALIDATION_DIR)


# Check dataset root
if not os.path.exists(DATASET_ROOT):

    raise RuntimeError(
        f"Dataset root not found:\n{DATASET_ROOT}\n\n"
        "Make sure the Kaggle dataset is added using "
        "Add Input."
    )


# Find training images
train_paths = []

for extension in [
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.bmp",
    "*.pgm",
    "*.ppm"
]:

    train_paths.extend(
        glob.glob(
            os.path.join(
                TRAIN_DIR,
                extension
            )
        )
    )


# Find validation images
validation_paths = []

for extension in [
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.bmp",
    "*.pgm",
    "*.ppm"
]:

    validation_paths.extend(
        glob.glob(
            os.path.join(
                VALIDATION_DIR,
                extension
            )
        )
    )


train_paths = sorted(train_paths)
validation_paths = sorted(validation_paths)


print(
    "Training images   :",
    len(train_paths)
)

print(
    "Validation images :",
    len(validation_paths)
)


if len(train_paths) == 0:

    raise RuntimeError(
        f"No training images found in:\n{TRAIN_DIR}"
    )


if len(validation_paths) == 0:

    raise RuntimeError(
        f"No validation images found in:\n{VALIDATION_DIR}"
    )


# ============================================================
# DATASET OBJECTS
# ============================================================

train_dataset = SteganographyDataset(
    train_paths
)

validation_dataset = SteganographyDataset(
    validation_paths
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


print("\nDataLoader")
print("-" * 70)

print(
    "Training batches   :",
    len(train_loader)
)

print(
    "Validation batches :",
    len(validation_loader)
)


# ============================================================
# IMPORT BASELINE MODEL
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:

    sys.path.insert(
        0,
        PROJECT_ROOT
    )


from models.baseline_cnn import BaselineSteganography


# ============================================================
# MODEL
# ============================================================

model = BaselineSteganography()

model = model.to(device)


# ============================================================
# MULTI-GPU SUPPORT
# ============================================================

# We use one GPU initially.
#
# This makes the baseline easier to reproduce.
# Later we can enable both Tesla T4 GPUs.

if torch.cuda.is_available():

    print("\nUsing GPU:")
    print(
        torch.cuda.get_device_name(0)
    )


# ============================================================
# MODEL PARAMETERS
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


print("\nModel")
print("-" * 70)

print(
    "Total parameters     :",
    f"{total_parameters:,}"
)

print(
    "Trainable parameters :",
    f"{trainable_parameters:,}"
)


# ============================================================
# LOSS FUNCTIONS
# ============================================================

image_loss_function = nn.MSELoss()

message_loss_function = nn.BCELoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    total_image_loss = 0.0
    total_message_loss = 0.0

    total_batches = len(train_loader)

    for batch_index, (cover, message) in enumerate(
        train_loader,
        start=1
    ):

        cover = cover.to(
            device,
            non_blocking=True
        )

        message = message.to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        stego, recovered_message = model(
            cover,
            message
        )

        # ----------------------------------------------------
        # Image reconstruction loss
        # ----------------------------------------------------

        image_loss = image_loss_function(
            stego,
            cover
        )

        # ----------------------------------------------------
        # Message recovery loss
        # ----------------------------------------------------

        message_loss = message_loss_function(
            recovered_message,
            message
        )

        # ----------------------------------------------------
        # Combined loss
        # ----------------------------------------------------

        loss = (
            LAMBDA_IMAGE * image_loss
            +
            LAMBDA_MESSAGE * message_loss
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_loss += loss.item()

        total_image_loss += (
            image_loss.item()
        )

        total_message_loss += (
            message_loss.item()
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_index == 1
            or batch_index % 50 == 0
            or batch_index == total_batches
        ):

            print(
                f"Batch "
                f"{batch_index}/{total_batches} | "
                f"Loss: {loss.item():.6f} | "
                f"Image Loss: {image_loss.item():.6f} | "
                f"Message Loss: {message_loss.item():.6f}"
            )

    average_loss = (
        total_loss / total_batches
    )

    average_image_loss = (
        total_image_loss / total_batches
    )

    average_message_loss = (
        total_message_loss / total_batches
    )

    return (
        average_loss,
        average_image_loss,
        average_message_loss
    )


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate():

    model.eval()

    total_loss = 0.0
    total_image_loss = 0.0
    total_message_loss = 0.0

    total_correct_bits = 0
    total_bits = 0

    with torch.no_grad():

        for cover, message in validation_loader:

            cover = cover.to(
                device,
                non_blocking=True
            )

            message = message.to(
                device,
                non_blocking=True
            )

            # Forward pass
            stego, recovered_message = model(
                cover,
                message
            )

            # Losses
            image_loss = image_loss_function(
                stego,
                cover
            )

            message_loss = message_loss_function(
                recovered_message,
                message
            )

            loss = (
                LAMBDA_IMAGE * image_loss
                +
                LAMBDA_MESSAGE * message_loss
            )

            total_loss += loss.item()

            total_image_loss += (
                image_loss.item()
            )

            total_message_loss += (
                message_loss.item()
            )

            # ------------------------------------------------
            # Bit accuracy
            # ------------------------------------------------

            predicted_bits = (
                recovered_message >= 0.5
            ).float()

            correct_bits = (
                predicted_bits == message
            ).sum().item()

            total_correct_bits += correct_bits

            total_bits += message.numel()

    average_loss = (
        total_loss / len(validation_loader)
    )

    average_image_loss = (
        total_image_loss /
        len(validation_loader)
    )

    average_message_loss = (
        total_message_loss /
        len(validation_loader)
    )

    bit_accuracy = (
        total_correct_bits /
        total_bits
    )

    return (
        average_loss,
        average_image_loss,
        average_message_loss,
        bit_accuracy
    )


# ============================================================
# TRAINING LOOP
# ============================================================

print("\n")
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)


best_validation_loss = float("inf")


training_history = []


for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    print("\n")
    print("=" * 70)

    print(
        f"Epoch {epoch}/{NUM_EPOCHS}"
    )

    print("=" * 70)


    # ========================================================
    # TRAIN
    # ========================================================

    train_loss, train_image_loss, train_message_loss = (
        train_one_epoch()
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    (
        validation_loss,
        validation_image_loss,
        validation_message_loss,
        bit_accuracy
    ) = validate()


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\nEpoch results")
    print("-" * 70)

    print(
        f"Train Loss           : "
        f"{train_loss:.6f}"
    )

    print(
        f"Train Image Loss     : "
        f"{train_image_loss:.6f}"
    )

    print(
        f"Train Message Loss   : "
        f"{train_message_loss:.6f}"
    )

    print(
        f"Validation Loss      : "
        f"{validation_loss:.6f}"
    )

    print(
        f"Validation Image Loss: "
        f"{validation_image_loss:.6f}"
    )

    print(
        f"Validation Msg Loss  : "
        f"{validation_message_loss:.6f}"
    )

    print(
        f"Message Bit Accuracy : "
        f"{bit_accuracy * 100:.2f}%"
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    training_history.append({

        "epoch": epoch,

        "train_loss":
            train_loss,

        "train_image_loss":
            train_image_loss,

        "train_message_loss":
            train_message_loss,

        "validation_loss":
            validation_loss,

        "validation_image_loss":
            validation_image_loss,

        "validation_message_loss":
            validation_message_loss,

        "bit_accuracy":
            bit_accuracy
    })


    # ========================================================
    # SAVE BEST CHECKPOINT
    # ========================================================

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        checkpoint = {

            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "validation_loss":
                validation_loss,

            "bit_accuracy":
                bit_accuracy,

            "config": {

                "image_size":
                    IMAGE_SIZE,

                "message_bits":
                    MESSAGE_BITS,

                "batch_size":
                    BATCH_SIZE,

                "learning_rate":
                    LEARNING_RATE,

                "lambda_image":
                    LAMBDA_IMAGE,

                "lambda_message":
                    LAMBDA_MESSAGE
            }
        }

        torch.save(
            checkpoint,
            CHECKPOINT_PATH
        )

        print("\nBest model saved!")
        print(
            "Checkpoint:",
            CHECKPOINT_PATH
        )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    OUTPUT_DIR,
    "training_history.txt"
)

with open(
    history_path,
    "w"
) as file:

    file.write(
        "BASELINE CNN TRAINING HISTORY\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    for result in training_history:

        file.write(
            f"Epoch: {result['epoch']}\n"
        )

        file.write(
            f"Train Loss: "
            f"{result['train_loss']:.6f}\n"
        )

        file.write(
            f"Train Image Loss: "
            f"{result['train_image_loss']:.6f}\n"
        )

        file.write(
            f"Train Message Loss: "
            f"{result['train_message_loss']:.6f}\n"
        )

        file.write(
            f"Validation Loss: "
            f"{result['validation_loss']:.6f}\n"
        )

        file.write(
            f"Validation Image Loss: "
            f"{result['validation_image_loss']:.6f}\n"
        )

        file.write(
            f"Validation Message Loss: "
            f"{result['validation_message_loss']:.6f}\n"
        )

        file.write(
            f"Message Bit Accuracy: "
            f"{result['bit_accuracy'] * 100:.2f}%\n"
        )

        file.write(
            "\n"
        )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("BASELINE TRAINING COMPLETE")
print("=" * 70)

print(
    "Best checkpoint :",
    CHECKPOINT_PATH
)

print(
    "Training history:",
    history_path
)

print(
    "Best validation loss:",
    f"{best_validation_loss:.6f}"
)

print("=" * 70)
