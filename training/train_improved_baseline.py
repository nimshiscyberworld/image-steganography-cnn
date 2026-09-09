import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.baseline_cnn import BaselineSteganography


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = "/kaggle/input/datasets/nimshipaul/image-steganography-processed"

OUTPUT_DIR = "/kaggle/working/improved_baseline_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "validation")

IMAGE_SIZE = 256
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

MESSAGE_BITS = 256

LAMBDA_IMAGE = 1.0
LAMBDA_MESSAGE = 1.0

RANDOM_SEED = 42
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("IMPROVED BASELINE CNN TRAINING")
print("=" * 60)

print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())
print("Device          :", device)

if torch.cuda.is_available():
    print("GPU             :", torch.cuda.get_device_name(0))


# ============================================================
# DATASET
# ============================================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=transform
)

print()
print("Dataset Information")
print("-" * 60)
print("Training images   :", len(train_dataset))
print("Validation images :", len(val_dataset))


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

print("Training batches   :", len(train_loader))
print("Validation batches :", len(val_loader))


# ============================================================
# MODEL
# ============================================================

model = BaselineSteganography(
    message_bits=MESSAGE_BITS
).to(device)

print()
print("Model Information")
print("-" * 60)

total_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print("Trainable parameters :", total_params)


# ============================================================
# LOSS FUNCTIONS
# ============================================================

image_criterion = nn.MSELoss()
message_criterion = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# CHECKPOINT
# ============================================================

checkpoint_path = os.path.join(
    OUTPUT_DIR,
    "improved_baseline_best.pth"
)

history_path = os.path.join(
    OUTPUT_DIR,
    "training_history.txt"
)

best_val_loss = float("inf")


# ============================================================
# TRAINING
# ============================================================

with open(history_path, "w") as history_file:

    history_file.write("IMPROVED BASELINE TRAINING\n")
    history_file.write("=" * 60 + "\n")

    for epoch in range(NUM_EPOCHS):

        model.train()

        total_train_loss = 0.0
        total_image_loss = 0.0
        total_message_loss = 0.0

        total_message_bits = 0
        correct_message_bits = 0

        for cover, _ in train_loader:

            cover = cover.to(device)

            batch_size = cover.size(0)

            # Generate random 256-bit message
            message = torch.randint(
                0,
                2,
                (batch_size, MESSAGE_BITS),
                device=device
            ).float()

            optimizer.zero_grad()

            stego, recovered_message = model(
                cover,
                message
            )

            # Image reconstruction loss
            image_loss = image_criterion(
                stego,
                cover
            )

            # Message recovery loss
            message_loss = message_criterion(
                recovered_message,
                message
            )

            # Total loss
            loss = (
                LAMBDA_IMAGE * image_loss
                + LAMBDA_MESSAGE * message_loss
            )

            loss.backward()

            optimizer.step()

            total_train_loss += loss.item()
            total_image_loss += image_loss.item()
            total_message_loss += message_loss.item()

            # Message bit accuracy
            predicted_bits = (
                recovered_message >= 0.5
            ).float()

            correct_message_bits += (
                predicted_bits == message
            ).sum().item()

            total_message_bits += message.numel()


        avg_train_loss = total_train_loss / len(train_loader)
        avg_image_loss = total_image_loss / len(train_loader)
        avg_message_loss = total_message_loss / len(train_loader)

        train_bit_accuracy = (
            correct_message_bits /
            total_message_bits
        ) * 100


        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        total_val_loss = 0.0
        total_val_image_loss = 0.0
        total_val_message_loss = 0.0

        total_val_bits = 0
        correct_val_bits = 0

        with torch.no_grad():

            for cover, _ in val_loader:

                cover = cover.to(device)

                batch_size = cover.size(0)

                message = torch.randint(
                    0,
                    2,
                    (batch_size, MESSAGE_BITS),
                    device=device
                ).float()

                stego, recovered_message = model(
                    cover,
                    message
                )

                image_loss = image_criterion(
                    stego,
                    cover
                )

                message_loss = message_criterion(
                    recovered_message,
                    message
                )

                loss = (
                    LAMBDA_IMAGE * image_loss
                    + LAMBDA_MESSAGE * message_loss
                )

                total_val_loss += loss.item()
                total_val_image_loss += image_loss.item()
                total_val_message_loss += message_loss.item()

                predicted_bits = (
                    recovered_message >= 0.5
                ).float()

                correct_val_bits += (
                    predicted_bits == message
                ).sum().item()

                total_val_bits += message.numel()


        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_image_loss = (
            total_val_image_loss / len(val_loader)
        )
        avg_val_message_loss = (
            total_val_message_loss / len(val_loader)
        )

        val_bit_accuracy = (
            correct_val_bits /
            total_val_bits
        ) * 100


        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print()
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        print("-" * 60)

        print(f"Train Loss           : {avg_train_loss:.6f}")
        print(f"Train Image Loss     : {avg_image_loss:.6f}")
        print(f"Train Message Loss   : {avg_message_loss:.6f}")
        print(f"Train Bit Accuracy   : {train_bit_accuracy:.2f}%")

        print(f"Validation Loss      : {avg_val_loss:.6f}")
        print(f"Validation Image Loss: {avg_val_image_loss:.6f}")
        print(f"Validation Msg Loss  : {avg_val_message_loss:.6f}")
        print(f"Validation Bit Acc.  : {val_bit_accuracy:.2f}%")


        # ====================================================
        # SAVE HISTORY
        # ====================================================

        history_file.write(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}\n"
        )

        history_file.write(
            f"Train Loss: {avg_train_loss:.6f}\n"
        )

        history_file.write(
            f"Train Image Loss: {avg_image_loss:.6f}\n"
        )

        history_file.write(
            f"Train Message Loss: {avg_message_loss:.6f}\n"
        )

        history_file.write(
            f"Train Bit Accuracy: {train_bit_accuracy:.2f}%\n"
        )

        history_file.write(
            f"Validation Loss: {avg_val_loss:.6f}\n"
        )

        history_file.write(
            f"Validation Image Loss: "
            f"{avg_val_image_loss:.6f}\n"
        )

        history_file.write(
            f"Validation Message Loss: "
            f"{avg_val_message_loss:.6f}\n"
        )

        history_file.write(
            f"Validation Bit Accuracy: "
            f"{val_bit_accuracy:.2f}%\n"
        )

        history_file.write("\n")


        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if avg_val_loss < best_val_loss:

            best_val_loss = avg_val_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "val_bit_accuracy": val_bit_accuracy
                },
                checkpoint_path
            )

            print()
            print("Best model saved!")
            print("Checkpoint:", checkpoint_path)


print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print("Best validation loss :", best_val_loss)
print("Checkpoint           :", checkpoint_path)
print("History              :", history_path)