import os
import glob
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# Add project root to Python path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from models.baseline_cnn import BaselineSteganography
from models.message_encoder import text_to_bits


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 256

BATCH_SIZE = 16

EPOCHS = 10

LEARNING_RATE = 1e-4

MESSAGE_BITS = 256

IMAGE_LOSS_WEIGHT = 1.0

MESSAGE_LOSS_WEIGHT = 1.0

NUM_WORKERS = 2

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


print("=" * 70)
print("BASELINE CNN TRAINING")
print("=" * 70)

print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())
print("Device          :", device)

if torch.cuda.is_available():

    print("GPU             :",
          torch.cuda.get_device_name(0))

    print("GPU count       :",
          torch.cuda.device_count())

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

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("L")

        image = self.transform(
            image
        )

        # ----------------------------------------------------
        # Generate a random 256-bit message
        # ----------------------------------------------------

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

train_dir = os.path.join(
    "dataset",
    "processed",
    "train"
)

validation_dir = os.path.join(
    "dataset",
    "processed",
    "validation"
)


train_paths = sorted(
    glob.glob(
        os.path.join(
            train_dir,
            "*"
        )
    )
)

validation_paths = sorted(
    glob.glob(
        os.path.join(
            validation_dir,
            "*"
        )
    )
)


print("\nDataset information")
print("-" * 70)

print("Training images   :", len(train_paths))

print("Validation images :", len(validation_paths))


if len(train_paths) == 0:

    raise RuntimeError(
        f"No training images found in: {train_dir}"
    )


if len(validation_paths) == 0:

    raise RuntimeError(
        f"No validation images found in: {validation_dir}"
    )


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = SteganographyDataset(
    train_paths
)

validation_dataset = SteganographyDataset(
    validation_paths
)


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

print("Training batches   :", len(train_loader))
print("Validation batches :", len(validation_loader))


# ============================================================
# MODEL
# ============================================================

model = BaselineSteganography(
    message_bits=MESSAGE_BITS
)

model = model.to(device)


# ============================================================
# LOSS FUNCTIONS
# ============================================================

image_criterion = nn.MSELoss()

message_criterion = nn.BCELoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

checkpoint_dir = os.path.join(
    "outputs",
    "checkpoints"
)

os.makedirs(
    checkpoint_dir,
    exist_ok=True
)


best_validation_loss = float("inf")


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(EPOCHS):

    print("\n")
    print("=" * 70)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_total_loss = 0.0

    train_image_loss = 0.0

    train_message_loss = 0.0


    for batch_index, (cover, message) in enumerate(
        train_loader
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
        # Calculate losses
        # ----------------------------------------------------

        image_loss = image_criterion(
            stego,
            cover
        )

        message_loss = message_criterion(
            recovered_message,
            message
        )


        total_loss = (
            IMAGE_LOSS_WEIGHT * image_loss
            +
            MESSAGE_LOSS_WEIGHT * message_loss
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        total_loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # Accumulate losses
        # ----------------------------------------------------

        train_total_loss += (
            total_loss.item()
        )

        train_image_loss += (
            image_loss.item()
        )

        train_message_loss += (
            message_loss.item()
        )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_index + 1
        ) % 100 == 0:

            print(
                f"Batch {batch_index + 1}/{len(train_loader)} "
                f"| Loss: {total_loss.item():.6f}"
            )


    # --------------------------------------------------------
    # Average training losses
    # --------------------------------------------------------

    train_total_loss /= len(
        train_loader
    )

    train_image_loss /= len(
        train_loader
    )

    train_message_loss /= len(
        train_loader
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    validation_total_loss = 0.0

    validation_image_loss = 0.0

    validation_message_loss = 0.0

    correct_bits = 0

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


            total_loss = (
                IMAGE_LOSS_WEIGHT * image_loss
                +
                MESSAGE_LOSS_WEIGHT * message_loss
            )


            validation_total_loss += (
                total_loss.item()
            )

            validation_image_loss += (
                image_loss.item()
            )

            validation_message_loss += (
                message_loss.item()
            )


            # ------------------------------------------------
            # Bit accuracy
            # ------------------------------------------------

            predicted_bits = (
                recovered_message >= 0.5
            ).float()


            correct_bits += (
                predicted_bits == message
            ).sum().item()


            total_bits += message.numel()


    # --------------------------------------------------------
    # Average validation losses
    # --------------------------------------------------------

    validation_total_loss /= len(
        validation_loader
    )

    validation_image_loss /= len(
        validation_loader
    )

    validation_message_loss /= len(
        validation_loader
    )


    bit_accuracy = (
        correct_bits / total_bits
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\nTraining Results")
    print("-" * 70)

    print(
        f"Training Total Loss    : "
        f"{train_total_loss:.6f}"
    )

    print(
        f"Training Image Loss    : "
        f"{train_image_loss:.6f}"
    )

    print(
        f"Training Message Loss  : "
        f"{train_message_loss:.6f}"
    )


    print("\nValidation Results")
    print("-" * 70)

    print(
        f"Validation Total Loss  : "
        f"{validation_total_loss:.6f}"
    )

    print(
        f"Validation Image Loss  : "
        f"{validation_image_loss:.6f}"
    )

    print(
        f"Validation Message Loss: "
        f"{validation_message_loss:.6f}"
    )

    print(
        f"Validation Bit Accuracy: "
        f"{bit_accuracy * 100:.2f}%"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if validation_total_loss < best_validation_loss:

        best_validation_loss = (
            validation_total_loss
        )

        checkpoint_path = os.path.join(
            checkpoint_dir,
            "baseline_best.pth"
        )

        torch.save(
            {
                "epoch": epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "validation_loss":
                    validation_total_loss,

                "bit_accuracy":
                    bit_accuracy
            },
            checkpoint_path
        )

        print(
            "\n✓ Best model saved:"
        )

        print(
            checkpoint_path
        )


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    "Best validation loss:",
    best_validation_loss
)

print(
    "Checkpoint:",
    os.path.join(
        checkpoint_dir,
        "baseline_best.pth"
    )
)

print("=" * 70)