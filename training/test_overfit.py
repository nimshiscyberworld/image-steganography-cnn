import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import Image

# --------------------------------------------------
# Project path
# --------------------------------------------------

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from models.baseline_cnn import BaselineSteganography


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_ROOT = "/kaggle/input/datasets/nimshipaul/image-steganography-processed"

IMAGE_SIZE = 256
BATCH_SIZE = 16
MESSAGE_BITS = 256

NUM_ITERATIONS = 300
LEARNING_RATE = 1e-4

LAMBDA_IMAGE = 1.0
LAMBDA_MESSAGE = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform

        self.image_files = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp")
            )
        ]

        self.image_files.sort()

        if len(self.image_files) == 0:
            raise RuntimeError(
                f"No images found in {root_dir}"
            )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        image_path = os.path.join(
            self.root_dir,
            self.image_files[idx]
        )

        image = Image.open(image_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


dataset = ImageDataset(
    os.path.join(DATASET_ROOT, "train"),
    transform=transform
)


# --------------------------------------------------
# Select ONLY 16 images
# --------------------------------------------------

indices = list(range(BATCH_SIZE))

small_dataset = Subset(
    dataset,
    indices
)

loader = DataLoader(
    small_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# Load the same 16 images
# --------------------------------------------------

cover = next(iter(loader)).to(DEVICE)

print("Cover shape:", cover.shape)


# --------------------------------------------------
# Create ONE fixed random message
# --------------------------------------------------

torch.manual_seed(42)

message = torch.randint(
    0,
    2,
    (BATCH_SIZE, MESSAGE_BITS),
    dtype=torch.float32
).to(DEVICE)

print("Message shape:", message.shape)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = BaselineSteganography(
    message_bits=MESSAGE_BITS
).to(DEVICE)

print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)


# --------------------------------------------------
# Loss + Optimizer
# --------------------------------------------------

image_loss_fn = nn.MSELoss()
message_loss_fn = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# Training
# --------------------------------------------------

model.train()

for iteration in range(1, NUM_ITERATIONS + 1):

    optimizer.zero_grad()

    # Forward
    stego, recovered_message = model(
        cover,
        message
    )

    # Image reconstruction loss
    image_loss = image_loss_fn(
        stego,
        cover
    )

    # Message recovery loss
    message_loss = message_loss_fn(
        recovered_message,
        message
    )

    # Total loss
    total_loss = (
        LAMBDA_IMAGE * image_loss
        +
        LAMBDA_MESSAGE * message_loss
    )

    # Backpropagation
    total_loss.backward()

    optimizer.step()


    # --------------------------------------------------
    # Calculate bit accuracy
    # --------------------------------------------------

    with torch.no_grad():

        predicted_bits = (
            recovered_message >= 0.5
        ).float()

        bit_accuracy = (
            predicted_bits == message
        ).float().mean().item()


    # --------------------------------------------------
    # Print progress
    # --------------------------------------------------

    if iteration == 1 or iteration % 20 == 0:

        print(
            f"Iteration [{iteration:03d}/{NUM_ITERATIONS}] "
            f"| Total Loss: {total_loss.item():.6f} "
            f"| Image Loss: {image_loss.item():.6f} "
            f"| Message Loss: {message_loss.item():.6f} "
            f"| Bit Accuracy: {bit_accuracy * 100:.2f}%"
        )


print("\nOverfitting test completed.")