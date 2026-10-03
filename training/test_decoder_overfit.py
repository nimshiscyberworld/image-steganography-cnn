
import os
import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.baseline_cnn import BaselineSteganography


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_ROOT = "/kaggle/input/datasets/nimshipaul/image-steganography-processed"
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")

CHECKPOINT_PATH = (
    "/kaggle/working/improved_baseline_outputs/"
    "improved_baseline_best.pth"
)

IMAGE_SIZE = 256
MESSAGE_BITS = 256

NUM_IMAGES = 16
NUM_ITERATIONS = 500
LEARNING_RATE = 1e-3

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ============================================================
# LOAD 16 FIXED IMAGES
# ============================================================

image_files = [
    f for f in os.listdir(TRAIN_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
]

image_files.sort()

selected_files = image_files[:NUM_IMAGES]

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


covers = []

for filename in selected_files:

    path = os.path.join(TRAIN_DIR, filename)

    image = Image.open(path).convert("L")
    image = transform(image)

    covers.append(image)


covers = torch.stack(covers).to(DEVICE)


# ============================================================
# FIXED RANDOM MESSAGES
# ============================================================

messages = torch.randint(
    0,
    2,
    (NUM_IMAGES, MESSAGE_BITS),
    device=DEVICE
).float()


print("=" * 60)
print("DECODER-ONLY OVERFIT TEST")
print("=" * 60)

print("Device:", DEVICE)
print("Number of images:", NUM_IMAGES)
print("Message bits:", MESSAGE_BITS)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = BaselineSteganography(
    message_bits=MESSAGE_BITS
).to(DEVICE)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)


if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]

elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]

else:
    state_dict = checkpoint


model.load_state_dict(state_dict)

print("\nLoaded checkpoint:")
print(CHECKPOINT_PATH)


# ============================================================
# FREEZE ENCODER
# ============================================================

for param in model.message_embedder.parameters():
    param.requires_grad = False

for param in model.encoder.parameters():
    param.requires_grad = False


model.message_embedder.eval()
model.encoder.eval()


# ============================================================
# GENERATE FIXED STEGO IMAGES
# ============================================================

with torch.no_grad():

    secret_features = model.message_embedder(messages)

    stego_images = model.encoder(
        covers,
        secret_features
    )


print("\nGenerated fixed stego images:")
print("Stego shape:", stego_images.shape)


# ============================================================
# TRAIN ONLY DECODER + MESSAGE DECODER
# ============================================================

model.decoder.train()
model.message_decoder.train()


optimizer = torch.optim.Adam(
    list(model.decoder.parameters()) +
    list(model.message_decoder.parameters()),
    lr=LEARNING_RATE
)

criterion = nn.BCELoss()


# ============================================================
# TRAINING LOOP
# ============================================================

for iteration in range(1, NUM_ITERATIONS + 1):

    optimizer.zero_grad()

    # Decode fixed stego images
    decoded_features = model.decoder(stego_images)

    recovered_messages = model.message_decoder(
        decoded_features
    )

    loss = criterion(
        recovered_messages,
        messages
    )

    loss.backward()

    optimizer.step()


    # --------------------------------------------------------
    # BIT ACCURACY
    # --------------------------------------------------------

    with torch.no_grad():

        predictions = (
            recovered_messages >= 0.5
        ).float()

        accuracy = (
            predictions == messages
        ).float().mean().item() * 100


    if (
        iteration == 1
        or iteration % 25 == 0
        or iteration == NUM_ITERATIONS
    ):

        print(
            f"Iteration [{iteration:03d}/{NUM_ITERATIONS}] "
            f"| Loss: {loss.item():.6f} "
            f"| Bit Accuracy: {accuracy:.2f}%"
        )


print("\n" + "=" * 60)
print("DECODER-ONLY TEST COMPLETED")
print("=" * 60)