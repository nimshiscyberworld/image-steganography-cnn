import os
import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

import sys
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

from models.baseline_cnn import BaselineSteganography


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DATASET_ROOT = (
    "/kaggle/input/datasets/nimshipaul/"
    "image-steganography-processed"
)

TRAIN_DIR = os.path.join(DATASET_ROOT, "train")

CHECKPOINT_PATH = (
    "/kaggle/working/improved_baseline_outputs/"
    "improved_baseline_best.pth"
)

IMAGE_SIZE = 256
MESSAGE_BITS = 256

TRAIN_IMAGES = 16
TEST_IMAGES = 16

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
# LOAD IMAGES
# ============================================================

image_files = [
    f for f in os.listdir(TRAIN_DIR)
    if f.lower().endswith(
        (".png", ".jpg", ".jpeg", ".bmp")
    )
]

image_files.sort()


# First 16 = decoder training
# Next 16 = completely different test images

train_files = image_files[:TRAIN_IMAGES]

test_files = image_files[
    TRAIN_IMAGES:
    TRAIN_IMAGES + TEST_IMAGES
]


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def load_images(files):

    images = []

    for filename in files:

        path = os.path.join(
            TRAIN_DIR,
            filename
        )

        image = Image.open(path).convert("L")

        image = transform(image)

        images.append(image)

    return torch.stack(images)


train_covers = load_images(
    train_files
).to(DEVICE)

test_covers = load_images(
    test_files
).to(DEVICE)


# ============================================================
# DIFFERENT RANDOM MESSAGES
# ============================================================

train_messages = torch.randint(
    0,
    2,
    (TRAIN_IMAGES, MESSAGE_BITS),
    device=DEVICE
).float()


test_messages = torch.randint(
    0,
    2,
    (TEST_IMAGES, MESSAGE_BITS),
    device=DEVICE
).float()


# ============================================================
# INFORMATION
# ============================================================

print("=" * 60)
print("DECODER GENERALIZATION TEST")
print("=" * 60)

print("Device:", DEVICE)

print("\nTraining images:")
print(train_files)

print("\nTesting images:")
print(test_files)


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

    state_dict = checkpoint[
        "model_state_dict"
    ]

elif "state_dict" in checkpoint:

    state_dict = checkpoint[
        "state_dict"
    ]

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict
)


print("\nLoaded checkpoint:")
print(CHECKPOINT_PATH)


# ============================================================
# FREEZE MESSAGE EMBEDDER + ENCODER
# ============================================================

for param in model.message_embedder.parameters():
    param.requires_grad = False

for param in model.encoder.parameters():
    param.requires_grad = False


model.message_embedder.eval()
model.encoder.eval()


# ============================================================
# GENERATE TRAINING STEGO IMAGES
# ============================================================

with torch.no_grad():

    train_secret_features = (
        model.message_embedder(
            train_messages
        )
    )

    train_stegos = model.encoder(
        train_covers,
        train_secret_features
    )


# ============================================================
# GENERATE TEST STEGO IMAGES
# ============================================================

with torch.no_grad():

    test_secret_features = (
        model.message_embedder(
            test_messages
        )
    )

    test_stegos = model.encoder(
        test_covers,
        test_secret_features
    )


print("\nStego images generated.")

print(
    "Training stego shape:",
    train_stegos.shape
)

print(
    "Testing stego shape:",
    test_stegos.shape
)


# ============================================================
# TRAIN ONLY DECODER + MESSAGE DECODER
# ============================================================

model.decoder.train()
model.message_decoder.train()


optimizer = torch.optim.Adam(
    list(model.decoder.parameters())
    +
    list(model.message_decoder.parameters()),
    lr=LEARNING_RATE
)


criterion = nn.BCELoss()


# ============================================================
# TRAINING
# ============================================================

for iteration in range(
    1,
    NUM_ITERATIONS + 1
):

    optimizer.zero_grad()

    decoded_features = model.decoder(
        train_stegos
    )

    recovered_messages = (
        model.message_decoder(
            decoded_features
        )
    )

    loss = criterion(
        recovered_messages,
        train_messages
    )

    loss.backward()

    optimizer.step()


    # ========================================================
    # ACCURACY
    # ========================================================

    with torch.no_grad():

        train_predictions = (
            recovered_messages >= 0.5
        ).float()

        train_accuracy = (
            train_predictions ==
            train_messages
        ).float().mean().item() * 100


    if (
        iteration == 1
        or iteration % 25 == 0
        or iteration == NUM_ITERATIONS
    ):

        print(
            f"Iteration "
            f"[{iteration:03d}/{NUM_ITERATIONS}] "
            f"| Loss: {loss.item():.6f} "
            f"| Train Accuracy: "
            f"{train_accuracy:.2f}%"
        )


# ============================================================
# FINAL TEST
# ============================================================

model.decoder.eval()
model.message_decoder.eval()


with torch.no_grad():

    test_decoded_features = model.decoder(
        test_stegos
    )

    test_recovered_messages = (
        model.message_decoder(
            test_decoded_features
        )
    )

    test_predictions = (
        test_recovered_messages >= 0.5
    ).float()

    test_accuracy = (
        test_predictions ==
        test_messages
    ).float().mean().item() * 100


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(
    f"Training Bit Accuracy : "
    f"{train_accuracy:.2f}%"
)

print(
    f"Testing Bit Accuracy  : "
    f"{test_accuracy:.2f}%"
)

print("=" * 60)