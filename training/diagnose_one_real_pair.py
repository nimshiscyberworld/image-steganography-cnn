
import os
import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path

from models.baseline_cnn import (
    MessageEmbedder, Encoder, Decoder, MessageDecoder
)

# Configuration
DATASET_ROOT = Path(
    "/kaggle/input/datasets/nimshipaul/image-steganography-processed"
)
TRAIN_DIR = DATASET_ROOT / "train"
IMAGE_SIZE = 256
MESSAGE_BITS = 256
STEPS = 500
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Find one real image
extensions = {".png", ".jpg", ".jpeg", ".bmp", ".pgm"}
images = [
    p for p in TRAIN_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
]

if not images:
    raise RuntimeError(
        f"No images found in {TRAIN_DIR}. Check your dataset path."
    )

image_path = images[0]
print("Using image:", image_path)

image = Image.open(image_path).convert("L").resize(
    (IMAGE_SIZE, IMAGE_SIZE)
)
cover = torch.from_numpy(
    np.array(image, dtype=np.float32) / 255.0
).unsqueeze(0).unsqueeze(0).to(device)

# Fixed message: do not generate a new one each step
torch.manual_seed(42)
message = torch.randint(
    0, 2, (1, MESSAGE_BITS), device=device
).float()

# Create model
embedder = MessageEmbedder().to(device)
encoder = Encoder().to(device)
decoder = Decoder().to(device)
message_decoder = MessageDecoder().to(device)

params = (
    list(embedder.parameters())
    + list(encoder.parameters())
    + list(decoder.parameters())
    + list(message_decoder.parameters())
)

optimizer = torch.optim.Adam(params, lr=LR)
criterion = nn.BCEWithLogitsLoss()

print("Cover shape:", tuple(cover.shape))
print("Message shape:", tuple(message.shape))
print("Target bit mean:", message.mean().item())

for step in range(1, STEPS + 1):
    optimizer.zero_grad()

    message_features = embedder(message)
    stego = encoder(cover, message_features)
    decoded_features = decoder(stego)
    logits = message_decoder(decoded_features)

    if logits.shape != message.shape:
        raise RuntimeError(
            f"Shape mismatch: logits {logits.shape}, "
            f"message {message.shape}"
        )

    loss = criterion(logits, message)
    loss.backward()

    if step == 1:
        total_grad = sum(
            p.grad.detach().abs().sum().item()
            for p in params if p.grad is not None
        )
        print("Initial gradient magnitude:", total_grad)

    optimizer.step()

    if step % 50 == 0 or step == 1:
        with torch.no_grad():
            accuracy = (
                (logits >= 0).float() == message
            ).float().mean().item() * 100

            print(
                f"Step {step:3d}/{STEPS} | "
                f"Loss: {loss.item():.6f} | "
                f"Bit Accuracy: {accuracy:.2f}%"
            )

print("\nDiagnostic complete.")
