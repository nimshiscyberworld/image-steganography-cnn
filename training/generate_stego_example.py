import os
import torch
import numpy as np
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

IMAGE_PATH = os.path.join(
    DATASET_ROOT,
    "train",
    "boss_00000.png"
)

CHECKPOINT_PATH = (
    "/kaggle/working/improved_baseline_outputs/"
    "improved_baseline_best.pth"
)

OUTPUT_DIR = (
    "/kaggle/working/image-steganography-cnn/"
    "outputs/stego_images"
)

IMAGE_SIZE = 256
MESSAGE_BITS = 256


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD IMAGE
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor()
])


image = Image.open(
    IMAGE_PATH
).convert("L")

cover = transform(image).unsqueeze(0).to(DEVICE)


# ============================================================
# CREATE SECRET MESSAGE
# ============================================================

torch.manual_seed(42)

message = torch.randint(
    0,
    2,
    (1, MESSAGE_BITS),
    device=DEVICE
).float()


print("=" * 60)
print("STEGO IMAGE GENERATION")
print("=" * 60)

print("Device:", DEVICE)
print("Cover:", IMAGE_PATH)
print("Cover shape:", cover.shape)

print("\nSecret message:")
print(message[0].int().cpu().numpy())


# ============================================================
# LOAD MODEL
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

model.eval()


# ============================================================
# GENERATE STEGO IMAGE
# ============================================================

with torch.no_grad():

    secret_feature = (
        model.message_embedder(
            message
        )
    )

    stego = model.encoder(
        cover,
        secret_feature
    )

    # Also recover the message
    decoded_feature = model.decoder(
        stego
    )

    recovered_message = (
        model.message_decoder(
            decoded_feature
        )
    )


# ============================================================
# MESSAGE ACCURACY
# ============================================================

predicted_message = (
    recovered_message >= 0.5
).float()


bit_accuracy = (
    predicted_message == message
).float().mean().item() * 100


# ============================================================
# IMAGE METRICS
# ============================================================

mse = torch.mean(
    (cover - stego) ** 2
).item()


if mse > 0:
    psnr = 10 * np.log10(
        1.0 / mse
    )
else:
    psnr = float("inf")


# ============================================================
# SAVE IMAGES
# ============================================================

cover_array = (
    cover[0, 0]
    .detach()
    .cpu()
    .numpy()
)

stego_array = (
    stego[0, 0]
    .detach()
    .cpu()
    .numpy()
)


cover_array = np.clip(
    cover_array * 255,
    0,
    255
).astype(np.uint8)


stego_array = np.clip(
    stego_array * 255,
    0,
    255
).astype(np.uint8)


cover_output = os.path.join(
    OUTPUT_DIR,
    "example_cover.png"
)

stego_output = os.path.join(
    OUTPUT_DIR,
    "example_stego.png"
)


Image.fromarray(
    cover_array
).save(cover_output)

Image.fromarray(
    stego_array
).save(stego_output)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(
    f"MSE               : {mse:.8f}"
)

print(
    f"PSNR              : {psnr:.2f} dB"
)

print(
    f"Message Bit Acc.  : {bit_accuracy:.2f}%"
)

print("\nSaved files:")

print(
    "Cover:",
    cover_output
)

print(
    "Stego:",
    stego_output
)

print("=" * 60)