import os
import sys
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Allow importing models from project root
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from models.baseline_cnn import BaselineSteganography


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = "/kaggle/input/datasets/nimshipaul/image-steganography-processed"

IMAGE_SIZE = 256
MESSAGE_BITS = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("MESSAGE INFLUENCE DIAGNOSTIC")
print("=" * 60)
print("Device:", DEVICE)


# ============================================================
# LOAD ONE IMAGE
# ============================================================

train_dir = os.path.join(DATASET_ROOT, "train")

image_files = [
    f for f in os.listdir(train_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
]

image_files.sort()

if len(image_files) == 0:
    raise RuntimeError("No images found!")

image_path = os.path.join(train_dir, image_files[0])

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

cover = Image.open(image_path).convert("L")
cover = transform(cover).unsqueeze(0).to(DEVICE)

print("\nCover image:")
print("Shape:", cover.shape)
print("Image:", image_files[0])


# ============================================================
# CREATE TWO VERY DIFFERENT MESSAGES
# ============================================================

message_A = torch.zeros(
    (1, MESSAGE_BITS),
    dtype=torch.float32,
    device=DEVICE
)

message_B = torch.ones(
    (1, MESSAGE_BITS),
    dtype=torch.float32,
    device=DEVICE
)

print("\nMessages:")
print("Message A: all 0")
print("Message B: all 1")


# ============================================================
# LOAD TRAINED MODEL CHECKPOINT
# ============================================================

CHECKPOINT_PATH = (
    "/kaggle/working/improved_baseline_outputs/"
    "improved_baseline_best.pth"
)

# Create model ONCE
model = BaselineSteganography(
    message_bits=MESSAGE_BITS
).to(DEVICE)

# Load checkpoint
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

# Handle different checkpoint formats
if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

# Load trained weights
model.load_state_dict(state_dict)

# Evaluation mode
model.eval()

print("\nLoaded trained checkpoint:")
print(CHECKPOINT_PATH)

print("\nModel parameters:")
print(sum(p.numel() for p in model.parameters()))


# ============================================================
# FORWARD PASS
# ============================================================

with torch.no_grad():

    stego_A, recovered_A = model(
        cover,
        message_A
    )

    stego_B, recovered_B = model(
        cover,
        message_B
    )


# ============================================================
# CHECK OUTPUT SHAPES
# ============================================================

print("\nOutput shapes:")

print("Stego A:", stego_A.shape)
print("Stego B:", stego_B.shape)

print("Recovered A:", recovered_A.shape)
print("Recovered B:", recovered_B.shape)


# ============================================================
# 1. DOES THE MESSAGE AFFECT THE STEGO IMAGE?
# ============================================================

stego_difference = torch.mean(
    torch.abs(stego_A - stego_B)
).item()

print("\n" + "=" * 60)
print("TEST 1: MESSAGE INFLUENCE ON STEGO IMAGE")
print("=" * 60)

print("Mean absolute difference:")
print(stego_difference)

if stego_difference < 1e-6:
    print("WARNING: Stego images are almost identical.")
    print("The model may be ignoring the message.")
else:
    print("GOOD: Different messages produce different stego images.")


# ============================================================
# 2. CHECK RECOVERED MESSAGE DIFFERENCE
# ============================================================

recovered_difference = torch.mean(
    torch.abs(recovered_A - recovered_B)
).item()

print("\n" + "=" * 60)
print("TEST 2: RECOVERED MESSAGE DIFFERENCE")
print("=" * 60)

print("Mean absolute difference:")
print(recovered_difference)


# ============================================================
# 3. CHECK BIT ACCURACY
# ============================================================

pred_A = (recovered_A >= 0.5).float()
pred_B = (recovered_B >= 0.5).float()

accuracy_A = (
    (pred_A == message_A).float().mean().item()
)

accuracy_B = (
    (pred_B == message_B).float().mean().item()
)

print("\n" + "=" * 60)
print("TEST 3: MESSAGE RECOVERY")
print("=" * 60)

print(f"Message A accuracy: {accuracy_A * 100:.2f}%")
print(f"Message B accuracy: {accuracy_B * 100:.2f}%")


# ============================================================
# 4. CHECK GRADIENT FLOW
# ============================================================

model.train()

criterion = nn.BCELoss()

stego, recovered = model(
    cover,
    message_A
)

loss = criterion(
    recovered,
    message_A
)

model.zero_grad()

loss.backward()

print("\n" + "=" * 60)
print("TEST 4: GRADIENT FLOW")
print("=" * 60)

print("Message-related gradients:")

gradient_found = False

for name, parameter in model.named_parameters():

    if parameter.grad is not None:

        grad_mean = parameter.grad.abs().mean().item()

        if grad_mean > 0:
            gradient_found = True

        if (
            "message" in name.lower()
            or "decoder" in name.lower()
            or "embed" in name.lower()
        ):
            print(
                f"{name:50s} "
                f"gradient = {grad_mean:.10e}"
            )

if gradient_found:
    print("\nGOOD: Gradients are flowing.")
else:
    print("\nWARNING: No useful gradients detected.")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DIAGNOSTIC SUMMARY")
print("=" * 60)

print("Stego difference :", stego_difference)
print("Recovered diff   :", recovered_difference)
print(f"Message A acc.   : {accuracy_A * 100:.2f}%")
print(f"Message B acc.   : {accuracy_B * 100:.2f}%")
print("Gradient flow    :", gradient_found)

print("\nDiagnostic completed.")
