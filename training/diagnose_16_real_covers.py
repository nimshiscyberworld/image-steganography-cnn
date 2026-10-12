
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

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
NUM_COVERS = 16
NUM_MESSAGES = 64
STEPS = 3000
LR = 1e-4
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Find and load 16 real images
extensions = {".png", ".jpg", ".jpeg", ".bmp", ".pgm"}
image_paths = sorted(
    p for p in TRAIN_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
)

if len(image_paths) < NUM_COVERS:
    raise RuntimeError(
        f"Found only {len(image_paths)} images in {TRAIN_DIR}"
    )

selected_paths = image_paths[:NUM_COVERS]
covers = []

for path in selected_paths:
    img = Image.open(path).convert("L").resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )
    arr = np.array(img, dtype=np.float32) / 255.0
    covers.append(torch.from_numpy(arr).unsqueeze(0))

covers = torch.stack(covers).to(device)
print("Covers shape:", tuple(covers.shape))
print("First image:", selected_paths[0])

# Create 64 fixed, distinct messages
messages = torch.randint(
    0, 2, (NUM_MESSAGES, MESSAGE_BITS), device=device
).float()

print("Messages shape:", tuple(messages.shape))
print("Unique messages:", torch.unique(messages, dim=0).shape[0])

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

# Fixed mapping: every cover is paired with every message
cover_indices = torch.arange(NUM_COVERS, device=device).repeat_interleave(
    NUM_MESSAGES
)
message_indices = torch.arange(NUM_MESSAGES, device=device).repeat(
    NUM_COVERS
)

all_covers = covers[cover_indices]
all_messages = messages[message_indices]

print("Total fixed pairs:", len(all_covers))

# Train repeatedly on the same fixed pairs
batch_size = 32
num_pairs = len(all_covers)

for step in range(1, STEPS + 1):
    permutation = torch.randperm(num_pairs, device=device)
    total_loss = 0.0
    total_correct = 0
    total_bits = 0

    for start in range(0, num_pairs, batch_size):
        idx = permutation[start:start + batch_size]
        cover_batch = all_covers[idx]
        message_batch = all_messages[idx]

        optimizer.zero_grad()

        message_features = embedder(message_batch)
        stego = encoder(cover_batch, message_features)
        decoded_features = decoder(stego)
        logits = message_decoder(decoded_features)

        if logits.shape != message_batch.shape:
            raise RuntimeError(
                f"Logits shape {logits.shape}, "
                f"target shape {message_batch.shape}"
            )

        loss = criterion(logits, message_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(idx)
        total_correct += (
            (logits.detach() >= 0) == message_batch.bool()
        ).sum().item()
        total_bits += message_batch.numel()

    if step == 1 or step % 100 == 0:
        print(
            f"Step {step:4d}/{STEPS} | "
            f"Loss: {total_loss / num_pairs:.6f} | "
            f"Bit Accuracy: {100 * total_correct / total_bits:.2f}%"
        )

# Evaluate all fixed pairs without updating weights
embedder.eval()
encoder.eval()
decoder.eval()
message_decoder.eval()

total_correct = 0
total_bits = 0
total_exact = 0
total_ber = 0.0
total_eval_loss = 0.0

with torch.no_grad():
    for start in range(0, num_pairs, batch_size):
        end = min(start + batch_size, num_pairs)
        cover_batch = all_covers[start:end]
        message_batch = all_messages[start:end]

        message_features = embedder(message_batch)
        stego = encoder(cover_batch, message_features)
        decoded_features = decoder(stego)
        logits = message_decoder(decoded_features)

        loss = criterion(logits, message_batch)
        pred = (logits >= 0).float()
        errors = (pred != message_batch).float()

        total_eval_loss += loss.item() * len(message_batch)
        total_correct += (pred == message_batch).sum().item()
        total_bits += message_batch.numel()
        total_exact += (errors.sum(dim=1) == 0).sum().item()
        total_ber += errors.mean(dim=1).sum().item()

print("\n===== D8-B FINAL RESULTS =====")
print(f"Evaluation loss: {total_eval_loss / num_pairs:.6f}")
print(f"Bit accuracy: {100 * total_correct / total_bits:.2f}%")
print(f"BER: {total_ber / num_pairs:.6f}")
print(f"Exact message matches: {total_exact}/{num_pairs}")
print(f"Exact match rate: {100 * total_exact / num_pairs:.2f}%")

output_dir = PROJECT_ROOT / "outputs" / "d8b"
output_dir.mkdir(parents=True, exist_ok=True)

torch.save({
    "embedder": embedder.state_dict(),
    "encoder": encoder.state_dict(),
    "decoder": decoder.state_dict(),
    "message_decoder": message_decoder.state_dict(),
    "seed": SEED,
    "num_covers": NUM_COVERS,
    "num_messages": NUM_MESSAGES,
    "steps": STEPS,
    "final_bit_accuracy": total_correct / total_bits,
    "final_exact_matches": total_exact,
}, output_dir / "d8b_16_real_covers.pth")

print("Checkpoint saved to:", output_dir / "d8b_16_real_covers.pth")
