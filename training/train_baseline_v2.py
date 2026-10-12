
import os
import random
import csv
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_ROOT = (
    "/kaggle/input/datasets/nimshipaul/"
    "image-steganography-processed"
)

TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "validation")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "baseline_v2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(OUTPUT_DIR, "last_model.pth")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "training_results.csv")

IMAGE_SIZE = 256
MESSAGE_BITS = 256
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

print("=" * 60)
print("BASELINE V2: CNN IMAGE STEGANOGRAPHY")
print("=" * 60)
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Training directory:", TRAIN_DIR)
print("Validation directory:", VAL_DIR)
print("Output directory:", OUTPUT_DIR)


# ============================================================
# 3. DATASET
# ============================================================

class ImageDataset(Dataset):
    def __init__(self, directory):
        if not os.path.isdir(directory):
            raise FileNotFoundError(
                f"Dataset directory not found: {directory}"
            )

        extensions = (".png", ".jpg", ".jpeg", ".bmp", ".pgm")

        self.image_paths = []

        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.lower().endswith(extensions):
                    self.image_paths.append(
                        os.path.join(root, filename)
                    )

        self.image_paths.sort()

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No supported images found in: {directory}"
            )

        print(f"Found {len(self.image_paths)} images in {directory}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        with Image.open(image_path) as image:
            image = image.convert("L")
            image = image.resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                Image.Resampling.BILINEAR
            )

            array = np.asarray(image, dtype=np.float32) / 255.0
            array = np.ascontiguousarray(array)

        # Shape: [1, 256, 256]
        return torch.from_numpy(array).unsqueeze(0)


# ============================================================
# 4. DATA LOADERS
# ============================================================

train_dataset = ImageDataset(TRAIN_DIR)
val_dataset = ImageDataset(VAL_DIR)

loader_kwargs = {
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "pin_memory": torch.cuda.is_available(),
}

train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    drop_last=True,
    **loader_kwargs
)

val_loader = DataLoader(
    val_dataset,
    shuffle=False,
    drop_last=False,
    **loader_kwargs
)

if len(train_loader) == 0:
    raise RuntimeError(
        "Training DataLoader is empty. Check dataset size and batch size."
    )


# ============================================================
# 5. LOAD YOUR EXISTING CNN ARCHITECTURE
# ============================================================

import sys
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.baseline_cnn import (
    MessageEmbedder,
    Encoder,
    Decoder,
    MessageDecoder,
)

message_embedder = MessageEmbedder(
    message_bits=MESSAGE_BITS,
    feature_channels=32
).to(DEVICE)

encoder = Encoder().to(DEVICE)
decoder = Decoder().to(DEVICE)

message_decoder = MessageDecoder(
    message_bits=MESSAGE_BITS
).to(DEVICE)

all_parameters = (
    list(message_embedder.parameters())
    + list(encoder.parameters())
    + list(decoder.parameters())
    + list(message_decoder.parameters())
)

optimizer = torch.optim.Adam(
    all_parameters,
    lr=LEARNING_RATE
)

# MessageDecoder returns logits, so use BCEWithLogitsLoss.
criterion = nn.BCEWithLogitsLoss()

print("\nCNN components loaded successfully.")
print("Message length:", MESSAGE_BITS, "bits")
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)


# ============================================================
# 6. METRICS
# ============================================================

def calculate_metrics(logits, messages):
    predictions = (logits >= 0).float()

    correct_bits = (predictions == messages).sum().item()
    total_bits = messages.numel()

    bit_accuracy = correct_bits / total_bits
    ber = 1.0 - bit_accuracy

    correct_messages = (
        (predictions == messages).all(dim=1).sum().item()
    )

    exact_rate = correct_messages / messages.size(0)

    return bit_accuracy, ber, correct_messages, exact_rate


def calculate_psnr(stego, cover):
    # Images are normalized to [0, 1].
    mse_per_image = (
        (stego.detach() - cover).pow(2)
        .mean(dim=(1, 2, 3))
    )

    psnr = 10.0 * torch.log10(
        1.0 / mse_per_image.clamp_min(1e-10)
    )

    return psnr.mean().item()


# ============================================================
# 7. FORWARD PASS
# ============================================================

def run_model(covers, messages):
    message_features = message_embedder(messages)

    stego_images = encoder(covers, message_features)

    secret_features = decoder(stego_images)

    recovered_logits = message_decoder(secret_features)

    return stego_images, recovered_logits


# ============================================================
# 8. TRAINING ONE EPOCH
# ============================================================

def train_one_epoch():
    message_embedder.train()
    encoder.train()
    decoder.train()
    message_decoder.train()

    total_loss = 0.0
    total_images = 0
    total_correct_bits = 0
    total_bits = 0
    total_exact = 0
    total_psnr = 0.0

    for covers in train_loader:
        covers = covers.to(
            DEVICE,
            non_blocking=True
        )

        batch_size = covers.size(0)

        # Fresh random messages during training.
        messages = torch.randint(
            0,
            2,
            (batch_size, MESSAGE_BITS),
            dtype=torch.float32,
            device=DEVICE
        )

        optimizer.zero_grad(set_to_none=True)

        stego_images, recovered_logits = run_model(
            covers, messages
        )

        loss = criterion(recovered_logits, messages)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            accuracy, _, exact, _ = calculate_metrics(
                recovered_logits, messages
            )

            total_loss += loss.item() * batch_size
            total_correct_bits += accuracy * messages.numel()
            total_bits += messages.numel()
            total_exact += exact
            total_psnr += calculate_psnr(stego_images, covers)
            total_images += batch_size

    return {
        "loss": total_loss / total_images,
        "bit_accuracy": total_correct_bits / total_bits,
        "ber": 1.0 - total_correct_bits / total_bits,
        "exact_rate": total_exact / total_images,
        "psnr": total_psnr / len(train_loader),
        "images": total_images,
    }


# ============================================================
# 9. VALIDATION WITH REPRODUCIBLE MESSAGES
# ============================================================

@torch.no_grad()
def validate():
    message_embedder.eval()
    encoder.eval()
    decoder.eval()
    message_decoder.eval()

    total_loss = 0.0
    total_images = 0
    total_correct_bits = 0
    total_bits = 0
    total_exact = 0
    total_psnr = 0.0

    # Reset this generator every epoch. Because validation data
    # order and batch sizes are fixed, the validation messages
    # are identical across epochs.
    validation_generator = torch.Generator(device="cpu")
    validation_generator.manual_seed(SEED + 1000)

    for covers in val_loader:
        covers = covers.to(
            DEVICE,
            non_blocking=True
        )

        batch_size = covers.size(0)

        messages = torch.randint(
            0,
            2,
            (batch_size, MESSAGE_BITS),
            generator=validation_generator,
            dtype=torch.int64,
            device="cpu"
        ).float().to(DEVICE)

        stego_images, recovered_logits = run_model(
            covers, messages
        )

        loss = criterion(recovered_logits, messages)

        accuracy, _, exact, _ = calculate_metrics(
            recovered_logits, messages
        )

        total_loss += loss.item() * batch_size
        total_correct_bits += accuracy * messages.numel()
        total_bits += messages.numel()
        total_exact += exact
        total_psnr += calculate_psnr(stego_images, covers)
        total_images += batch_size

    return {
        "loss": total_loss / total_images,
        "bit_accuracy": total_correct_bits / total_bits,
        "ber": 1.0 - total_correct_bits / total_bits,
        "exact_rate": total_exact / total_images,
        "psnr": total_psnr / len(val_loader),
        "images": total_images,
    }


# ============================================================
# 10. CHECKPOINT SAVING
# ============================================================

def model_state():
    return {
        "embedder": message_embedder.state_dict(),
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "message_decoder": message_decoder.state_dict(),
    }


def save_checkpoint(path, epoch, train_metrics, val_metrics,
                    include_optimizer=False):
    checkpoint = {
        **model_state(),
        "epoch": epoch,
        "seed": SEED,
        "message_bits": MESSAGE_BITS,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }

    if include_optimizer:
        checkpoint["optimizer"] = optimizer.state_dict()

    torch.save(checkpoint, path)


# ============================================================
# 11. MAIN TRAINING LOOP
# ============================================================

best_val_loss = float("inf")

csv_fields = [
    "epoch",
    "train_loss",
    "train_bit_accuracy",
    "train_ber",
    "train_exact_rate",
    "train_psnr",
    "val_loss",
    "val_bit_accuracy",
    "val_ber",
    "val_exact_rate",
    "val_psnr",
]

with open(RESULTS_CSV, "w", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    writer.writeheader()

    for epoch in range(1, EPOCHS + 1):
        print("\n" + "=" * 60)
        print(f"EPOCH {epoch}/{EPOCHS}")
        print("=" * 60)

        train_metrics = train_one_epoch()
        val_metrics = validate()

        print("\nTRAIN")
        print(f"Loss:           {train_metrics['loss']:.6f}")
        print(f"Bit accuracy:   {train_metrics['bit_accuracy'] * 100:.4f}%")
        print(f"BER:            {train_metrics['ber']:.6f}")
        print(f"Exact recovery: {train_metrics['exact_rate'] * 100:.4f}%")
        print(f"Average PSNR:   {train_metrics['psnr']:.2f} dB")

        print("\nVALIDATION")
        print(f"Loss:           {val_metrics['loss']:.6f}")
        print(f"Bit accuracy:   {val_metrics['bit_accuracy'] * 100:.4f}%")
        print(f"BER:            {val_metrics['ber']:.6f}")
        print(f"Exact recovery: {val_metrics['exact_rate'] * 100:.4f}%")
        print(f"Average PSNR:   {val_metrics['psnr']:.2f} dB")

        writer.writerow({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_bit_accuracy": train_metrics["bit_accuracy"],
            "train_ber": train_metrics["ber"],
            "train_exact_rate": train_metrics["exact_rate"],
            "train_psnr": train_metrics["psnr"],
            "val_loss": val_metrics["loss"],
            "val_bit_accuracy": val_metrics["bit_accuracy"],
            "val_ber": val_metrics["ber"],
            "val_exact_rate": val_metrics["exact_rate"],
            "val_psnr": val_metrics["psnr"],
        })

        csv_file.flush()

        # Save latest weights and optimizer state for later resuming.
        save_checkpoint(
            LAST_MODEL_PATH,
            epoch,
            train_metrics,
            val_metrics,
            include_optimizer=True
        )

        # Keep the best model according to validation loss.
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            save_checkpoint(
                BEST_MODEL_PATH,
                epoch,
                train_metrics,
                val_metrics,
                include_optimizer=False
            )

            print("\nBest model updated.")

        print("\nSaved latest checkpoint:", LAST_MODEL_PATH)
        print("Best validation loss so far:", f"{best_val_loss:.6f}")


print("\n" + "=" * 60)
print("BASELINE V2 TRAINING FINISHED")
print("=" * 60)
print("Best model:", BEST_MODEL_PATH)
print("Last model:", LAST_MODEL_PATH)
print("Training results:", RESULTS_CSV)
