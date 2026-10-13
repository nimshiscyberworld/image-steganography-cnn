

import os
import sys
import csv
import random
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = "/kaggle/working/image-steganography-cnn"
DATASET_ROOT = "/kaggle/input/datasets/nimshipaul/image-steganography-processed"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "canny_guided")

sys.path.insert(0, PROJECT_ROOT)
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4
IMAGE_SIZE = 256
MESSAGE_BITS = 256
NUM_WORKERS = 2

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, flush=True)
print("Dataset:", DATASET_ROOT, flush=True)
print("Output:", OUTPUT_DIR, flush=True)


class CannyImageDataset(Dataset):
    def __init__(self, folder):
        self.files = []

        for root, _, names in os.walk(folder):
            for name in names:
                if name.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".pgm")
                ):
                    self.files.append(os.path.join(root, name))

        self.files.sort()
        if not self.files:
            raise RuntimeError(f"No images found in {folder}")

        print(f"Loaded {len(self.files)} paths from {folder}", flush=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        with Image.open(path) as im:
            im = im.convert("L").resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                Image.Resampling.BILINEAR
            )
            image = np.array(im, dtype=np.uint8)

        # Real Canny edge detection on the cover image.
        edges = cv2.Canny(image, threshold1=100, threshold2=200)

        cover = torch.from_numpy(
            image.astype(np.float32) / 255.0
        ).unsqueeze(0)

        edge_map = torch.from_numpy(
            edges.astype(np.float32) / 255.0
        ).unsqueeze(0)

        return cover, edge_map


def calculate_psnr(cover, stego):
    mse = torch.mean((cover - stego) ** 2, dim=(1, 2, 3))
    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))
    return psnr.mean().item()


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_bits = 0
    correct_bits = 0
    total_exact = 0
    total_images = 0
    total_ber_errors = 0
    psnr_sum = 0.0

    # Reproducible validation messages each epoch.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + 1000)

    with torch.no_grad():
        for cover, edge_map in loader:
            cover = cover.to(device, non_blocking=True)
            edge_map = edge_map.to(device, non_blocking=True)

            message = torch.randint(
                0, 2, (cover.size(0), MESSAGE_BITS),
                generator=generator
            ).float().to(device)

            stego, logits = model(cover, message, edge_map)
            loss = criterion(logits, message)

            predicted = (logits >= 0).float()
            errors = (predicted != message).sum().item()
            batch_bits = message.numel()
            batch_size = cover.size(0)

            total_loss += loss.item() * batch_size
            correct_bits += batch_bits - errors
            total_bits += batch_bits
            total_ber_errors += errors
            total_exact += (predicted == message).all(dim=1).sum().item()
            total_images += batch_size
            psnr_sum += calculate_psnr(cover, stego) * batch_size

    return {
        "loss": total_loss / total_images,
        "bit_accuracy": correct_bits / total_bits,
        "ber": total_ber_errors / total_bits,
        "exact_recovery": total_exact / total_images,
        "psnr": psnr_sum / total_images
    }


def main():
    train_dataset = CannyImageDataset(
        os.path.join(DATASET_ROOT, "train")
    )
    val_dataset = CannyImageDataset(
        os.path.join(DATASET_ROOT, "validation")
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    from models.canny_guided_cnn import CannyGuidedSteganography

    model = CannyGuidedSteganography(
        message_bits=MESSAGE_BITS
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )

    best_val_loss = float("inf")
    history_path = os.path.join(OUTPUT_DIR, "history.csv")

    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "val_loss", "bit_accuracy",
            "ber", "exact_recovery", "psnr"
        ])

    print("\nStarting Canny-Guided CNN training", flush=True)
    print("Epochs:", EPOCHS, "| Batch size:", BATCH_SIZE, flush=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_images = 0

        for batch_index, (cover, edge_map) in enumerate(train_loader):
            cover = cover.to(device, non_blocking=True)
            edge_map = edge_map.to(device, non_blocking=True)

            message = torch.randint(
                0, 2, (cover.size(0), MESSAGE_BITS),
                device=device
            ).float()

            optimizer.zero_grad(set_to_none=True)

            stego, logits = model(cover, message, edge_map)
            loss = criterion(logits, message)

            loss.backward()
            optimizer.step()

            batch_size = cover.size(0)
            train_loss_sum += loss.item() * batch_size
            train_images += batch_size

            if (batch_index + 1) % 100 == 0:
                print(
                    f"Epoch {epoch}/{EPOCHS} | "
                    f"Batch {batch_index + 1}/{len(train_loader)} | "
                    f"Loss {loss.item():.6f}",
                    flush=True
                )

        train_loss = train_loss_sum / train_images
        metrics = evaluate(model, val_loader, criterion)

        print(f"\nEpoch {epoch}/{EPOCHS}", flush=True)
        print(f"Train loss: {train_loss:.6f}", flush=True)
        print(f"Validation loss: {metrics['loss']:.6f}", flush=True)
        print(
            f"Bit accuracy: {metrics['bit_accuracy'] * 100:.2f}%",
            flush=True
        )
        print(f"BER: {metrics['ber']:.6f}", flush=True)
        print(
            f"Exact recovery: {metrics['exact_recovery'] * 100:.2f}%",
            flush=True
        )
        print(f"PSNR: {metrics['psnr']:.2f} dB", flush=True)

        with open(history_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_loss, metrics["loss"],
                metrics["bit_accuracy"], metrics["ber"],
                metrics["exact_recovery"], metrics["psnr"]
            ])

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "seed": SEED,
            "message_bits": MESSAGE_BITS
        }

        torch.save(
            checkpoint,
            os.path.join(OUTPUT_DIR, "last_model.pth")
        )

        if metrics["loss"] < best_val_loss:
            best_val_loss = metrics["loss"]
            torch.save(
                checkpoint,
                os.path.join(OUTPUT_DIR, "best_model.pth")
            )
            print("Saved new best checkpoint.", flush=True)

    print("\nTRAINING FINISHED", flush=True)
    print("History:", history_path, flush=True)
    print("Best checkpoint:", os.path.join(OUTPUT_DIR, "best_model.pth"),
          flush=True)
    print("Last checkpoint:", os.path.join(OUTPUT_DIR, "last_model.pth"),
          flush=True)


if __name__ == "__main__":
    main()
