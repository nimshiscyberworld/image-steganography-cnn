import os
import glob
import random
from PIL import Image
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
IMAGE_SIZE = 256

# Original datasets
BOSSBASE_DIR = r"dataset\BOSSBase"
DIV2K_DIR = r"dataset\DIV2K"

# Processed dataset directories
OUTPUT_DIR = r"dataset\processed"

BOSS_TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
BOSS_VAL_DIR = os.path.join(OUTPUT_DIR, "validation")
BOSS_TEST_DIR = os.path.join(OUTPUT_DIR, "test")

DIV2K_TEST_DIR = os.path.join(OUTPUT_DIR, "external_test")


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for directory in [
    BOSS_TRAIN_DIR,
    BOSS_VAL_DIR,
    BOSS_TEST_DIR,
    DIV2K_TEST_DIR
]:
    os.makedirs(directory, exist_ok=True)


# ============================================================
# FIND BOSSBASE IMAGES
# ============================================================

print("\nSearching for BOSSBase images...")

boss_files = glob.glob(
    os.path.join(BOSSBASE_DIR, "**", "*.pgm"),
    recursive=True
)

print(f"BOSSBase images found: {len(boss_files)}")

if len(boss_files) == 0:
    raise FileNotFoundError(
        "No BOSSBase .pgm files found. Check dataset\\BOSSBase."
    )


# ============================================================
# SHUFFLE BOSSBASE
# ============================================================

random.seed(SEED)
random.shuffle(boss_files)


# ============================================================
# SPLIT BOSSBASE
# ============================================================

train_files = boss_files[:8000]
val_files = boss_files[8000:9000]
test_files = boss_files[9000:10000]

print("\nBOSSBase split:")
print(f"Training   : {len(train_files)}")
print(f"Validation : {len(val_files)}")
print(f"Testing    : {len(test_files)}")


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(input_path, output_path):
    """
    Open image, convert to grayscale,
    resize to 256x256 and save as PNG.
    """

    with Image.open(input_path) as img:

        img = img.convert("L")

        img = img.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.LANCZOS
        )

        img.save(output_path, format="PNG")


# ============================================================
# PROCESS BOSSBASE SPLIT
# ============================================================

def process_boss_split(files, output_dir, split_name):

    print(f"\nProcessing BOSSBase {split_name}...")

    for index, file_path in enumerate(tqdm(files)):

        output_path = os.path.join(
            output_dir,
            f"boss_{index:05d}.png"
        )

        process_image(file_path, output_path)


process_boss_split(
    train_files,
    BOSS_TRAIN_DIR,
    "training"
)

process_boss_split(
    val_files,
    BOSS_VAL_DIR,
    "validation"
)

process_boss_split(
    test_files,
    BOSS_TEST_DIR,
    "testing"
)


# ============================================================
# FIND DIV2K IMAGES
# ============================================================

print("\nSearching for DIV2K images...")

div2k_files = glob.glob(
    os.path.join(DIV2K_DIR, "**", "*.png"),
    recursive=True
)

print(f"DIV2K images found: {len(div2k_files)}")

if len(div2k_files) == 0:
    raise FileNotFoundError(
        "No DIV2K PNG files found. Check dataset\\DIV2K."
    )


# ============================================================
# PROCESS DIV2K
# ============================================================

print("\nProcessing DIV2K external test dataset...")


def center_crop(image, size):

    width, height = image.size

    left = (width - size) // 2
    top = (height - size) // 2

    right = left + size
    bottom = top + size

    return image.crop(
        (left, top, right, bottom)
    )


for index, file_path in enumerate(tqdm(div2k_files)):

    with Image.open(file_path) as img:

        # Convert RGB → grayscale
        img = img.convert("L")

        # Make sure the image is large enough
        if img.width < IMAGE_SIZE or img.height < IMAGE_SIZE:
            continue

        # Center crop
        img = center_crop(
            img,
            IMAGE_SIZE
        )

        output_path = os.path.join(
            DIV2K_TEST_DIR,
            f"div2k_{index:04d}.png"
        )

        img.save(
            output_path,
            format="PNG"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

def count_png(directory):

    return len(
        glob.glob(
            os.path.join(directory, "*.png")
        )
    )


print("\n")
print("=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print(f"BOSSBase Train      : {count_png(BOSS_TRAIN_DIR)}")
print(f"BOSSBase Validation : {count_png(BOSS_VAL_DIR)}")
print(f"BOSSBase Test       : {count_png(BOSS_TEST_DIR)}")
print(f"DIV2K External Test : {count_png(DIV2K_TEST_DIR)}")

print("=" * 60)

print("\nProcessed dataset location:")
print(OUTPUT_DIR)