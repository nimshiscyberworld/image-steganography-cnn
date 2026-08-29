import os
import glob
from PIL import Image
from collections import Counter


DATASETS = {
    "DIV2K": r"dataset\DIV2K",
    "BOSSBase": r"dataset\BOSSBase"
}


def analyze_dataset(name, path):
    print("\n" + "=" * 60)
    print(f"{name} DATASET")
    print("=" * 60)

    extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.pgm"]

    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(path, "**", ext), recursive=True))

    print(f"Total images found : {len(files)}")

    if not files:
        print("No images found. Check the dataset path.")
        return

    modes = Counter()
    sizes = Counter()
    formats = Counter()
    corrupted = []

    for i, file in enumerate(files):

        try:
            with Image.open(file) as img:
                img.verify()

            with Image.open(file) as img:
                modes[img.mode] += 1
                sizes[img.size] += 1
                formats[img.format] += 1

        except Exception:
            corrupted.append(file)

    widths = [size[0] for size in sizes]
    heights = [size[1] for size in sizes]

    print(f"Image formats      : {dict(formats)}")
    print(f"Image modes         : {dict(modes)}")

    print(f"Minimum width      : {min(widths)}")
    print(f"Maximum width      : {max(widths)}")
    print(f"Minimum height     : {min(heights)}")
    print(f"Maximum height     : {max(heights)}")

    print(f"Different sizes    : {len(sizes)}")
    print(f"Corrupted images   : {len(corrupted)}")

    print("\nMost common resolutions:")

    for size, count in sizes.most_common(10):
        print(f"  {size}: {count} images")


for name, path in DATASETS.items():
    analyze_dataset(name, path)