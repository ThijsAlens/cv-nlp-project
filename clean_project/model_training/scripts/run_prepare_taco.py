"""
TACO dataset preparation runner.

Downloads the TACO (Trash Annotations in Context) dataset annotations and images
from the internet, then converts them to YOLO format with a train/val/test split.

This script is only needed if you want to (re-)incorporate TACO data into training.
For the default Totaal_dataset workflow this script is not used.

Edit the CONFIG block below before running.

Usage:
  uv run python scripts/run_prepare_taco.py
"""

import sys
from pathlib import Path

# Add 'src' to the import path so 'waste_detector' can be found when
# running directly without installing the package.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from waste_detector.data.taco_manager import TacoDatasetManager

# ---------------------------------------------------------------
# Configuration -- edit before running
# ---------------------------------------------------------------

# Root directory where TACO raw data and the prepared dataset will be stored.
# The manager creates 'data/taco/raw/' and 'data/taco/prepared/' inside this path.
TACO_ROOT = _PROJECT_ROOT

# Maximum number of images to download. Set to None to download all (~1500).
MAX_IMAGES = None

# Number of parallel download workers.
NUM_WORKERS = 8

# Name for the prepared YOLO dataset subfolder inside 'data/taco/prepared/'.
OUTPUT_NAME = "taco_yolo"

# Path to a label map JSON if you want only a subset of TACO classes.
# Set to None to use all 60 TACO categories.
# Example file format: {"classes": ["Bottle", "Can", "Cigarette"]}
LABEL_MAP_PATH = None

# Train / val / test split ratios (must sum to 1.0).
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Minimum bounding box size in pixels to include (smaller boxes are skipped).
MIN_BOX_PIXELS = 8

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main() -> None:
    manager = TacoDatasetManager(TACO_ROOT)

    # --- Step 1: download annotation JSON from GitHub ---
    print("Step 1/3: Downloading TACO annotations ...")
    manager.download_annotations()

    # Write a category summary so you can inspect available classes before choosing a label map.
    print("Writing category summary ...")
    summary_path = manager.write_category_summary()
    print(f"Review categories at: {summary_path}")

    # --- Step 2: download images ---
    print("\nStep 2/3: Downloading images ...")
    manager.download_images(max_images=MAX_IMAGES, num_workers=NUM_WORKERS)

    # --- Step 3: convert to YOLO format ---
    label_map = Path(LABEL_MAP_PATH) if LABEL_MAP_PATH else None
    print("\nStep 3/3: Converting to YOLO format ...")
    dataset_yaml = manager.prepare_yolo_dataset(
        output_name=OUTPUT_NAME,
        label_map_path=label_map,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        min_box_pixels=MIN_BOX_PIXELS,
    )

    print(f"\nDone. Dataset YAML: {dataset_yaml}")
    print("Point 'dataset.path' in 'config/train_config.yaml' to the dataset folder to use it.")


if __name__ == "__main__":
    main()
