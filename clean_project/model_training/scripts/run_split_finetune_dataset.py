"""
Split the FINALE_TESTSET_merged real-world images into train/val/test folders
for fine-tuning a pretrained checkpoint, then write a manifest that records
exactly which image filenames landed in each split.

The original 'data/FINALE_TESTSET_merged/' is left untouched. A new sibling
folder is created with the split layout that 'load_dataset_spec' already
auto-discovers. The manifest JSON exists so the held-out test images can be
audited later when running 'run_finetune_evaluate.py'.

Usage:
  uv run python scripts/run_split_finetune_dataset.py
"""

import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add 'src' to the import path so 'waste_detector' can be found when this
# script is run directly (without installing the package).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from waste_detector.utils.io import ensure_dir, write_json, write_yaml

# ---------------------------------------------------------------
# Configuration. Edit these constants directly; no CLI flags by project convention.
# ---------------------------------------------------------------

# Source dataset: the 200 real-world images, currently all under 'test/'.
SOURCE_DATASET = _PROJECT_ROOT / "data" / "FINALE_TESTSET_merged"

# Destination dataset: a new folder with proper train/val/test splits.
# The original SOURCE_DATASET is left untouched so it can still be used as a
# pure evaluation set by the v4-2 (pre-finetune) checkpoint if needed.
OUTPUT_DATASET = _PROJECT_ROOT / "data" / "FINALE_TESTSET_finetune_split"

# Subfolder of SOURCE_DATASET that holds the source images and labels.
# 'data/FINALE_TESTSET_merged/data.yaml' currently points every split at
# 'test/images', so the only real source folder is the 'test/' one.
SOURCE_SPLIT_DIR = "test"

# Class names. Used for the generated 'data.yaml' and for the manifest record.
# Must match the source 'data.yaml' (sanity-checked at runtime).
CLASSES = ["Glass", "Metal", "Plastic", "Paper"]

# Split ratios as fractions of the source image count. Must sum to 1.0.
# 50/10/40 of 200 gives 100 train / 20 val / 80 test for this run.
SPLIT_RATIOS = {"train": 0.50, "val": 0.10, "test": 0.40}

# Fixed RNG seed so the same source folder always produces the same split.
# Change this if you genuinely want a different split; never change it just
# to "shake things up" mid-experiment because that breaks reproducibility.
RANDOM_SEED = 42

# Image extensions to consider as sources. Case-insensitive on Windows but
# match on disk anyway for portability.
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _list_source_images(source_split: Path) -> list[Path]:
    """
    Return the list of image paths under 'source_split/images/' that have
    a recognised extension. Sorted for determinism before shuffling.
    """
    images_dir = source_split / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Source images directory not found: {images_dir}")

    # Sorted ensures the seeded shuffle is fully reproducible regardless of OS.
    found = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not found:
        raise RuntimeError(f"No images with extensions {IMAGE_EXTS} found in {images_dir}")
    return found


def _partition_counts(total: int, ratios: dict) -> dict:
    """
    Convert split ratios into exact integer counts that sum to 'total'.

    Floors the counts for every split except the last, then hands the remainder
    to the last split so the counts always add up to 'total', even when the
    ratios give fractional counts (e.g. 0.10 * 47 -> 4.7).
    """
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError(f"SPLIT_RATIOS must sum to 1.0, got {sum(ratios.values())}")

    # Compute floored counts for all but the last split, then take whatever
    # is left for the last one so the partition is exact.
    keys = list(ratios.keys())
    counts: dict = {}
    running = 0
    for key in keys[:-1]:
        c = int(total * ratios[key])
        counts[key] = c
        running += c
    counts[keys[-1]] = total - running
    return counts


def _validate_source_yaml(source_dataset: Path) -> None:
    """
    Sanity-check the source 'data.yaml' has the same class list we expect.
    A mismatch here usually means the source dataset was regenerated with a
    different class layout, in which case the labels in this split would
    reference the wrong class ids.
    """
    src_yaml = source_dataset / "data.yaml"
    if not src_yaml.is_file():
        raise FileNotFoundError(f"Missing source data.yaml: {src_yaml}")

    # Parse the YAML manually rather than depending on the dataset spec loader,
    # because the source data.yaml here points every split at 'test/images',
    # which would confuse 'load_dataset_spec'.
    from waste_detector.utils.io import read_yaml

    payload = read_yaml(src_yaml)
    raw_names = payload.get("names", [])
    if isinstance(raw_names, dict):
        # Sort by key so dict-style class lists become an ordered list.
        names = [str(raw_names[k]) for k in sorted(raw_names.keys())]
    else:
        names = [str(n) for n in raw_names]

    if names != CLASSES:
        raise ValueError(
            f"Source dataset class list does not match CLASSES.\n"
            f"  source data.yaml names: {names}\n"
            f"  expected CLASSES:       {CLASSES}\n"
            f"Update CLASSES at the top of this script or re-check the source dataset."
        )


def _copy_pair(image_path: Path, source_split: Path, dest_split: Path) -> bool:
    """
    Copy one image and its matching YOLO label file into 'dest_split'.

    Returns True if the pair was copied, False if the label was missing
    (image is skipped with a warning so a stray unlabelled file does not
    break the whole split).
    """
    # YOLO labels live in a sibling 'labels/' folder with the same stem.
    label_path = source_split / "labels" / f"{image_path.stem}.txt"
    if not label_path.is_file():
        print(f"  WARNING: skipping image with no label file: {image_path.name}")
        return False

    # 'copy2' preserves timestamps and permission bits.
    shutil.copy2(image_path, dest_split / "images" / image_path.name)
    shutil.copy2(label_path, dest_split / "labels" / label_path.name)
    return True


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main() -> None:
    # --- Refuse to overwrite an existing output folder ---
    # Re-running the splitter on an existing folder would either duplicate
    # files or silently produce a split that no longer matches the existing
    # manifest. The user has to delete the folder explicitly to re-split.
    if OUTPUT_DATASET.exists():
        raise FileExistsError(
            f"Output dataset already exists: {OUTPUT_DATASET}\n"
            f"Delete it manually if you want to re-split. (Keeping it prevents\n"
            f"accidental loss of an existing 'split_manifest.json'.)"
        )

    # --- Sanity-check the source dataset ---
    _validate_source_yaml(SOURCE_DATASET)
    source_split = SOURCE_DATASET / SOURCE_SPLIT_DIR
    images = _list_source_images(source_split)
    total = len(images)
    print(f"Found {total} source images under: {source_split / 'images'}")

    # --- Reproducible shuffle ---
    # Use a local RNG so the global random state stays untouched.
    rng = random.Random(RANDOM_SEED)
    shuffled = list(images)
    rng.shuffle(shuffled)

    # --- Partition into exact counts ---
    counts = _partition_counts(total, SPLIT_RATIOS)
    print(f"Splitting {total} into: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    # Slice the shuffled list into the configured splits.
    splits_files: dict = {}
    cursor = 0
    for split_name, count in counts.items():
        splits_files[split_name] = shuffled[cursor : cursor + count]
        cursor += count

    # --- Create destination folders and copy files ---
    print(f"Writing split to: {OUTPUT_DATASET}")
    copied_counts: dict = {}
    for split_name, files in splits_files.items():
        dest_split = OUTPUT_DATASET / split_name
        ensure_dir(dest_split / "images")
        ensure_dir(dest_split / "labels")

        # Copy each image+label pair; track the filenames that were actually
        # written (so the manifest reflects reality, not intent).
        copied: list[str] = []
        for img_path in files:
            if _copy_pair(img_path, source_split, dest_split):
                copied.append(img_path.name)
        copied_counts[split_name] = len(copied)
        # Replace the Path objects with plain filenames for the manifest.
        splits_files[split_name] = copied
        print(f"  {split_name}: copied {len(copied)} images")

    # --- Write the dataset 'data.yaml' for Ultralytics ---
    # The structure matches 'Totaal_dataset_merged/data.yaml' so that
    # 'load_dataset_spec' auto-discovers the splits without any tweaks.
    data_yaml_payload = {
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(CLASSES),
        "names": list(CLASSES),
    }
    write_yaml(OUTPUT_DATASET / "data.yaml", data_yaml_payload)
    print(f"Wrote: {OUTPUT_DATASET / 'data.yaml'}")

    # --- Write the split manifest ---
    # The 'files' lists are the record of which source images landed where.
    # The 'test' list is the one that matters most, since it documents which
    # 80 images were held out from fine-tuning.
    manifest = {
        "source_dataset": str(SOURCE_DATASET.resolve()),
        "source_split": SOURCE_SPLIT_DIR,
        "output_dataset": str(OUTPUT_DATASET.resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "split_ratios": SPLIT_RATIOS,
        "classes": list(CLASSES),
        "counts": {**copied_counts, "total": sum(copied_counts.values())},
        "files": splits_files,
    }
    manifest_path = OUTPUT_DATASET / "split_manifest.json"
    write_json(manifest_path, manifest)
    print(f"Wrote: {manifest_path}")

    print("\nDone. Next step:")
    print("  uv run python scripts/run_finetune_train.py")


if __name__ == "__main__":
    main()
