"""
Class-merging runner for the waste detector datasets.

Reads an existing YOLO dataset on disk and writes a sibling dataset folder
where some classes have been merged together (and optionally renamed).

Image files are not duplicated. Each image in the new dataset is created
as a hardlink to the corresponding source image (via 'os.link'), so the
new dataset adds essentially no bytes on disk. Hardlinks survive
'Path.resolve()' (unlike symlinks/junctions), which Ultralytics calls on
the dataset split paths during validation. Only the YOLO '.txt' label
files are regenerated with remapped class ids, and a new 'data.yaml' is
written at the root of the new dataset.

Note: the source and target paths must be on the same filesystem / drive
letter, because hardlinks cannot cross volumes.

Edit the CONFIG block below to change the source/target paths or the
merge specification, then run:

  uv run python scripts/run_merge_classes.py
"""

import sys
from pathlib import Path

# Add the project's 'src' folder to the import path so the package can be
# imported when this script is run directly without an editable install.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from waste_detector.data.class_merger import build_merged_dataset


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

# Path to the existing dataset (absolute or relative to the project root).
SOURCE_DATASET = "./data/FINALE_TESTSET"

# Path where the merged dataset will be written.
# The script refuses to overwrite an existing folder, so delete it manually
# if you want to rebuild from scratch.
TARGET_DATASET = "./data/FINALE_TESTSET_merged"

# Merge specification.
# Each entry is either:
#   - a string: the named class is kept unchanged in the final class list, OR
#   - a list:   first item is the target class name (kept in the final list);
#               every following item is an original class name that is merged
#               into the target. If the target name itself also exists in
#               the original dataset, that original class is also merged in.
#
# Rules:
#   - Every original class must appear somewhere in this list (either as a
#     string entry or inside a merge group). This prevents accidental drops.
#   - Each original class may appear in at most one entry.
#   - The order of this list defines the new YOLO class ids (top -> id 0, etc.).
#
# Example variations:
#   ["Glass", "Metal", "Plastic", ["Paper", "Cardboard"]]
#       -> 4 classes; Cardboard merged into the existing Paper class.
#
#   ["Glass", "Metal", "Plastic", ["PaperCardboard", "Paper", "Cardboard"]]
#       -> 4 classes; both Paper and Cardboard merged into a new
#          'PaperCardboard' class.
MERGED_CLASSES = [
    "Glass",
    "Metal",
    "Plastic",
    # Merge 'Cardboard' into the existing 'Paper' class.
    ["Paper", "Cardboard"],
]


# ---------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------

def main() -> None:
    # Resolve any relative paths against the project root so the script can
    # be invoked from any working directory.
    source_path = Path(SOURCE_DATASET)
    if not source_path.is_absolute():
        source_path = (_PROJECT_ROOT / source_path).resolve()

    target_path = Path(TARGET_DATASET)
    if not target_path.is_absolute():
        target_path = (_PROJECT_ROOT / target_path).resolve()

    # Print a short summary so the user can confirm the configuration at a glance.
    print(f"Source dataset: {source_path}")
    print(f"Target dataset: {target_path}")
    print(f"Merge spec:     {MERGED_CLASSES}")
    print()

    # Run the actual build. The function returns a small summary structure
    # that we surface to the console for visibility.
    result = build_merged_dataset(
        source_root=source_path,
        target_root=target_path,
        merged_classes=MERGED_CLASSES,
    )

    print()
    print(f"Done. New dataset YAML: {result.data_yaml}")
    print(f"Final class list ({len(result.new_names)} classes): {result.new_names}")
    print(f"Class id remapping (old -> new): {result.id_map}")
    print(f"Splits processed: {result.splits_processed}")


if __name__ == "__main__":
    main()
