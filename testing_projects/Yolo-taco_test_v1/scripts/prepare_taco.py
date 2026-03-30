#!/usr/bin/env python3
"""Convert raw TACO data into Ultralytics YOLO format."""

from __future__ import annotations

import argparse
from pathlib import Path

from trash_detector.data.taco_dataset import TacoDatasetManager


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset preparation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-name", default="taco_bottle_focus")
    parser.add_argument(
        "--label-map",
        type=Path,
        default=None,
        help="Optional label-map JSON. If omitted, all TACO classes are used.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-box-pixels", type=int, default=8)
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into the prepared dataset instead of creating symlinks.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    manager = TacoDatasetManager(args.project_root)
    manager.write_category_summary()

    label_map_path = args.label_map
    if label_map_path is not None and not label_map_path.is_absolute():
        label_map_path = args.project_root / label_map_path

    dataset_root = manager.prepare_yolo_dataset(
        output_name=args.output_name,
        label_map_path=label_map_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        min_box_pixels=args.min_box_pixels,
        copy_images=args.copy_images,
    )
    print(f"Prepared dataset: {dataset_root}")
    print(f"Data YAML:         {dataset_root / 'dataset.yaml'}")
