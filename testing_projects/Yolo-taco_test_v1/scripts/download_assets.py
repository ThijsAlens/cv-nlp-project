#!/usr/bin/env python3
"""Download TACO annotations, TACO images, and optionally warm up YOLO weights."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from trash_detector.data.taco_dataset import TacoDatasetManager


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the asset download step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--max-images", type=int, default=None, help="Download only the first N images for a smoke test.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel download workers.")
    parser.add_argument("--weights", default="yolov8n.pt", help="YOLO checkpoint name or path to warm up.")
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="Skip the YOLO checkpoint warm-up download.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    manager = TacoDatasetManager(args.project_root)
    annotations_path = manager.download_annotations()
    images_dir = manager.download_images(max_images=args.max_images, num_workers=args.workers)

    # Loading the model once is enough for Ultralytics to fetch and cache the weights.
    if not args.skip_weights:
        YOLO(args.weights)

    print(f"Annotations: {annotations_path}")
    print(f"Images:      {images_dir}")
    if not args.skip_weights:
        print(f"Weights:     {args.weights}")
