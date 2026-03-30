#!/usr/bin/env python3
"""Run inference with a trained YOLO checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trash_detector.inference.predictor import TrashPredictor


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path, help="Path to a trained YOLO checkpoint.")
    parser.add_argument("source", help="Image, directory, video, or webcam source accepted by Ultralytics.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predictor = TrashPredictor(args.weights)
    predictions = predictor.predict(
        source=args.source,
        conf=args.conf,
        save=args.save,
        imgsz=args.imgsz,
        device=args.device,
    )
    print(json.dumps(predictions, indent=2))
