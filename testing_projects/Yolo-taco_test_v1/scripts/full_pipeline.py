#!/usr/bin/env python3
"""Convenience entry point to run download, preparation, and training in sequence."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from trash_detector.data.taco_dataset import TacoDatasetManager
from trash_detector.training.config import TrainConfig
from trash_detector.training.trainer import YoloTrainer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the full end-to-end pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-name", default="taco_bottle_focus")
    parser.add_argument("--label-map", type=Path, default=Path("configs/label_maps/bottle_focus.json"))
    parser.add_argument("--max-images", type=int, default=None, help="Use a smaller subset for a smoke test.")
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--freeze", type=int, default=10)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--copy-images", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    manager = TacoDatasetManager(args.project_root)
    manager.download_annotations()
    manager.download_images(max_images=args.max_images, num_workers=args.download_workers)
    manager.write_category_summary()

    label_map_path = args.label_map if args.label_map.is_absolute() else args.project_root / args.label_map
    dataset_root = manager.prepare_yolo_dataset(
        output_name=args.output_name,
        label_map_path=label_map_path,
        copy_images=args.copy_images,
    )

    # Force a checkpoint warm-up so weight download happens before training starts.
    YOLO(args.model)

    trainer = YoloTrainer(
        TrainConfig(
            data_yaml=dataset_root / "dataset.yaml",
            model_weights=args.model,
            run_name=f"{Path(args.model).stem}_{args.output_name}",
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            freeze=args.freeze,
            cache=args.cache,
            project=str(args.project_root / "runs" / "train"),
        )
    )
    summary = trainer.train()
    print(summary)
