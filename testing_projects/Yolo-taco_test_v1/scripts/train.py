#!/usr/bin/env python3
"""Train a detector on a prepared dataset (CLI). Defaults target Totaal_dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from trash_detector.training.config import TrainConfig
from trash_detector.training.trainer import YoloTrainer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=Path("data/Totaal_dataset/dataset.ultralytics.yaml"),
        help="Path relative to the project root, unless absolute.",
    )
    parser.add_argument("--model", default="yolo11s.pt", help="Base checkpoint used for fine-tuning.")
    parser.add_argument("--run-name", default="yolo11s_total_cli")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="GPU id such as 0, or 'cpu'.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--freeze", type=int, default=10)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    parser.add_argument("--export-onnx", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_yaml = args.data_yaml if args.data_yaml.is_absolute() else args.project_root / args.data_yaml
    config = TrainConfig(
        data_yaml=data_yaml,
        model_weights=args.model,
        run_name=args.run_name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        freeze=args.freeze,
        patience=args.patience,
        cache=args.cache,
        amp=not args.no_amp,
        project=str(args.project_root / "runs" / "train"),
    )
    trainer = YoloTrainer(config)
    summary = trainer.train()
    print(f"Training outputs saved to: {summary['save_dir']}")

    if args.export_onnx:
        onnx_path = trainer.export_onnx(Path(summary["best_weights"]), imgsz=args.imgsz)
        print(f"ONNX export: {onnx_path}")
