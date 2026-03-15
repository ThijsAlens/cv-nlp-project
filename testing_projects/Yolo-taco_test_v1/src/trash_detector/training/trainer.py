"""Wrapper around Ultralytics YOLO training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ultralytics import YOLO

from trash_detector.training.config import TrainConfig
from trash_detector.utils.io import ensure_dir, write_json


class YoloTrainer:
    """Train, validate, and export a YOLO detector."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config

    def train(self) -> Dict[str, Any]:
        """Launch fine-tuning using the Ultralytics Python API.

        The model checkpoint will be auto-downloaded by Ultralytics if it is not present locally.
        """
        model = YOLO(self.config.model_weights)
        results = model.train(
            data=str(self.config.data_yaml),
            epochs=self.config.epochs,
            imgsz=self.config.imgsz,
            batch=self.config.batch,
            device=self.config.device,
            workers=self.config.workers,
            freeze=self.config.freeze,
            patience=self.config.patience,
            cache=self.config.cache,
            amp=self.config.amp,
            project=self.config.project,
            name=self.config.run_name,
            pretrained=True,
            optimizer="auto",
            cos_lr=True,
            multi_scale=0.25,
            close_mosaic=20,
            degrees=5.0,
            translate=0.05,
            scale=0.2,
            fliplr=0.5,
            mosaic=0.5,
            mixup=0.0,
            copy_paste=0.0,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            plots=True,
            val=True,
            verbose=True,
        )

        # Ultralytics stores run artifacts inside the run directory. This summary file makes
        # it easier to inspect the final location from scripts or CI jobs.
        save_dir = Path(results.save_dir)
        summary = {
            "save_dir": str(save_dir.resolve()),
            "best_weights": str((save_dir / "weights" / "best.pt").resolve()),
            "last_weights": str((save_dir / "weights" / "last.pt").resolve()),
            "results_csv": str((save_dir / "results.csv").resolve()),
        }
        ensure_dir(save_dir)
        write_json(save_dir / "run_summary.json", summary)
        return summary

    def validate(self, weights_path: Path | None = None) -> Dict[str, Any]:
        """Run validation for a trained checkpoint and return key metrics."""
        model = YOLO(str(weights_path or self.config.model_weights))
        metrics = model.val(data=str(self.config.data_yaml), imgsz=self.config.imgsz, device=self.config.device)
        return {
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }

    def export_onnx(self, weights_path: Path, imgsz: int | None = None) -> Path:
        """Export a trained checkpoint to ONNX for later deployment."""
        model = YOLO(str(weights_path))
        exported_path = model.export(format="onnx", imgsz=imgsz or self.config.imgsz)
        return Path(exported_path)
