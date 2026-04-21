"""Wrapper around Ultralytics YOLO training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import ultralytics
from ultralytics import YOLO

from trash_detector.training.config import TrainConfig
from trash_detector.utils.io import ensure_dir, write_json


def _ultralytics_supports_cls_pw() -> bool:
    """True if this Ultralytics build accepts the 'cls_pw' train argument (added in 8.4.40)."""
    default_yaml = Path(ultralytics.__file__).resolve().parent / "cfg" / "default.yaml"
    if not default_yaml.is_file():
        return False
    # 'cls_pw' must appear as a YAML key in the default config (added in Ultralytics 8.4.40).
    text = default_yaml.read_text(encoding="utf-8")
    return "cls_pw:" in text


class YoloTrainer:
    """Train, validate, and export a YOLO detector."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config

    def train(self) -> Dict[str, Any]:
        """Launch fine-tuning using the Ultralytics Python API.

        The model checkpoint will be auto-downloaded by Ultralytics if it is not present locally.
        """
        model = YOLO(self.config.model_weights)
        train_kwargs: Dict[str, Any] = {
            "data": str(self.config.data_yaml),
            "epochs": self.config.epochs,
            "imgsz": self.config.imgsz,
            "batch": self.config.batch,
            "device": self.config.device,
            "workers": self.config.workers,
            "freeze": self.config.freeze,
            "patience": self.config.patience,
            "cache": self.config.cache,
            "amp": self.config.amp,
            "project": self.config.project,
            "name": self.config.run_name,
            "pretrained": True,
            "optimizer": "auto",
            "cos_lr": True,
            "multi_scale": 0.25,
            "close_mosaic": 20,
            "degrees": 5.0,
            "translate": 0.05,
            "scale": 0.2,
            "fliplr": 0.5,
            "mosaic": 0.5,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "plots": True,
            "val": True,
            "verbose": True,
        }
        # Ultralytics uses 'cls_pw' from label counts: higher values upweight rare classes on cls loss.
        if self.config.balanced_training:
            train_kwargs["cls_pw"] = self.config.balanced_cls_pw
        # User-supplied kwargs win, including an explicit 'cls_pw' to override 'balanced_cls_pw'.
        train_kwargs.update(self.config.extra_train_args)
        # Older Ultralytics rejects unknown keys; 'cls_pw' exists only from 8.4.40 onward.
        if "cls_pw" in train_kwargs and not _ultralytics_supports_cls_pw():
            from importlib.metadata import version as pkg_version

            ver = pkg_version("ultralytics")
            raise RuntimeError(
                "balanced_training (and the 'cls_pw' argument) need Ultralytics >= 8.4.40. "
                f"Your environment has ultralytics {ver}. Upgrade with: "
                "uv pip install 'ultralytics>=8.4.40'  or  pip install -U 'ultralytics>=8.4.40'"
            )
        results = model.train(**train_kwargs)

        # Ultralytics stores run artifacts inside the run directory. This summary file makes
        # it easier to inspect the final location from scripts or CI jobs.
        save_dir = Path(results.save_dir)
        summary = {
            "save_dir": str(save_dir.resolve()),
            "best_weights": str((save_dir / "weights" / "best.pt").resolve()),
            "last_weights": str((save_dir / "weights" / "last.pt").resolve()),
            "results_csv": str((save_dir / "results.csv").resolve()),
            # Record what was passed to 'train()' after merges (same as effective cls_pw for this run).
            "balanced_training": self.config.balanced_training,
            "balanced_cls_pw": self.config.balanced_cls_pw,
            "cls_pw": train_kwargs.get("cls_pw", 0.0),
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
