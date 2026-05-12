"""
YoloTrainer: wraps Ultralytics YOLO training with the project's TrainConfig.

Handles class balancing, augmentation defaults, and post-training summary output.
The 'train()' method is the single call used by 'run_train.py'.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO

from waste_detector.training.config import TrainConfig
from waste_detector.utils.io import ensure_dir, write_json


# ---------------------------------------------------------------
# Version check helper
# ---------------------------------------------------------------

def _ultralytics_supports_cls_pw() -> bool:
    """
    Return True if the installed Ultralytics version supports the 'cls_pw' argument.
    Support was added in version 8.4.40.
    """
    try:
        import ultralytics
        # Parse the version string into a tuple of ints for comparison.
        parts = [int(x) for x in ultralytics.__version__.split(".")[:3]]
        return tuple(parts) >= (8, 4, 40)
    except Exception:
        # If parsing fails for any reason, assume it is not supported.
        return False


# ---------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------

class YoloTrainer:
    """Orchestrates a YOLO training run from a TrainConfig instance."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config

    # -----------------------------------------------------------
    # Main training method
    # -----------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """
        Run YOLO training and return a summary dictionary.

        Steps:
          1. Load the pretrained YOLO model.
          2. Assemble training keyword arguments from the config.
          3. Optionally add class-balancing weights.
          4. Apply any user overrides from 'extra_train_args'.
          5. Launch training.
          6. Write a run_summary.json to the output directory.
        """
        cfg = self.config
        model = YOLO(cfg.model_weights)

        # --- Assemble core training arguments ---
        train_kwargs: Dict[str, Any] = {
            "data": str(cfg.data_yaml),
            "epochs": cfg.epochs,
            "imgsz": cfg.imgsz,
            "batch": cfg.batch,
            "device": cfg.device,
            "workers": cfg.workers,
            "freeze": cfg.freeze,
            "patience": cfg.patience,
            "cache": cfg.cache,
            "amp": cfg.amp,
            "project": cfg.project,
            "name": cfg.run_name,
            # Optimizer settings.
            "optimizer": "auto",
            "cos_lr": True,
            # Augmentation settings (conservative defaults for materials dataset).
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
            # Output options.
            "plots": True,
            "val": True,
            "verbose": True,
        }

        # --- Add class-balancing weights if requested ---
        if cfg.balanced_training:
            if _ultralytics_supports_cls_pw():
                # 'cls_pw' scales the classification loss weight per class.
                train_kwargs["cls_pw"] = cfg.balanced_cls_pw
            else:
                print(
                    "Warning: 'balanced_training' is enabled but the installed "
                    "Ultralytics version does not support 'cls_pw'. "
                    "Upgrade to >=8.4.40 to use class balancing."
                )

        # --- Apply user overrides last so they take precedence ---
        train_kwargs.update(cfg.extra_train_args)

        # --- Launch training ---
        results = model.train(**train_kwargs)

        # --- Locate output artefacts ---
        save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else (
            Path(cfg.project) / cfg.run_name
        )
        best_weights = save_dir / "weights" / "best.pt"

        # --- Write a summary JSON for later reference ---
        summary = {
            "run_name": cfg.run_name,
            "model_weights": cfg.model_weights,
            "data_yaml": str(cfg.data_yaml),
            "epochs": cfg.epochs,
            "save_dir": str(save_dir),
            "best_weights": str(best_weights),
            "balanced_training": cfg.balanced_training,
        }
        write_json(save_dir / "run_summary.json", summary)

        return summary

    # -----------------------------------------------------------
    # Validation
    # -----------------------------------------------------------

    def validate(self, weights_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Run Ultralytics validation on the training data YAML.

        Uses 'weights_path' if provided, otherwise re-validates from the last run.
        Returns a dict with precision, recall, mAP50, and mAP50-95.
        """
        cfg = self.config

        # Load either the specified checkpoint or the base model weights.
        model_to_eval = YOLO(str(weights_path)) if weights_path else YOLO(cfg.model_weights)

        val_results = model_to_eval.val(
            data=str(cfg.data_yaml),
            imgsz=cfg.imgsz,
            device=cfg.device,
        )

        return {
            "map50": float(val_results.box.map50),
            "map50_95": float(val_results.box.map),
            "precision": float(val_results.box.mp),
            "recall": float(val_results.box.mr),
        }

    # -----------------------------------------------------------
    # ONNX export
    # -----------------------------------------------------------

    def export_onnx(self, weights_path: Path, imgsz: Optional[int] = None) -> Path:
        """
        Export a trained checkpoint to ONNX format.

        Returns the path to the generated .onnx file.
        """
        model = YOLO(str(weights_path))
        export_imgsz = imgsz or self.config.imgsz
        # Ultralytics returns the export path as a string.
        onnx_path = model.export(format="onnx", imgsz=export_imgsz)
        return Path(onnx_path)
