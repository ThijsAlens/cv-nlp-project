"""Training configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrainConfig:
    """Configuration for one YOLO fine-tuning run."""

    data_yaml: Path
    model_weights: str = "yolo11s.pt"
    output_dir: Path = Path("runs")
    run_name: str = "yolo11s_all_classes"
    epochs: int = 60
    imgsz: int = 640
    batch: int = 16
    device: str = "0"
    workers: int = 4
    freeze: int = 10
    patience: int = 20
    cache: bool = False
    amp: bool = True
    project: str = "runs/train"
    # When True, pass Ultralytics 'cls_pw' so rare classes get higher weight in the classification loss.
    balanced_training: bool = False
    # Strength of inverse-frequency weighting: 0.25 is a common starting point; use 1.0 for full weighting.
    # Ignored when 'balanced_training' is False unless you set 'cls_pw' via 'extra_train_args'.
    balanced_cls_pw: float = 0.25
    #: Extra keyword arguments merged into ``YOLO.train()`` (override defaults from :class:`YoloTrainer`).
    extra_train_args: dict[str, Any] = field(default_factory=dict)
