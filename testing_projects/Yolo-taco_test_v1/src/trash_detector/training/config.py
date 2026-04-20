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
    #: Extra keyword arguments merged into ``YOLO.train()`` (override defaults from :class:`YoloTrainer`).
    extra_train_args: dict[str, Any] = field(default_factory=dict)
