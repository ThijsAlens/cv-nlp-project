"""Training configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TrainConfig:
    """Configuration for one YOLO fine-tuning run."""

    data_yaml: Path
    model_weights: str = "yolov8n.pt"
    output_dir: Path = Path("runs")
    run_name: str = "yolov8n_taco_bottle_focus"
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
