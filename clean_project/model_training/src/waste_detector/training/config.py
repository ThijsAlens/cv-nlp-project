"""
TrainConfig: a dataclass that holds all parameters for one YOLO training run.

Instances are constructed by the runner script from the train_config.yaml file.
The trainer reads fields from this object and passes them to Ultralytics.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass(slots=True)
class TrainConfig:
    """All settings needed to launch a single YOLO training run."""

    # Path to the dataset YAML that Ultralytics will read.
    data_yaml: Path

    # Pretrained weights to start from (Ultralytics model name or local path).
    model_weights: str = "yolo11s.pt"

    # Subfolder name for the output inside 'project'.
    run_name: str = "yolo11s_garbage_v1"

    # Number of training epochs.
    epochs: int = 60

    # Input image size (pixels, square).
    imgsz: int = 640

    # Training batch size.
    batch: int = 32

    # Device string: '0' for first GPU, 'cpu', or '0,1' for multi-GPU.
    device: str = "0"

    # Number of data-loading worker processes.
    workers: int = 4

    # How many backbone layers to freeze at the start of training.
    freeze: int = 10

    # Early-stopping patience in epochs (0 to disable).
    patience: int = 20

    # Cache images in memory ('ram') or on disk ('disk'). False to disable.
    cache: bool = False

    # Use automatic mixed precision (faster on modern GPUs).
    amp: bool = True

    # Top-level output directory; runs are saved to '<project>/<run_name>'.
    project: str = "runs/train"

    # Enable inverse-frequency class weighting for imbalanced datasets.
    balanced_training: bool = False

    # Strength of class balancing (0.0 = no effect, 1.0 = full inverse weighting).
    balanced_cls_pw: float = 0.25

    # Path to a 'last.pt' to resume training from. Empty string = start a fresh run.
    # When set, the model is loaded from this checkpoint and 'resume=True' is passed
    # to Ultralytics, which restores the saved epoch and all original training args.
    resume: str = ""

    # --- Optimizer ---
    # Optimizer name passed to Ultralytics ('auto', 'SGD', 'Adam', 'AdamW', ...).
    optimizer: str = "auto"
    # Use cosine learning-rate schedule.
    cos_lr: bool = True

    # --- Augmentation ---
    # Vary image size per batch.
    multi_scale: bool = False
    # Disable mosaic for the last N epochs.
    close_mosaic: int = 20
    # Max random rotation in degrees.
    degrees: float = 5.0
    # Max random translation as a fraction of imgsz.
    translate: float = 0.05
    # Max random scale variation.
    scale: float = 0.2
    # Probability of horizontal flip.
    fliplr: float = 0.5
    # Probability of mosaic augmentation.
    mosaic: float = 0.5
    # Probability of mixup augmentation.
    mixup: float = 0.0
    # Probability of copy-paste augmentation.
    copy_paste: float = 0.0
    # HSV hue jitter.
    hsv_h: float = 0.015
    # HSV saturation jitter.
    hsv_s: float = 0.7
    # HSV value (brightness) jitter.
    hsv_v: float = 0.4

    # --- Logging ---
    # Generate training plots.
    plots: bool = True
    # Run validation each epoch.
    val: bool = True
    # Verbose training output.
    verbose: bool = True

    # Extra keyword arguments passed directly to model.train(); these override all above.
    extra_train_args: Dict[str, Any] = field(default_factory=dict)
