"""
Training runner for the waste material detector.

Reads 'config/train_config.yaml', validates the dataset, and launches
a YOLO training run. Optionally evaluates the best checkpoint and exports
it to ONNX after training finishes.

Usage:
  uv run python scripts/run_train.py
"""

import sys
from pathlib import Path

# Add the 'src' folder to the import path so 'waste_detector' can be found
# when this script is run directly (without installing the package first).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from waste_detector.training.config import TrainConfig
from waste_detector.training.dataset import load_dataset_spec
from waste_detector.training.evaluator import evaluate_checkpoint, save_evaluation_report
from waste_detector.training.trainer import YoloTrainer
from waste_detector.utils.io import read_yaml, write_json

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

# Path to the training configuration YAML. Edit that file to change settings.
CONFIG_PATH = _PROJECT_ROOT / "config" / "train_config.yaml"

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main() -> None:
    # --- Load and parse the YAML config ---
    cfg = read_yaml(CONFIG_PATH)

    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    eval_cfg = cfg.get("evaluation", {})
    export_cfg = cfg.get("export", {})

    # --- Resolve the dataset path (supports relative paths from project root) ---
    dataset_path = Path(dataset_cfg["path"])
    if not dataset_path.is_absolute():
        dataset_path = (_PROJECT_ROOT / dataset_path).resolve()

    print(f"Loading dataset from: {dataset_path}")
    spec = load_dataset_spec(dataset_path)

    # --- Validate expected classes against the dataset (informational check) ---
    expected_classes = dataset_cfg.get("classes", [])
    if expected_classes and set(expected_classes) != set(spec.names):
        print(
            f"Warning: config 'classes' {expected_classes} do not match "
            f"dataset classes {spec.names}. Training will use the dataset classes."
        )

    print(f"Dataset classes ({spec.nc}): {spec.names}")
    print(f"Training YAML: {spec.training_yaml}")

    # --- Build the TrainConfig object from the parsed YAML ---
    output_dir = Path(model_cfg.get("output_dir", "./runs/train"))
    if not output_dir.is_absolute():
        output_dir = (_PROJECT_ROOT / output_dir).resolve()

    train_config = TrainConfig(
        data_yaml=spec.training_yaml,
        model_weights=model_cfg["pretrained_weights"],
        run_name=model_cfg["run_name"],
        epochs=train_cfg.get("epochs", 60),
        imgsz=train_cfg.get("imgsz", 640),
        batch=train_cfg.get("batch", 32),
        device=str(train_cfg.get("device", "0")),
        workers=train_cfg.get("workers", 4),
        freeze=train_cfg.get("freeze", 10),
        patience=train_cfg.get("patience", 20),
        amp=train_cfg.get("amp", True),
        cache=train_cfg.get("cache", False),
        project=str(output_dir),
        balanced_training=train_cfg.get("balanced_training", False),
        balanced_cls_pw=float(train_cfg.get("balanced_cls_pw", 0.25)),
    )

    # --- Run training ---
    print(f"\nStarting training run: '{train_config.run_name}'")
    trainer = YoloTrainer(train_config)
    summary = trainer.train()

    print(f"\nTraining complete. Best weights: {summary['best_weights']}")

    # --- Optional post-training evaluation ---
    if eval_cfg.get("run_after_training", False):
        best_weights = Path(summary["best_weights"])
        splits = eval_cfg.get("splits", ["val"])
        eval_batch = eval_cfg.get("batch", 8)

        print(f"\nRunning post-training evaluation on splits: {splits}")
        eval_summary = evaluate_checkpoint(
            weights=best_weights,
            data_yaml=spec.training_yaml,
            splits=splits,
            imgsz=train_config.imgsz,
            batch=eval_batch,
            device=train_config.device,
            eval_project=str(output_dir / train_config.run_name / "eval"),
        )

        # Save the evaluation results next to the training outputs.
        eval_out = Path(summary["save_dir"]) / "evaluation_metrics.json"
        save_evaluation_report(eval_summary, eval_out)

    # --- Optional ONNX export ---
    if export_cfg.get("onnx", False):
        best_weights = Path(summary["best_weights"])
        print(f"\nExporting to ONNX: {best_weights}")
        onnx_path = trainer.export_onnx(best_weights)
        print(f"ONNX model saved to: {onnx_path}")


if __name__ == "__main__":
    main()
