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

def main(config_path: Path = CONFIG_PATH) -> None:
    # --- Load and parse the YAML config ---
    # 'config_path' defaults to the main training config, but thin wrapper
    # scripts (e.g. 'run_finetune_train.py') can pass a different YAML so the
    # fine-tune workflow can reuse this runner without editing it.
    cfg = read_yaml(config_path)

    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    # New optional sections (each falls back to dataclass defaults if absent).
    opt_cfg = cfg.get("optimizer", {})
    aug_cfg = cfg.get("augmentation", {})
    log_cfg = cfg.get("logging", {})
    eval_cfg = cfg.get("evaluation", {})
    export_cfg = cfg.get("export", {})

    # --- Resolve the dataset path (supports relative paths from project root) ---
    dataset_path = Path(dataset_cfg["path"])
    if not dataset_path.is_absolute():
        dataset_path = (_PROJECT_ROOT / dataset_path).resolve()

    print(f"Loading dataset from: {dataset_path}")
    spec = load_dataset_spec(dataset_path)

    # --- Sanity-check the expected classes against the dataset ---
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

    # Resolve the 'resume' option: accept either false/empty (fresh run) or a path string.
    resume_val = train_cfg.get("resume", False)
    if isinstance(resume_val, str) and resume_val:
        resume_path = Path(resume_val)
        if not resume_path.is_absolute():
            resume_path = (_PROJECT_ROOT / resume_path).resolve()
        resume_str = str(resume_path)
    else:
        resume_str = ""

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
        resume=resume_str,
        # Optimizer.
        optimizer=opt_cfg.get("name", "auto"),
        cos_lr=opt_cfg.get("cos_lr", True),
        # Augmentation.
        multi_scale=aug_cfg.get("multi_scale", False),
        close_mosaic=aug_cfg.get("close_mosaic", 20),
        degrees=float(aug_cfg.get("degrees", 5.0)),
        translate=float(aug_cfg.get("translate", 0.05)),
        scale=float(aug_cfg.get("scale", 0.2)),
        fliplr=float(aug_cfg.get("fliplr", 0.5)),
        mosaic=float(aug_cfg.get("mosaic", 0.5)),
        mixup=float(aug_cfg.get("mixup", 0.0)),
        copy_paste=float(aug_cfg.get("copy_paste", 0.0)),
        hsv_h=float(aug_cfg.get("hsv_h", 0.015)),
        hsv_s=float(aug_cfg.get("hsv_s", 0.7)),
        hsv_v=float(aug_cfg.get("hsv_v", 0.4)),
        # Logging.
        plots=log_cfg.get("plots", True),
        val=log_cfg.get("val", True),
        verbose=log_cfg.get("verbose", True),
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
        # When enabled, the evaluator also reports precision / recall / F1
        # from the confusion matrix with the background row and column dropped.
        exclude_background = bool(eval_cfg.get("exclude_background", False))

        # Use the real training output folder (Ultralytics may have appended a
        # suffix like '-2' when the configured run_name was already taken), so
        # the eval plots land inside the same folder as the rest of this run.
        actual_run_dir = Path(summary["save_dir"])
        eval_project_dir = actual_run_dir / "eval"

        print(f"\nRunning post-training evaluation on splits: {splits}")
        print(f"Post-eval outputs will be saved under: {eval_project_dir}")
        eval_summary = evaluate_checkpoint(
            weights=best_weights,
            data_yaml=spec.training_yaml,
            splits=splits,
            imgsz=train_config.imgsz,
            batch=eval_batch,
            device=train_config.device,
            eval_project=str(eval_project_dir),
            exclude_background=exclude_background,
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
