#!/usr/bin/env python3
"""Train on a prepared dataset folder (Ultralytics layout). Edit the CONFIG block below (no CLI)."""

from __future__ import annotations

import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Configuration: adjust these values only.
# -----------------------------------------------------------------------------
CONFIG = {
    # Project root (folder that contains `data/`, `runs/`, etc.).
    "project_root": Path(__file__).resolve().parents[1],
    # Folder that contains data.yaml (or dataset.yaml) plus train/valid/test (or images/train …).
    "dataset_dir": Path("data/Totaal_dataset"),
    # Ultralytics pretrained checkpoint (downloads automatically if missing).
    "model_weights": "yolo11s.pt",
    "run_name": "yolo11s_garbage_5c",
    "epochs": 60,
    "imgsz": 640,
    "batch": 32,
    "device": "0",
    "workers": 4,
    "freeze": 10,
    "patience": 20,
    "cache": False,
    "amp": True,
    "export_onnx": True,
    # After training: run Ultralytics val on chosen splits (saves confusion matrix, PR/F1 curves, predictions JSON).
    "run_post_train_eval": True,
    "eval_splits": ["val", "test"],
    "eval_batch": 8,
    # Optional: extra model.train() kwargs to override augmentations or learning rate.
    "extra_train_args": {},
}
# -----------------------------------------------------------------------------

# Allow `python scripts/run_train.py` without installing the package globally.
_PROJECT_ROOT = CONFIG["project_root"]
_SRC = _PROJECT_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from trash_detector.training.config import TrainConfig
from trash_detector.training.eval_report import evaluate_checkpoint, save_evaluation_report
from trash_detector.training.trainer import YoloTrainer
from trash_detector.training.yolo_data import load_dataset_spec
from trash_detector.utils.io import read_json, write_json


def main() -> None:
    root = CONFIG["project_root"].resolve()
    dataset_dir = CONFIG["dataset_dir"]
    dataset_dir = dataset_dir if dataset_dir.is_absolute() else root / dataset_dir

    spec = load_dataset_spec(dataset_dir)
    print(f"Dataset root: {spec.dataset_root}")
    print(f"Classes (nc={spec.nc}): {spec.names}")
    print(f"Training YAML: {spec.training_yaml}")

    train_project = str(root / "runs" / "train")
    cfg = TrainConfig(
        data_yaml=spec.training_yaml,
        model_weights=CONFIG["model_weights"],
        run_name=CONFIG["run_name"],
        epochs=CONFIG["epochs"],
        imgsz=CONFIG["imgsz"],
        batch=CONFIG["batch"],
        device=CONFIG["device"],
        workers=CONFIG["workers"],
        freeze=CONFIG["freeze"],
        patience=CONFIG["patience"],
        cache=CONFIG["cache"],
        amp=CONFIG["amp"],
        project=train_project,
        extra_train_args=dict(CONFIG["extra_train_args"]),
    )
    trainer = YoloTrainer(cfg)
    summary = trainer.train()
    save_dir = Path(summary["save_dir"])
    print(f"Training outputs saved to: {summary['save_dir']}")

    if CONFIG.get("run_post_train_eval"):
        eval_project = save_dir / "post_eval"
        eval_summary = evaluate_checkpoint(
            weights=Path(summary["best_weights"]),
            data_yaml=spec.training_yaml,
            splits=tuple(CONFIG["eval_splits"]),
            imgsz=CONFIG["imgsz"],
            batch=int(CONFIG["eval_batch"]),
            device=CONFIG["device"],
            eval_project=eval_project,
            run_name_prefix="split",
        )
        metrics_path = save_dir / "evaluation_metrics.json"
        save_evaluation_report(eval_summary, metrics_path)
        run_summary_path = save_dir / "run_summary.json"
        merged = read_json(run_summary_path)
        merged["evaluation_metrics_json"] = str(metrics_path.resolve())
        merged["post_eval_ultralytics_dirs"] = {
            split: eval_summary["results"][split].get("ultralytics_save_dir")
            for split in eval_summary["results"]
        }
        write_json(run_summary_path, merged)
        print(f"Evaluation metrics saved to: {metrics_path}")
        for split, block in eval_summary["results"].items():
            m = block["metrics"]
            print(f"  [{split}] mAP50-95={m['map50_95']}  P={m['precision']}  R={m['recall']}  F1={m['f1']}")

    if CONFIG["export_onnx"]:
        onnx_path = trainer.export_onnx(Path(summary["best_weights"]), imgsz=CONFIG["imgsz"])
        print(f"ONNX export: {onnx_path}")


if __name__ == "__main__":
    main()
