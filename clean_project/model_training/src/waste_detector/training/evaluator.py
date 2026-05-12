"""
Evaluation metric extraction for trained YOLO checkpoints.

Runs Ultralytics validation on one or more dataset splits and collects
precision, recall, F1, mAP50, mAP75, and mAP50-95 per split and per class.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from ultralytics import YOLO

from waste_detector.utils.io import write_json


# ---------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------

def _safe_float(value: Any) -> Optional[float]:
    """Convert a value to float, returning None if conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_f1(box: Any) -> Optional[float]:
    """
    Compute mean F1 from per-class precision and recall arrays.
    Returns None if the arrays are not available or are empty.
    """
    try:
        p = np.asarray(box.p, dtype=float)   # Per-class precision
        r = np.asarray(box.r, dtype=float)   # Per-class recall
        denom = p + r
        # Avoid division by zero for classes with no predictions.
        f1_per_class = np.where(denom > 0, 2 * p * r / denom, 0.0)
        return float(f1_per_class.mean())
    except Exception:
        return None


def _extract_metrics(val_results: Any, split: str) -> Dict[str, Any]:
    """
    Pull standard detection metrics out of an Ultralytics validation result object.

    Returns a nested dict with the split name, aggregate metrics, and per-class
    precision/recall/F1 values.
    """
    box = val_results.box

    # Collect aggregate metrics.
    metrics = {
        "precision": _safe_float(box.mp),
        "recall": _safe_float(box.mr),
        "f1": _mean_f1(box),
        "map50": _safe_float(box.map50),
        "map75": _safe_float(getattr(box, "map75", None)),
        "map50_95": _safe_float(box.map),
    }

    # Collect per-class precision and recall if class names are available.
    per_class: List[Dict[str, Any]] = []
    class_names = getattr(val_results, "names", {})
    try:
        for idx, name in class_names.items():
            per_class.append({
                "class_id": int(idx),
                "class_name": str(name),
                "precision": _safe_float(float(np.asarray(box.p)[idx])),
                "recall": _safe_float(float(np.asarray(box.r)[idx])),
            })
    except Exception:
        # Per-class data is best-effort; a failure here should not break evaluation.
        pass

    return {
        "split": split,
        "num_classes": int(box.nc) if hasattr(box, "nc") else None,
        "metrics": metrics,
        "per_class": per_class,
        "ultralytics_save_dir": str(val_results.save_dir)
            if hasattr(val_results, "save_dir") else None,
    }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

def run_split_evaluation(
    model: YOLO,
    data_yaml: Path,
    split: str,
    imgsz: int,
    batch: int,
    device: str,
    project: str,
    run_name: str,
    *,
    plots: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate 'model' on one dataset split (e.g. 'val' or 'test').

    Returns a metrics dict as produced by '_extract_metrics'.
    """
    val_results = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=run_name,
        plots=plots,
        save_json=True,
    )
    return _extract_metrics(val_results, split)


def evaluate_checkpoint(
    weights: Path,
    data_yaml: Path,
    *,
    splits: List[str] = ("val", "test"),
    imgsz: int = 640,
    batch: int = 8,
    device: str = "0",
    eval_project: str = "runs/evaluate",
    run_name_prefix: str = "eval",
    plots: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a trained checkpoint on one or more dataset splits.

    Loads the model once and runs validation for each requested split.
    Returns a summary dict containing the weights path and per-split results.
    """
    # Load the checkpoint once to avoid reloading for each split.
    model = YOLO(str(weights))

    results_per_split: Dict[str, Any] = {}
    for split in splits:
        # Use a separate subfolder per split so Ultralytics does not overwrite outputs.
        run_name = f"{run_name_prefix}_{split}"
        results_per_split[split] = run_split_evaluation(
            model=model,
            data_yaml=data_yaml,
            split=split,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=eval_project,
            run_name=run_name,
            plots=plots,
        )

        # Print a short summary after each split.
        m = results_per_split[split]["metrics"]
        print(
            f"[{split}] "
            f"P={m['precision']:.4f}  "
            f"R={m['recall']:.4f}  "
            f"F1={m['f1']:.4f}  "
            f"mAP50={m['map50']:.4f}  "
            f"mAP50-95={m['map50_95']:.4f}"
        )

    return {
        "weights": str(weights),
        "data_yaml": str(data_yaml),
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "results": results_per_split,
    }


def save_evaluation_report(summary: Dict[str, Any], output_path: Path) -> None:
    """Write the evaluation summary dictionary to a JSON file."""
    write_json(output_path, summary)
    print(f"Evaluation report saved to: {output_path}")
