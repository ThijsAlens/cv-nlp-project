"""Run Ultralytics validation splits and collect metrics / artifact paths for reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ultralytics import YOLO


def safe_float(value: Any) -> float | None:
    """Convert values to float when possible, otherwise return None."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_f1(box: Any) -> float | None:
    f1 = getattr(box, "f1", None)
    if f1 is None or len(f1) == 0:
        return None
    arr = np.asarray(f1, dtype=float)
    return safe_float(float(np.nanmean(arr)))


def extract_metrics(metrics: Any, split: str) -> dict[str, Any]:
    """Extract a compact summary from an Ultralytics detection metrics object."""
    box = metrics.box
    class_names: dict[int, str] = {}
    if hasattr(metrics, "names") and metrics.names is not None:
        class_names = {int(k): str(v) for k, v in dict(metrics.names).items()}

    per_class: list[dict[str, Any]] = []
    for i in range(len(box.ap_class_index)):
        c = int(box.ap_class_index[i])
        p_i, r_i, ap50_i, ap_i = box.class_result(i)
        f1_i = safe_float(box.f1[i]) if i < len(box.f1) else None
        name = class_names.get(c, str(c))
        per_class.append(
            {
                "class_id": c,
                "class_name": name,
                "precision": safe_float(p_i),
                "recall": safe_float(r_i),
                "f1": f1_i,
                "map50": safe_float(ap50_i),
                "map50_95": safe_float(ap_i),
            }
        )

    mp = safe_float(box.mp)
    mr = safe_float(box.mr)
    f1_from_pr: float | None = None
    if mp is not None and mr is not None and (mp + mr) > 0:
        f1_from_pr = safe_float(2.0 * mp * mr / (mp + mr))

    save_dir = getattr(metrics, "save_dir", None)
    save_dir_str = str(Path(save_dir).resolve()) if save_dir is not None else None

    return {
        "split": split,
        "num_classes": len(class_names),
        "metrics": {
            "precision": mp,
            "recall": mr,
            "f1": _mean_f1(box),
            "f1_from_precision_recall": f1_from_pr,
            "map50": safe_float(box.map50),
            "map75": safe_float(box.map75),
            "map50_95": safe_float(box.map),
        },
        "per_class": per_class,
        "ultralytics_save_dir": save_dir_str,
        "note": (
            "Object-detection metrics (mAP, P, R, F1 at max-F1 confidence). "
            "There is no single 'accuracy' like in image classification; see per-class boxes and mAP."
        ),
    }


def run_split_evaluation(
    model: YOLO,
    data_yaml: Path,
    split: str,
    imgsz: int,
    batch: int,
    device: str,
    project: Path,
    run_name: str,
    *,
    plots: bool = True,
    save_json: bool = True,
) -> dict[str, Any]:
    """Run ``model.val`` on one split; plots include confusion matrix and PR/F1 curves when *plots* is True."""
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        name=run_name,
        plots=plots,
        save_json=save_json,
        verbose=True,
    )
    return extract_metrics(metrics, split)


def evaluate_checkpoint(
    weights: Path,
    data_yaml: Path,
    *,
    splits: Sequence[str] = ("val", "test"),
    imgsz: int = 640,
    batch: int = 8,
    device: str = "0",
    eval_project: Path,
    run_name_prefix: str = "eval",
    plots: bool = True,
    save_json: bool = True,
) -> dict[str, Any]:
    """Evaluate *weights* on each split; Ultralytics writes images and JSON under *eval_project* / *run_name*."""
    weights = weights.resolve()
    data_yaml = data_yaml.resolve()
    eval_project = eval_project.resolve()
    eval_project.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    summary: dict[str, Any] = {
        "weights": str(weights),
        "data_yaml": str(data_yaml),
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "results": {},
    }

    for split in splits:
        run_name = f"{run_name_prefix}_{split}"
        summary["results"][split] = run_split_evaluation(
            model=model,
            data_yaml=data_yaml,
            split=split,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=eval_project,
            run_name=run_name,
            plots=plots,
            save_json=save_json,
        )
    return summary


def save_evaluation_report(summary: dict[str, Any], output_path: Path) -> None:
    """Write *summary* as JSON (creates parent directories)."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
