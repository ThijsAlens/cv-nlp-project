#!/usr/bin/env python3
"""Evaluate a trained YOLO detection model on validation and test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "weights",
        type=Path,
        help="Path to the trained YOLO checkpoint.",
    )
    parser.add_argument(
        "data_yaml",
        type=Path,
        help="Path to the Ultralytics dataset YAML file.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=832,
        help="Inference image size used during evaluation.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size used during evaluation.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Device to use, for example '0' or 'cpu'.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/eval"),
        help="Directory where Ultralytics evaluation outputs will be saved.",
    )
    parser.add_argument(
        "--name-prefix",
        default="summary",
        help="Prefix for the Ultralytics run names.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/eval/evaluation_summary.json"),
        help="Path to the combined JSON summary file.",
    )
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    """Convert values to float when possible, otherwise return None."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_metrics(metrics: Any, split: str) -> dict[str, Any]:
    """Extract a compact summary from an Ultralytics metrics object."""
    class_names = {}
    if hasattr(metrics, "names") and metrics.names is not None:
        class_names = {int(k): str(v) for k, v in dict(metrics.names).items()}

    per_class: list[dict[str, Any]] = []
    maps = list(getattr(metrics.box, "maps", []))

    # Per-class precision/recall are not always exposed as stable public fields
    # across Ultralytics versions, so this summary guarantees mAP values and
    # includes the class name mapping.
    for class_id, class_name in class_names.items():
        map_5095 = safe_float(maps[class_id]) if class_id < len(maps) else None
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "map50_95": map_5095,
            }
        )

    return {
        "split": split,
        "num_classes": len(class_names),
        "metrics": {
            "precision": safe_float(metrics.box.mp),
            "recall": safe_float(metrics.box.mr),
            "map50": safe_float(metrics.box.map50),
            "map75": safe_float(metrics.box.map75),
            "map50_95": safe_float(metrics.box.map),
        },
        "per_class": per_class,
    }


def run_evaluation(
    model: YOLO,
    data_yaml: Path,
    split: str,
    imgsz: int,
    batch: int,
    device: str,
    project: Path,
    run_name: str,
) -> dict[str, Any]:
    """Run Ultralytics evaluation on one dataset split and return summary metrics."""
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        name=run_name,
        plots=True,
        save_json=True,
        verbose=True,
    )
    return extract_metrics(metrics, split)


def main() -> None:
    """Evaluate the checkpoint on validation and test and save a combined report."""
    args = parse_args()

    weights = args.weights.resolve()
    data_yaml = args.data_yaml.resolve()
    project = args.project.resolve()
    output = args.output.resolve()

    project.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))

    summary = {
        "weights": str(weights),
        "data_yaml": str(data_yaml),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "results": {},
    }

    for split in ("val", "test"):
        run_name = f"{args.name_prefix}_{split}"
        summary["results"][split] = run_evaluation(
            model=model,
            data_yaml=data_yaml,
            split=split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=project,
            run_name=run_name,
        )

    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nEvaluation summary\n")
    for split in ("val", "test"):
        result = summary["results"][split]
        metrics = result["metrics"]
        print(f"[{split}]")
        print(f"  Precision : {metrics['precision']}")
        print(f"  Recall    : {metrics['recall']}")
        print(f"  mAP50     : {metrics['map50']}")
        print(f"  mAP75     : {metrics['map75']}")
        print(f"  mAP50-95  : {metrics['map50_95']}")
        print()

    print(f"Combined summary saved to: {output}")


if __name__ == "__main__":
    main()
