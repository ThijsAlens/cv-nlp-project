#!/usr/bin/env python3
"""Evaluate a trained YOLO detection model on validation and test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from trash_detector.training.eval_report import evaluate_checkpoint, save_evaluation_report


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


def main() -> None:
    """Evaluate the checkpoint on validation and test and save a combined report."""
    args = parse_args()

    weights = args.weights.resolve()
    data_yaml = args.data_yaml.resolve()
    project = args.project.resolve()
    output = args.output.resolve()

    summary = evaluate_checkpoint(
        weights=weights,
        data_yaml=data_yaml,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        eval_project=project,
        run_name_prefix=args.name_prefix,
    )
    save_evaluation_report(summary, output)

    print("\nEvaluation summary\n")
    for split in ("val", "test"):
        result = summary["results"][split]
        metrics = result["metrics"]
        print(f"[{split}]")
        print(f"  Precision : {metrics['precision']}")
        print(f"  Recall    : {metrics['recall']}")
        print(f"  F1 (mean) : {metrics['f1']}")
        print(f"  mAP50     : {metrics['map50']}")
        print(f"  mAP75     : {metrics['map75']}")
        print(f"  mAP50-95  : {metrics['map50_95']}")
        print(f"  Artifacts : {result.get('ultralytics_save_dir')}")
        print()

    print(f"Combined summary saved to: {output}")


if __name__ == "__main__":
    main()
