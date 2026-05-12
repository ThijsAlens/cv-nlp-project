"""
Evaluation runner for a trained waste detector checkpoint.

Reads 'config/evaluate_config.yaml' and evaluates the specified checkpoint
on the configured dataset splits. Saves a JSON report with all metrics.

Usage:
  uv run python scripts/run_evaluate.py
"""

import sys
from pathlib import Path

# Add 'src' to the import path so 'waste_detector' can be found when
# running directly without installing the package.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from waste_detector.training.evaluator import evaluate_checkpoint, save_evaluation_report
from waste_detector.utils.io import read_yaml

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

# Path to the evaluation configuration YAML.
CONFIG_PATH = _PROJECT_ROOT / "config" / "evaluate_config.yaml"

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main() -> None:
    # --- Load config ---
    cfg = read_yaml(CONFIG_PATH)

    # --- Resolve paths (support relative paths from project root) ---
    weights = Path(cfg["weights"])
    if not weights.is_absolute():
        weights = (_PROJECT_ROOT / weights).resolve()

    data_yaml = Path(cfg["dataset_yaml"])
    if not data_yaml.is_absolute():
        data_yaml = (_PROJECT_ROOT / data_yaml).resolve()

    output = Path(cfg.get("output", "./runs/evaluate/results.json"))
    if not output.is_absolute():
        output = (_PROJECT_ROOT / output).resolve()

    # --- Validate that required files exist before starting ---
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    splits = cfg.get("splits", ["val"])
    imgsz = cfg.get("imgsz", 640)
    batch = cfg.get("batch", 8)
    device = str(cfg.get("device", "0"))

    print(f"Evaluating: {weights}")
    print(f"Dataset:    {data_yaml}")
    print(f"Splits:     {splits}")

    # --- Run evaluation ---
    summary = evaluate_checkpoint(
        weights=weights,
        data_yaml=data_yaml,
        splits=splits,
        imgsz=imgsz,
        batch=batch,
        device=device,
        eval_project=str(output.parent),
        run_name_prefix=weights.parent.parent.name,
    )

    # --- Save and print results ---
    save_evaluation_report(summary, output)


if __name__ == "__main__":
    main()
