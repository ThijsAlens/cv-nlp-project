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
# Helpers
# ---------------------------------------------------------------

def _normalize_names(raw_names) -> list:
    """
    Convert the 'names' field from a YOLO dataset YAML to a list of strings.
    Accepts either a list ('[Glass, Metal, ...]') or a dict
    ('{0: Glass, 1: Metal, ...}') and preserves the original id order.
    """
    if raw_names is None:
        return []
    if isinstance(raw_names, list):
        return [str(n) for n in raw_names]
    if isinstance(raw_names, dict):
        # Sort by key so the order matches the original YOLO class ids.
        return [str(raw_names[k]) for k in sorted(raw_names.keys())]
    return []


def _read_dataset_classes(data_yaml: Path) -> list:
    """Read the 'names' field from a dataset YAML and return it as a list."""
    payload = read_yaml(data_yaml)
    return _normalize_names(payload.get("names"))


def _read_model_classes(weights: Path) -> list:
    """
    Load the YOLO checkpoint just to read its class names and return them as
    a list in YOLO id order. Loading is cheap and catches a checkpoint whose
    class count disagrees with the dataset before the evaluation starts.
    """
    # Local import so 'ultralytics' is only imported when this script runs.
    from ultralytics import YOLO

    model = YOLO(str(weights))
    return _normalize_names(getattr(model, "names", None))


def _warn_on_class_mismatch(
    expected_classes: list,
    dataset_classes: list,
    model_classes: list,
) -> None:
    """
    Compare the user-declared class list against the dataset YAML and the
    model checkpoint. Prints one warning per mismatch so config mistakes
    show up before the slow evaluation runs. Does not raise: a mismatched
    checkpoint can still be evaluated on purpose, and the evaluator itself
    will still run.
    """
    # Use sets for the membership comparison so a different list order does
    # not flag a mismatch when the actual class names line up.
    if expected_classes:
        if set(expected_classes) != set(dataset_classes):
            print(
                "WARNING: 'classes' in evaluate_config.yaml does not match the "
                "dataset YAML's class list.\n"
                f"  evaluate_config.classes: {expected_classes}\n"
                f"  dataset YAML names:      {dataset_classes}\n"
                "Evaluation will use the dataset YAML's class list."
            )
        if set(expected_classes) != set(model_classes):
            print(
                "WARNING: 'classes' in evaluate_config.yaml does not match the "
                "class list baked into the model checkpoint.\n"
                f"  evaluate_config.classes: {expected_classes}\n"
                f"  model checkpoint names:  {model_classes}\n"
                "This usually means the checkpoint was trained on a different "
                "class set than the one you intend to evaluate against. The "
                "resulting metrics and confusion matrix will reflect the "
                "checkpoint's classes, not the ones listed in this config."
            )

    # Also warn when the dataset and the model themselves disagree, since
    # that is the usual cause of confusing evaluation results.
    if set(dataset_classes) != set(model_classes):
        print(
            "WARNING: the dataset YAML and the model checkpoint declare "
            "different class sets.\n"
            f"  dataset YAML names:     {dataset_classes}\n"
            f"  model checkpoint names: {model_classes}\n"
            "The confusion matrix and per-class metrics will use the model's "
            "class list, not the dataset's. Re-train the model on the new "
            "class layout, or point 'weights' at a checkpoint that was "
            "trained with the same classes as the dataset."
        )


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main(config_path: Path = CONFIG_PATH) -> None:
    # --- Load config ---
    # 'config_path' defaults to the main evaluation config, but thin wrapper
    # scripts (e.g. 'run_finetune_evaluate.py') can pass a different YAML so
    # the fine-tune workflow can reuse this runner without editing it.
    cfg = read_yaml(config_path)

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
    # When enabled, the evaluator also reports precision / recall / F1
    # from the confusion matrix with the background row and column dropped.
    exclude_background = bool(cfg.get("exclude_background", False))

    # --- Validate the configured class list against the dataset and model ---
    # The 'classes' field is only a sanity check; the evaluator itself uses
    # the dataset YAML's class list. Running this check before evaluation
    # catches config mistakes (like pointing 'weights' at an old 5-class
    # checkpoint while 'dataset_yaml' references the 4-class merged dataset)
    # straight away, instead of after a long inference pass.
    expected_classes = list(cfg.get("classes", []) or [])
    dataset_classes = _read_dataset_classes(data_yaml)
    model_classes = _read_model_classes(weights)
    _warn_on_class_mismatch(expected_classes, dataset_classes, model_classes)

    print(f"Evaluating: {weights}")
    print(f"Dataset:    {data_yaml}")
    print(f"Splits:     {splits}")
    if expected_classes:
        print(f"Expected classes: {expected_classes}")
    print(f"Dataset YAML classes: {dataset_classes}")
    print(f"Model checkpoint classes: {model_classes}")

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
        exclude_background=exclude_background,
    )

    # --- Save and print results ---
    save_evaluation_report(summary, output)


if __name__ == "__main__":
    main()
