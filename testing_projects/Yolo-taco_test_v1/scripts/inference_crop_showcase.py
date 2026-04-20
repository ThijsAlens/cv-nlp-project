#!/usr/bin/env python3
"""Material-only inference demo for the Totaal_dataset-trained run (yolo11s_garbage_5c2).

Outputs are written under runs/inference/<timestamp>/ with a JSON manifest.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

# Non-interactive backend so figure export works without a display server.
import matplotlib

matplotlib.use("Agg")

# Allow running as `python scripts/inference_crop_showcase.py` without a package install.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ultralytics import YOLO

from trash_detector.inference.crop_showcase import build_detection_crops, showcase_crops_to_file
from trash_detector.utils.io import read_json, read_yaml, write_json

# -----------------------------------------------------------------------------
# Configuration: edit paths and inference parameters here only.
# Paths are relative to project_root unless they are absolute.
# -----------------------------------------------------------------------------
CONFIG: dict[str, Any] = {
    "project_root": _PROJECT_ROOT,
    # Trained weights for the Totaal_dataset five-class material model.
    "weights": Path("runs/train/yolo11s_garbage_5c2/weights/best.pt"),
    # Authoritative class list and order for that training run (names must match the model).
    "dataset_yaml": Path("data/Totaal_dataset/dataset.ultralytics.yaml"),
    # Maps the five material strings to household bin keys.
    "bin_mapping": Path("data/bin_mapping.json"),
    # Image to run detection on (set to a real file before running).
    "target_image": Path(r"data\Totaal_dataset\inference_tests_visual\example_6.jpg"),
    "conf": 0.25,
    "imgsz": 640,
    "device": "0",
    # Padding around each box, as a fraction of local box width and height.
    "margin_frac": 0.12,
    # If True, the saved grid uses only bin name and a numeric score in a clean footer.
    "PRETTY_IMAGES": True,
    # Parent folder for each timestamped inference run.
    "inference_output_root": Path("runs/inference"),
    "crop_grid_filename": "crop_grid.png",
    "detections_filename": "detections.json",
    "manifest_filename": "run_manifest.json",
}


def resolve_config_path(project_root: Path, value: Path) -> Path:
    """Turn a CONFIG path into an absolute Path on disk."""
    if value.is_absolute():
        return value
    return (project_root / value).resolve()


def load_dataset_class_names(dataset_yaml: Path) -> List[str]:
    """Read the dataset YAML and return model class names in dataset order."""
    payload = read_yaml(dataset_yaml)
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset YAML must parse to an object: {dataset_yaml}")

    raw_names = payload.get("names")
    if raw_names is None:
        raise ValueError(f"Dataset YAML missing 'names': {dataset_yaml}")

    if isinstance(raw_names, list):
        # List form matches Ultralytics exports where order equals class index.
        names = [str(item) for item in raw_names]
    elif isinstance(raw_names, dict):
        # Dict form maps integer ids to strings; YAML may stringify numeric keys.
        pairs: List[tuple[int, str]] = []
        for key, val in raw_names.items():
            try:
                idx = int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Unexpected 'names' key in {dataset_yaml}: {key!r}") from exc
            pairs.append((idx, str(val)))
        pairs.sort(key=lambda item: item[0])
        names = [label for _, label in pairs]
    else:
        raise ValueError(f"Unsupported 'names' shape in {dataset_yaml}")

    if not names:
        raise ValueError(f"Dataset YAML has empty 'names': {dataset_yaml}")

    declared_nc = payload.get("nc")
    if declared_nc is not None and int(declared_nc) != len(names):
        raise ValueError(
            f"Dataset YAML nc={declared_nc} does not match len(names)={len(names)} in {dataset_yaml}"
        )

    return names


def model_class_list(model: YOLO) -> List[str]:
    """Convert Ultralytics name map to a stable 0..N-1 list."""
    raw = model.names
    if isinstance(raw, dict):
        ordered: List[str] = []
        # Keys are contiguous class indices produced during training.
        for index in range(len(raw)):
            label = raw.get(index)
            if label is None:
                # Some serializers turn integer keys into strings; accept both shapes.
                label = raw.get(str(index))
            if label is None:
                raise ValueError(f"Model name map missing index {index}: {raw}")
            ordered.append(str(label))
        return ordered
    if isinstance(raw, list):
        return [str(item) for item in raw]
    raise ValueError(f"Unsupported model.names type: {type(raw)}")


def assert_bin_mapping_covers_dataset(bin_mapping_path: Path, expected_names: List[str]) -> None:
    """Ensure every dataset class string has a household bin assignment."""
    payload = read_json(bin_mapping_path)
    material_map = payload.get("material_to_bin")
    if not isinstance(material_map, dict):
        raise ValueError(f"'material_to_bin' must be an object in {bin_mapping_path}")

    missing = [name for name in expected_names if name not in material_map]
    if missing:
        raise ValueError(
            f"Bin mapping at {bin_mapping_path} lacks entries for dataset classes: {missing}"
        )


def assert_dataset_matches_model(expected_names: List[str], model: YOLO) -> None:
    """Fail fast when the checkpoint class head does not match the dataset YAML list."""
    weight_names = model_class_list(model)
    if expected_names != weight_names:
        raise ValueError(
            "Class names from the dataset YAML do not match the loaded weights.\n"
            f"  dataset_yaml: {expected_names}\n"
            f"  weights:      {weight_names}"
        )


def utc_run_id() -> str:
    """Build a filesystem-friendly timestamp for the run folder name."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]


def main() -> None:
    """Run inference, validate labels, write crops, detections, and manifest."""
    project_root: Path = CONFIG["project_root"]
    weights_path = resolve_config_path(project_root, CONFIG["weights"])
    dataset_yaml_path = resolve_config_path(project_root, CONFIG["dataset_yaml"])
    bin_mapping_path = resolve_config_path(project_root, CONFIG["bin_mapping"])
    target_image_path = resolve_config_path(project_root, CONFIG["target_image"])
    inference_root = resolve_config_path(project_root, CONFIG["inference_output_root"])

    if not weights_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
    if not dataset_yaml_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml_path}")
    if not bin_mapping_path.is_file():
        raise FileNotFoundError(f"Bin mapping JSON not found: {bin_mapping_path}")
    if not target_image_path.is_file():
        raise FileNotFoundError(f"Target image not found: {target_image_path}")

    expected_names = load_dataset_class_names(dataset_yaml_path)
    assert_bin_mapping_covers_dataset(bin_mapping_path, expected_names)

    # One model load: first for label checks, then for prediction in the shared helper.
    model = YOLO(str(weights_path))
    assert_dataset_matches_model(expected_names, model)

    _, crops, resolved_weights = build_detection_crops(
        weights_path,
        target_image_path,
        bin_mapping_path=bin_mapping_path,
        conf=float(CONFIG["conf"]),
        imgsz=int(CONFIG["imgsz"]),
        device=str(CONFIG["device"]),
        margin_frac=float(CONFIG["margin_frac"]),
        model=model,
    )

    run_id = utc_run_id()
    # Each execution gets a fresh directory so previous runs stay untouched.
    run_dir = inference_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root.resolve()),
        "weights": str(weights_path),
        "resolved_weights": str(resolved_weights),
        "dataset_yaml": str(dataset_yaml_path),
        "dataset_class_names": expected_names,
        "bin_mapping_json": str(bin_mapping_path),
        "target_image": str(target_image_path),
        "inference": {
            "conf": CONFIG["conf"],
            "imgsz": CONFIG["imgsz"],
            "device": CONFIG["device"],
            "margin_frac": CONFIG["margin_frac"],
            "PRETTY_IMAGES": bool(CONFIG["PRETTY_IMAGES"]),
        },
        "outputs": {
            "run_directory": str(run_dir),
            "crop_grid": CONFIG["crop_grid_filename"],
            "detections": CONFIG["detections_filename"],
            "manifest": CONFIG["manifest_filename"],
        },
        "detection_count": len(crops),
        "label_validation": {
            "dataset_yaml_matches_checkpoint": True,
        },
    }

    detections_payload = {
        "run_id": run_id,
        "weights": str(resolved_weights),
        "target_image": str(target_image_path),
        "detections": [
            {
                "material_name": item.material_name,
                "bin": item.bin_name,
                "confidence": item.confidence,
            }
            for item in crops
        ],
    }

    manifest_path = run_dir / str(CONFIG["manifest_filename"])
    detections_path = run_dir / str(CONFIG["detections_filename"])
    grid_path = run_dir / str(CONFIG["crop_grid_filename"])

    write_json(detections_path, detections_payload)

    title = f"Inference crops | {target_image_path.name}"

    if crops:
        # Matplotlib renders the grid to PNG; interactive display is intentionally disabled.
        showcase_crops_to_file(
            crops,
            grid_path,
            title=title,
            pretty=bool(CONFIG["PRETTY_IMAGES"]),
        )
        manifest["outputs"]["crop_grid_written"] = True
    else:
        # Still record the empty result on disk so runs stay comparable.
        empty_note = run_dir / "no_detections.txt"
        empty_note.write_text(
            "No boxes above the confidence threshold; crop_grid.png was not created.\n",
            encoding="utf-8",
        )
        manifest["outputs"]["crop_grid_written"] = False

    # Refresh manifest now that crop_grid_written is decided.
    write_json(manifest_path, manifest)

    # Short console summary for quick local checks.
    print(f"Run directory: {run_dir}")
    print(f"Detections: {len(crops)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
