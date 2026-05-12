"""
Dataset YAML discovery and preparation for Ultralytics YOLO training.

Ultralytics requires a YAML file with absolute paths for each split.
The source YAML (data.yaml from Roboflow or custom) often uses relative paths.
This module reads the source YAML and generates a canonical training YAML
(dataset.ultralytics.yaml) with absolute paths and a validated class list.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from waste_detector.utils.io import read_yaml, write_yaml


# ---------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------

@dataclass(slots=True)
class YoloDatasetSpec:
    """Metadata describing a validated YOLO dataset ready for training."""
    dataset_root: Path        # Root folder of the dataset
    source_yaml: Path         # The original data.yaml (from Roboflow etc.)
    training_yaml: Path       # The generated dataset.ultralytics.yaml
    nc: int                   # Number of classes
    names: List[str]          # Class names in order (index = YOLO class ID)


# ---------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------

def _find_source_yaml(dataset_root: Path) -> Path:
    """
    Locate the source dataset YAML inside 'dataset_root'.
    Tries 'data.yaml' first, then 'dataset.yaml'.
    Raises FileNotFoundError if neither is found.
    """
    for candidate in ("data.yaml", "dataset.yaml"):
        p = dataset_root / candidate
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"No 'data.yaml' or 'dataset.yaml' found in: {dataset_root}"
    )


def _normalize_names(raw_names: Any) -> List[str]:
    """
    Convert the 'names' field from a dataset YAML to a plain list of strings.

    Ultralytics YAMLs use either a list (['Cardboard', 'Glass']) or a dict
    ({0: 'Cardboard', 1: 'Glass'}). Both formats are handled here.
    """
    if isinstance(raw_names, list):
        return [str(n) for n in raw_names]
    if isinstance(raw_names, dict):
        # Sort by key to ensure the order matches the original class indices.
        return [str(raw_names[k]) for k in sorted(raw_names.keys())]
    raise ValueError(f"Unexpected 'names' format in dataset YAML: {type(raw_names)}")


def _detect_split_paths(dataset_root: Path) -> Dict[str, str]:
    """
    Auto-discover image directories for train/val/test splits.

    Tries the two common Roboflow layouts:
      - '<split>/images/'  (e.g. 'train/images/')
      - 'images/<split>/'  (e.g. 'images/train/')
    Also accepts 'valid' as an alias for 'val'.

    Returns a dict mapping split name ('train', 'val', 'test') to a
    relative path string from 'dataset_root'.
    """
    # Candidate directory names to try for each logical split.
    candidates: Dict[str, List[str]] = {
        "train": ["train/images", "images/train"],
        "val": ["valid/images", "val/images", "images/valid", "images/val"],
        "test": ["test/images", "images/test"],
    }

    found: Dict[str, str] = {}
    for split, paths in candidates.items():
        for rel_path in paths:
            if (dataset_root / rel_path).is_dir():
                found[split] = rel_path
                break
    return found


def _resolve_split_paths(
    dataset_root: Path,
    yaml_paths: Dict[str, Optional[str]],
) -> Dict[str, str]:
    """
    Merge split paths from the source YAML with auto-detected ones.

    The source YAML takes precedence; auto-detection fills in any gaps.
    Returns a dict with only the splits that could be resolved.
    """
    auto = _detect_split_paths(dataset_root)
    resolved: Dict[str, str] = {}

    for split in ("train", "val", "test"):
        # Prefer the path from the source YAML if it points to a real directory.
        yaml_val = yaml_paths.get(split)
        if yaml_val and (dataset_root / yaml_val).is_dir():
            resolved[split] = yaml_val
        elif split in auto:
            resolved[split] = auto[split]
        # If neither exists, the split is simply omitted (test split is optional).

    return resolved


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

def build_training_yaml(
    dataset_root: Path,
    *,
    output_name: str = "dataset.ultralytics.yaml",
) -> Path:
    """
    Read the source dataset YAML and write a canonical Ultralytics training YAML.

    The generated YAML uses the absolute dataset path and validates that
    'nc' matches the number of class names. It is written next to the source YAML.
    Returns the path of the generated file.
    """
    source_yaml_path = _find_source_yaml(dataset_root)
    payload = read_yaml(source_yaml_path)

    # Parse and normalise class names.
    raw_names = payload.get("names")
    if raw_names is None:
        raise ValueError(f"Dataset YAML '{source_yaml_path}' is missing a 'names' key.")
    names = _normalize_names(raw_names)

    # Validate that 'nc' matches the actual number of names.
    nc = payload.get("nc")
    if nc is not None and int(nc) != len(names):
        raise ValueError(
            f"Dataset YAML 'nc' ({nc}) does not match number of class names ({len(names)})."
        )

    # Determine absolute split paths.
    yaml_splits = {
        "train": payload.get("train"),
        "val": payload.get("val") or payload.get("valid"),
        "test": payload.get("test"),
    }
    splits = _resolve_split_paths(dataset_root, yaml_splits)

    if "train" not in splits:
        raise ValueError(
            f"Could not find a training image directory under: {dataset_root}"
        )

    # Build the canonical YAML payload with absolute paths.
    training_payload: Dict[str, Any] = {
        "path": str(dataset_root.resolve()),
        "nc": len(names),
        "names": names,
    }
    for split, rel_path in splits.items():
        training_payload[split] = rel_path

    # Write the generated YAML next to the source YAML.
    out_path = dataset_root / output_name
    write_yaml(out_path, training_payload)
    return out_path


def load_dataset_spec(
    dataset_root: Path,
    *,
    training_yaml_name: str = "dataset.ultralytics.yaml",
) -> YoloDatasetSpec:
    """
    Build (or rebuild) the Ultralytics training YAML and return a 'YoloDatasetSpec'.

    This is the main entry point for training and evaluation scripts.
    """
    source_yaml = _find_source_yaml(dataset_root)
    training_yaml = build_training_yaml(dataset_root, output_name=training_yaml_name)

    # Read back the generated YAML to extract validated metadata.
    payload = read_yaml(training_yaml)
    names = _normalize_names(payload["names"])

    return YoloDatasetSpec(
        dataset_root=dataset_root,
        source_yaml=source_yaml,
        training_yaml=training_yaml,
        nc=len(names),
        names=names,
    )
