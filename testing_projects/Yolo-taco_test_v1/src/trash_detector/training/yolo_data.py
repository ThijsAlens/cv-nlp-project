"""Discover YOLO dataset layout and build a training YAML Ultralytics can load reliably."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from trash_detector.utils.io import read_yaml, write_yaml

_YAML_NAMES = ("data.yaml", "dataset.yaml")


@dataclass(slots=True)
class YoloDatasetSpec:
    """Summary of a detection dataset for training (class count comes from the YAML, not hard-coded)."""

    dataset_root: Path
    source_yaml: Path
    training_yaml: Path
    nc: int
    names: list[str]


def find_dataset_yaml(dataset_root: Path) -> Path:
    """Return ``data.yaml`` or ``dataset.yaml`` inside the dataset directory."""
    root = dataset_root.resolve()
    for name in _YAML_NAMES:
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No {_YAML_NAMES[0]} or {_YAML_NAMES[1]} found under {root}"
    )


def _normalize_names(names: Any) -> list[str]:
    if isinstance(names, list):
        return [str(n) for n in names]
    if isinstance(names, dict):
        items = sorted(((int(k), str(v)) for k, v in names.items()), key=lambda x: x[0])
        return [v for _, v in items]
    raise TypeError(f"Unsupported names format: {type(names)}")


def _first_existing_dir(root: Path, candidates: tuple[str, ...]) -> str | None:
    for rel in candidates:
        if (root / rel).is_dir():
            return rel
    return None


def detect_split_paths(dataset_root: Path) -> dict[str, str]:
    """Pick train/val/test image directories relative to *dataset_root* (common export layouts)."""
    root = dataset_root.resolve()
    train = _first_existing_dir(root, ("train/images", "images/train"))
    val = _first_existing_dir(root, ("valid/images", "val/images", "images/val"))
    test = _first_existing_dir(root, ("test/images", "images/test"))
    out: dict[str, str] = {}
    if train:
        out["train"] = train
    if val:
        out["val"] = val
    if test:
        out["test"] = test
    if "train" not in out or "val" not in out:
        raise FileNotFoundError(
            f"Could not find train/val image folders under {root}. "
            "Expected e.g. train/images and valid/images (Roboflow) or images/train and images/val (prepared export)."
        )
    return out


def _resolved_path_exists(dataset_root: Path, yaml_dir: Path, rel: str) -> bool:
    p = (yaml_dir / rel).resolve()
    if p.is_dir():
        return True
    p2 = (dataset_root / rel).resolve()
    return p2.is_dir()


def _relative_from_root(dataset_root: Path, folder: Path) -> str:
    folder = folder.resolve()
    root = dataset_root.resolve()
    return str(folder.relative_to(root)).replace("\\", "/")


def resolve_split_paths(
    dataset_root: Path, raw: Mapping[str, Any], yaml_path: Path
) -> dict[str, str]:
    """Map split keys to paths relative to *dataset_root* (for use with Ultralytics ``path``)."""
    root = dataset_root.resolve()
    yaml_dir = yaml_path.parent.resolve()
    detected = detect_split_paths(root)
    out: dict[str, str] = {}
    for key in ("train", "val", "test"):
        if key not in raw:
            if key in detected:
                out[key] = detected[key]
            continue
        rel = str(raw[key]).replace("\\", "/")
        if _resolved_path_exists(root, yaml_dir, rel):
            folder = (yaml_dir / rel).resolve()
            if not folder.is_dir():
                folder = (root / rel).resolve()
            out[key] = _relative_from_root(root, folder)
        elif key in detected:
            out[key] = detected[key]
        else:
            raise FileNotFoundError(
                f"Split {key!r} path {rel!r} does not exist under {root} and could not be inferred."
            )
    if "train" not in out or "val" not in out:
        raise FileNotFoundError(f"Resolved YAML must define train and val splits; got keys {list(out)}")
    return out


def build_training_yaml(
    dataset_root: Path,
    *,
    output_name: str = "dataset.ultralytics.yaml",
    drop_extra_keys: bool = True,
) -> Path:
    """Write a canonical Ultralytics YAML with absolute ``path`` and working split paths.

    Ultralytics replaces the detection head using ``nc`` (and names) from this file, so new datasets
    do not need any manual model head size configuration.
    """
    dataset_root = dataset_root.resolve()
    source = find_dataset_yaml(dataset_root)
    raw = read_yaml(source)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping in {source}, got {type(raw)}")

    names = _normalize_names(raw["names"])
    nc = int(raw["nc"])
    if nc != len(names):
        raise ValueError(f"nc={nc} but len(names)={len(names)} in {source}")

    splits = resolve_split_paths(dataset_root, raw, source)
    payload: dict[str, Any] = {
        "path": str(dataset_root),
        "train": splits["train"],
        "val": splits["val"],
        "nc": nc,
        "names": names,
    }
    if "test" in splits:
        payload["test"] = splits["test"]

    if not drop_extra_keys:
        for k, v in raw.items():
            if k in {"path", "train", "val", "test", "nc", "names"}:
                continue
            payload.setdefault(k, v)

    out_path = dataset_root / output_name
    write_yaml(out_path, payload)
    return out_path


def load_dataset_spec(
    dataset_root: Path,
    *,
    training_yaml_name: str = "dataset.ultralytics.yaml",
) -> YoloDatasetSpec:
    """Resolve layout, write the training YAML, and return metadata (including class count)."""
    dataset_root = dataset_root.resolve()
    source = find_dataset_yaml(dataset_root)
    training_yaml = build_training_yaml(dataset_root, output_name=training_yaml_name)
    raw = read_yaml(training_yaml)
    names = _normalize_names(raw["names"])
    nc = int(raw["nc"])
    return YoloDatasetSpec(
        dataset_root=dataset_root,
        source_yaml=source,
        training_yaml=training_yaml,
        nc=nc,
        names=names,
    )
