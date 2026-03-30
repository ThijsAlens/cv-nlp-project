"""Helpers for working with label remapping definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from trash_detector.utils.io import read_json


class LabelMapError(RuntimeError):
    """Raised when a label mapping file is malformed."""



def build_identity_map(category_names: Iterable[str]) -> Dict[str, int]:
    """Build a one-to-one class mapping preserving the provided order."""
    return {name: index for index, name in enumerate(category_names)}



def load_explicit_label_map(path: Path, available_categories: List[str]) -> Dict[str, int]:
    """Load a label-map JSON file and validate that all requested classes exist.

    Expected JSON schema:
    {
      "classes": ["Class A", "Class B"]
    }
    """
    payload = read_json(path)
    classes = payload.get("classes")
    if not isinstance(classes, list) or not classes:
        raise LabelMapError(f"Label map {path} must contain a non-empty 'classes' list.")

    unknown = sorted(set(classes) - set(available_categories))
    if unknown:
        raise LabelMapError(
            f"Label map {path} contains unknown TACO classes: {', '.join(unknown)}"
        )
    return {name: index for index, name in enumerate(classes)}
