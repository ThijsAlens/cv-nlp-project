"""Resolve household bin labels from trained material class names."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trash_detector.utils.io import read_json


class BinMappingError(RuntimeError):
    """Raised when the bin mapping JSON cannot be interpreted."""


def load_bin_mapping_payload(path: Path) -> dict[str, Any]:
    """Load the raw bin mapping JSON file from disk."""
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise BinMappingError(f"Bin mapping file {path} must contain a JSON object.")
    material_map = payload.get("material_to_bin")
    if not isinstance(material_map, dict) or not material_map:
        raise BinMappingError(f"Bin mapping file {path} must define a non-empty 'material_to_bin' object.")
    default_bin = payload.get("default_bin")
    if default_bin is not None and not isinstance(default_bin, str):
        raise BinMappingError("'default_bin', when present, must be a string.")
    return payload


def resolve_bin_for_material(payload: dict[str, Any], material_name: str) -> str:
    """Return the bin key for a detector class label, falling back when unknown."""
    material_map_any = payload.get("material_to_bin")
    assert isinstance(material_map_any, dict)
    material_map: dict[str, str] = {}
    for raw_key, raw_val in material_map_any.items():
        if not isinstance(raw_key, str) or not isinstance(raw_val, str):
            raise BinMappingError("'material_to_bin' keys and values must be strings.")
        material_map[raw_key] = raw_val

    default_bin = payload.get("default_bin", "Rest")
    assert isinstance(default_bin, str)

    direct = material_map.get(material_name)
    if direct is not None:
        return direct

    # Fall back when YAML or UI strings differ slightly from the JSON spellings.
    target = material_name.strip().casefold()
    for key, value in material_map.items():
        if key.strip().casefold() == target:
            return value

    return default_bin
