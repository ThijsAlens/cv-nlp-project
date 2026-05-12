"""
Material-to-bin resolution using the bin_mapping.json configuration file.

The JSON file maps detected material names (e.g. 'Cardboard') to disposal
bin keys (e.g. 'Paper'). A default bin ('Rest') is used when no match is found.

Schema expected in bin_mapping.json:
  {
    "default_bin": "Rest",
    "material_to_bin": {
      "Cardboard": "Paper",
      "Plastic": "PMD",
      ...
    }
  }
"""

from pathlib import Path
from typing import Any, Dict

from waste_detector.utils.io import read_json


# ---------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------

class BinMappingError(Exception):
    """Raised when the bin_mapping.json file is missing required keys."""


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

def load_bin_mapping(path: Path) -> Dict[str, Any]:
    """
    Read and validate bin_mapping.json.

    Returns the full payload dict. Call 'resolve_bin()' with this payload
    to look up individual materials.

    Raises BinMappingError if the file does not have the expected structure.
    """
    payload = read_json(path)

    # Validate that the required keys are present.
    if not isinstance(payload, dict) or "material_to_bin" not in payload:
        raise BinMappingError(
            f"'{path}' must be a JSON object with a 'material_to_bin' key."
        )
    if not isinstance(payload["material_to_bin"], dict):
        raise BinMappingError(
            f"'material_to_bin' in '{path}' must be a JSON object."
        )

    return payload


def resolve_bin(payload: Dict[str, Any], material_name: str) -> str:
    """
    Return the disposal bin key for 'material_name'.

    Lookup order:
      1. Exact match in 'material_to_bin'.
      2. Case-insensitive match (strips surrounding whitespace).
      3. Falls back to 'default_bin' (or 'Rest' if that key is absent).
    """
    material_map: Dict[str, str] = payload["material_to_bin"]
    default_bin: str = payload.get("default_bin", "Rest")

    # Exact match.
    if material_name in material_map:
        return material_map[material_name]

    # Case-insensitive fallback.
    normalised = material_name.strip().casefold()
    for key, bin_name in material_map.items():
        if key.strip().casefold() == normalised:
            return bin_name

    # No match found; use the default bin.
    return default_bin


def assert_mapping_covers_classes(payload: Dict[str, Any], class_names: list) -> None:
    """
    Raise ValueError if any class name in 'class_names' is missing from the mapping.

    Used during startup to catch mismatches between the model and bin_mapping.json
    before any inference is attempted.
    """
    material_map: Dict[str, str] = payload["material_to_bin"]
    missing = [name for name in class_names if name not in material_map]
    if missing:
        raise ValueError(
            f"bin_mapping.json is missing entries for: {missing}. "
            f"Add them to 'material_to_bin' or update the mapping file."
        )
