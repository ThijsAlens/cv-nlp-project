"""
File I/O helpers for reading and writing JSON and YAML files.
All other modules should use these functions instead of calling json/yaml directly.
"""

import json
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    """Create a directory (and any missing parents) if it does not exist yet."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------

def read_json(path: Path) -> Any:
    """Read and return the contents of a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    """Write 'payload' to a JSON file with 2-space indentation."""
    # Ensure the parent directory exists before writing.
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------

def read_yaml(path: Path) -> Any:
    """Read and return the contents of a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, payload: Any) -> None:
    """Write 'payload' to a YAML file."""
    # Ensure the parent directory exists before writing.
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
