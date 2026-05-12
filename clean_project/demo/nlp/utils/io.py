"""
File I/O helpers for reading JSON and YAML files.
All modules in this package should use these functions rather than
calling json/yaml directly, so error handling stays consistent.
"""

import json
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------

def read_json(path: Path) -> Any:
    """Read and return the contents of a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------

def read_yaml(path: Path) -> Any:
    """Read and return the contents of a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------

def read_text(path: Path) -> str:
    """Read and return the full text content of a file."""
    return path.read_text(encoding="utf-8")
