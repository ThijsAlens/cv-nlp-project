#!/usr/bin/env python3
"""Small example: load bin mapping JSON and print resolved bins for a few materials."""

from __future__ import annotations

from pathlib import Path

from trash_detector.inference.bin_mapping import load_bin_mapping_payload, resolve_bin_for_material


def main() -> None:
    # Resolve paths from this file so the script works no matter the current working directory.
    repo_root = Path(__file__).resolve().parent
    mapping_path = repo_root / "data" / "bin_mapping.json"

    # Load once; reuse the dict for many lookups in a real app.
    payload = load_bin_mapping_payload(mapping_path)

    # Pull example keys from the JSON so the demo stays aligned with your file.
    material_map = payload.get("material_to_bin")
    if not isinstance(material_map, dict) or not material_map:
        raise SystemExit("bin_mapping.json is missing a non-empty 'material_to_bin' object.")

    # Walk a few entries in stable sorted order for readable console output.
    sample_names = sorted(material_map.keys())[:8]
    print(f"Loaded mapping from: {mapping_path}")
    for name in sample_names:
        # Each call returns the bin key chosen by the same rules your services would use.
        bin_key = resolve_bin_for_material(payload, name)
        print(f"  {name!r} -> {bin_key!r}")

    # Also show how an unknown label falls back to the default bin.
    unknown = "__not_a_real_material__"
    fallback = resolve_bin_for_material(payload, unknown)
    print(f"  {unknown!r} -> {fallback!r}  (expected fallback)")


if __name__ == "__main__":
    main()
