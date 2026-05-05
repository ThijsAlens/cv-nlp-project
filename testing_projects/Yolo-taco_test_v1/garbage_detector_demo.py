#!/usr/bin/env python3
"""Demo runner for the 'GarbageDetector'.

Loads the latest trained model, runs detection on a single example image, and
prints the resulting JSON-shaped dictionary. Saved crops and the JSON file land
under 'runs/detector_inference/<image_stem>/'.
"""

from __future__ import annotations

# =============================================================================
# Imports
# =============================================================================

import json
from pathlib import Path
from typing import Optional

from trash_detector.detector import GarbageDetector


# =============================================================================
# Configuration (edit before running)
# =============================================================================

# Demo image used by the runner. A single path string keeps the demo simple.
DEMO_IMAGE = Path("data/Totaal_dataset/inference_tests_visual/example_6.jpg")

# Trained model checkpoint used by the demo. The literal default below mirrors
# 'MODEL_RELATIVE_PATH' inside 'garbage_detector.py'; change it to point at a
# different run, or set it to 'None' to fall back to the module default.
MODEL_PATH: Optional[Path] = Path("runs/train/yolo11s_garbage_5c-2/weights/best.pt")

# Output folder for the saved JSON file and crop previews. The literal default
# below mirrors what the module would auto-create for 'example_6.jpg'; change
# it freely, or set it to 'None' to let the module pick the folder for you
# ('runs/detector_inference/<image_stem>/').
DEFAULT_OUTPUT_DIR: Optional[Path] = Path("runs/detector_inference/example_6")


# =============================================================================
# Demo entry point
# =============================================================================


def _anchor_at_repo_root(path: Optional[Path], repo_root: Path) -> Optional[Path]:
    """Anchor a relative path at the script's repo root.

    Absolute paths and 'None' are passed through unchanged so callers can mix
    relative defaults (easy to read) with absolute overrides (easy to script).
    """
    if path is None:
        return None
    return path if path.is_absolute() else repo_root / path


def main(
    output_dir: Optional[Path] = DEFAULT_OUTPUT_DIR,
    model_path: Optional[Path] = MODEL_PATH,
) -> None:
    """Run the demo.

    Pass 'output_dir' to override where outputs are saved, or 'model_path' to
    swap in a different trained checkpoint without editing the module.
    """
    # -------------------------------------------------------------------------
    # Step 1: resolve every configured path against the script location so the
    #          script works no matter the current working directory.
    # -------------------------------------------------------------------------
    repo_root = Path(__file__).resolve().parent
    demo_image = _anchor_at_repo_root(DEMO_IMAGE, repo_root)
    resolved_weights = _anchor_at_repo_root(model_path, repo_root)
    resolved_output_dir = _anchor_at_repo_root(output_dir, repo_root)

    # The demo image must exist; weights and output dir are validated downstream.
    if demo_image is None or not demo_image.is_file():
        raise SystemExit(f"Demo image not found at: {demo_image}")

    # -------------------------------------------------------------------------
    # Step 2: build the detector. Passing 'weights_path=None' lets the module
    #          pick its own default checkpoint; passing a real path overrides
    #          it without touching 'garbage_detector.py'.
    # -------------------------------------------------------------------------
    detector = GarbageDetector(weights_path=resolved_weights)
    print(f"Loaded model weights: {detector.weights_path}")
    print(f"Known classes:        {detector.class_names}")

    # -------------------------------------------------------------------------
    # Step 3: run detection. Crops and the JSON file are saved by default.
    #          Forwarding 'output_dir' lets callers redirect the saved files.
    # -------------------------------------------------------------------------
    result = detector.detect(demo_image, output_dir=resolved_output_dir)

    # -------------------------------------------------------------------------
    # Step 4: print a short summary, then the full JSON payload.
    # -------------------------------------------------------------------------
    print(f"\nProcessed image:  {result.image_path}")
    print(f"Image size (W x H): {result.image_size[0]} x {result.image_size[1]}")
    print(f"Output folder:    {result.output_dir}")
    print(f"JSON file:        {result.json_path}")
    print(f"Detections found: {len(result.detections)}")

    # Each detection prints its core attributes plus the saved crop path.
    for det in result.detections:
        print(
            f"  - #{det.index:02d} {det.material:>10s} -> bin '{det.bin}' "
            f"(conf {det.confidence:.2f})  crop: {det.crop_path}"
        )

    # Show the full JSON-style dict so a downstream consumer can see the schema.
    print("\nFull JSON payload:")
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


# =============================================================================
# Script guard
# =============================================================================


if __name__ == "__main__":
    main()
