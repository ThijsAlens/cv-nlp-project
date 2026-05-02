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

# Output folder for the saved JSON file and crop previews. Set to 'None' to
# use the detector default ('runs/detector_inference/<image_stem>/').
DEFAULT_OUTPUT_DIR: Optional[Path] = None


# =============================================================================
# Demo entry point
# =============================================================================


def main(output_dir: Optional[Path] = DEFAULT_OUTPUT_DIR) -> None:
    """Run the demo. Pass 'output_dir' to override where outputs are saved."""
    # -------------------------------------------------------------------------
    # Step 1: resolve the demo image to an absolute path so the script also
    #          works when launched from a different working directory.
    # -------------------------------------------------------------------------
    repo_root = Path(__file__).resolve().parent
    # If 'DEMO_IMAGE' is relative, anchor it at the repo root.
    demo_image = DEMO_IMAGE if DEMO_IMAGE.is_absolute() else repo_root / DEMO_IMAGE
    if not demo_image.is_file():
        raise SystemExit(f"Demo image not found at: {demo_image}")

    # -------------------------------------------------------------------------
    # Step 2: build the detector. Defaults already point at the latest
    #          checkpoint and the bin mapping JSON, so no extra arguments
    #          are required.
    # -------------------------------------------------------------------------
    detector = GarbageDetector()
    print(f"Loaded model weights: {detector.weights_path}")
    print(f"Known classes:        {detector.class_names}")

    # -------------------------------------------------------------------------
    # Step 3: run detection. Crops and the JSON file are saved by default.
    #          Forwarding 'output_dir' lets callers redirect the saved files.
    # -------------------------------------------------------------------------
    result = detector.detect(demo_image, output_dir=output_dir)

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
