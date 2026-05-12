"""
Prediction runner: run inference on a single image and print the detections.

Reads 'config/inference_config.yaml'. Detected objects are printed to the
console as a JSON-formatted list and saved to a 'detections.json' file in a
timestamped subfolder of the configured 'output_dir'.

Usage:
  uv run python scripts/run_predict.py
"""

import json
import sys
from pathlib import Path

# Add 'src' to the import path so 'waste_detector' can be found when
# running directly without installing the package.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from waste_detector.inference.detector import GarbageDetector
from waste_detector.utils.io import read_yaml

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

# Path to the inference configuration YAML.
CONFIG_PATH = _PROJECT_ROOT / "config" / "inference_config.yaml"

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main() -> None:
    # --- Load config ---
    cfg = read_yaml(CONFIG_PATH)

    # --- Resolve paths relative to the project root ---
    weights = Path(cfg["weights"])
    if not weights.is_absolute():
        weights = (_PROJECT_ROOT / weights).resolve()

    bin_mapping = Path(cfg["bin_mapping"])
    if not bin_mapping.is_absolute():
        bin_mapping = (_PROJECT_ROOT / bin_mapping).resolve()

    target_image = Path(cfg["target_image"])
    if not target_image.is_absolute():
        target_image = (_PROJECT_ROOT / target_image).resolve()

    output_dir = Path(cfg.get("output_dir", "./runs/inference"))
    if not output_dir.is_absolute():
        output_dir = (_PROJECT_ROOT / output_dir).resolve()

    conf = cfg.get("conf", 0.25)
    imgsz = cfg.get("imgsz", 640)
    device = str(cfg.get("device", "0"))

    # --- Validate that required files exist ---
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    if not target_image.is_file():
        raise FileNotFoundError(f"Target image not found: {target_image}")
    if not bin_mapping.is_file():
        raise FileNotFoundError(f"Bin mapping not found: {bin_mapping}")

    print(f"Model:  {weights}")
    print(f"Image:  {target_image}")

    # --- Load model and run detection ---
    detector = GarbageDetector(
        weights_path=weights,
        bin_mapping_path=bin_mapping,
        conf=conf,
        imgsz=imgsz,
        device=device,
    )

    # Use a timestamped subfolder so outputs from different runs are not overwritten.
    import datetime
    run_dir = output_dir / datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    result = detector.detect(
        target_image,
        output_dir=run_dir,
        save_crops=True,
        save_json=True,
    )

    # --- Print detections to console ---
    print(f"\nDetections ({len(result.detections)}):")
    detections_out = [d.to_dict() for d in result.detections]
    print(json.dumps(detections_out, indent=2))
    print(f"\nOutputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
