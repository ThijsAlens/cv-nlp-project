"""
Crop showcase runner: detect objects in an image and save a visualisation grid.

Reads 'config/inference_config.yaml'. Produces a PNG grid showing each detected
object crop labelled with its disposal bin. Also saves a 'detections.json' file.

The 'pretty_images' flag in the config controls whether bin names only (true)
or full debug information (false) is shown on each crop panel.

Usage:
  uv run python scripts/run_crop_showcase.py
"""

import sys
from pathlib import Path

# Add 'src' to the import path so 'waste_detector' can be found when
# running directly without installing the package.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from waste_detector.inference.crop_showcase import run_showcase
from waste_detector.utils.io import read_yaml

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

# Path to the inference configuration YAML (shared with run_predict.py).
CONFIG_PATH = _PROJECT_ROOT / "config" / "inference_config.yaml"

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main() -> None:
    # --- Load config ---
    cfg = read_yaml(CONFIG_PATH)

    # --- Resolve paths relative to project root ---
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
    margin_frac = cfg.get("margin_frac", 0.05)
    pretty = cfg.get("pretty_images", True)

    # --- Validate required files exist ---
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    if not target_image.is_file():
        raise FileNotFoundError(f"Target image not found: {target_image}")
    if not bin_mapping.is_file():
        raise FileNotFoundError(f"Bin mapping not found: {bin_mapping}")

    print(f"Model:  {weights}")
    print(f"Image:  {target_image}")
    print(f"Pretty: {pretty}")

    # --- Run the full showcase pipeline ---
    run_folder = run_showcase(
        weights_path=weights,
        image_path=target_image,
        bin_mapping_path=bin_mapping,
        output_dir=output_dir,
        conf=conf,
        imgsz=imgsz,
        device=device,
        margin_frac=margin_frac,
        pretty=pretty,
    )

    print(f"Outputs saved to: {run_folder}")


if __name__ == "__main__":
    main()
