"""
Crop showcase: run detection on an image and produce a grid of detected object crops
with their material names and disposal bin labels.

Two display modes are supported:
  - 'pretty' mode: clean labels showing only the bin name (for presentations).
  - debug mode:    shows material name, bin, and confidence score.

Output is saved to a timestamped folder inside the configured 'output_dir'.
"""

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from waste_detector.inference.bin_mapper import load_bin_mapping, resolve_bin
from waste_detector.utils.io import ensure_dir, write_json


# ---------------------------------------------------------------
# Data structure for one detected crop
# ---------------------------------------------------------------

@dataclass(frozen=True)
class DetectionCrop:
    """A single cropped region from a detected object, ready for visualisation."""
    material_name: str    # Detected material class name
    bin_name: str         # Mapped disposal bin key
    confidence: float     # Detection confidence in [0, 1]
    crop_rgb: np.ndarray  # Cropped image in RGB colour order


# ---------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------

def _utc_run_id() -> str:
    """Generate a timestamp string for naming output folders: YYYYMMDD_HHMMSS."""
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _expand_xyxy(
    xyxy: List[float],
    margin_frac: float,
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    """Expand a bounding box by a margin fraction and clip to image boundaries."""
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    bw = x2 - x1
    bh = y2 - y1
    mx = int(bw * margin_frac)
    my = int(bh * margin_frac)
    return (
        max(0, x1 - mx),
        max(0, y1 - my),
        min(img_w, x2 + mx),
        min(img_h, y2 + my),
    )


def _decorate_crop_axis(
    ax: plt.Axes,
    crop: DetectionCrop,
    pretty: bool,
) -> None:
    """Draw one panel of the crop grid on 'ax'."""
    # Display the crop image.
    ax.imshow(crop.crop_rgb)
    ax.set_xticks([])
    ax.set_yticks([])

    if pretty:
        # Show only the bin name with a coloured background bar.
        ax.set_xlabel(
            crop.bin_name,
            fontsize=10,
            fontweight="bold",
            color="white",
            labelpad=4,
        )
        ax.xaxis.label.set_backgroundcolor("#2e7d32")
    else:
        # Show material, bin, and confidence for debugging.
        label = (
            f"{crop.material_name}\n"
            f"Bin: {crop.bin_name}\n"
            f"Conf: {crop.confidence:.2f}"
        )
        ax.set_xlabel(label, fontsize=8)
        ax.set_title(crop.material_name, fontsize=9, pad=3)


def _build_crop_grid_figure(
    crops: List[DetectionCrop],
    title: Optional[str],
    pretty: bool,
) -> plt.Figure:
    """
    Build a matplotlib figure containing a grid of crop panels.

    Panels are arranged in rows of 4 columns.
    Returns the figure object (caller is responsible for saving or showing it).
    """
    n = len(crops)
    if n == 0:
        # Return a blank figure with a 'no detections' message.
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.text(0.5, 0.5, "No detections", ha="center", va="center")
        ax.axis("off")
        return fig

    # Calculate grid dimensions: up to 4 columns.
    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    # Normalise axes to always be a 1-D list for uniform iteration.
    axes_flat = np.array(axes).flatten()

    for i, crop in enumerate(crops):
        _decorate_crop_axis(axes_flat[i], crop, pretty)

    # Hide any unused panels in the last row.
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

def build_detection_crops(
    weights_path: Path,
    image_path: Path,
    *,
    bin_mapping_path: Path,
    conf: float = 0.25,
    imgsz: int = 640,
    device: str = "0",
    margin_frac: float = 0.05,
    model: Optional[YOLO] = None,
) -> Tuple[np.ndarray, List[DetectionCrop], Path]:
    """
    Run YOLO inference and collect cropped object images with their bin labels.

    If 'model' is provided it is reused (avoids reloading the checkpoint).
    Returns a tuple of:
      - 'image_rgb': the full source image as an RGB array
      - 'crops': list of DetectionCrop objects
      - 'weights_path': the checkpoint that was used
    """
    # Load model if not provided.
    if model is None:
        model = YOLO(str(weights_path))

    # Load bin mapping.
    bin_payload = load_bin_mapping(bin_mapping_path)

    # Read the source image and convert to RGB for matplotlib.
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = img_bgr.shape[:2]

    # Run inference.
    results = model.predict(
        img_bgr, conf=conf, imgsz=imgsz, device=device, save=False, verbose=False
    )

    crops: List[DetectionCrop] = []
    for box in results[0].boxes:
        class_id = int(box.cls.item())
        material = model.names[class_id]
        confidence = float(box.conf.item())
        xyxy = box.xyxy.squeeze().tolist()

        # Expand and clip the bounding box.
        cx1, cy1, cx2, cy2 = _expand_xyxy(xyxy, margin_frac, img_w, img_h)
        crop_rgb = image_rgb[cy1:cy2, cx1:cx2]

        # Map material to its disposal bin.
        bin_name = resolve_bin(bin_payload, material)

        crops.append(DetectionCrop(
            material_name=material,
            bin_name=bin_name,
            confidence=confidence,
            crop_rgb=crop_rgb,
        ))

    return image_rgb, crops, weights_path


def showcase_crops_to_file(
    crops: List[DetectionCrop],
    output_path: Path,
    *,
    title: Optional[str] = None,
    pretty: bool = True,
) -> None:
    """
    Save a crop grid figure as a PNG file to 'output_path'.

    The parent directory is created automatically if it does not exist.
    """
    ensure_dir(output_path.parent)
    fig = _build_crop_grid_figure(crops, title=title, pretty=pretty)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Crop grid saved to: {output_path}")


def run_showcase(
    weights_path: Path,
    image_path: Path,
    bin_mapping_path: Path,
    output_dir: Path,
    *,
    conf: float = 0.25,
    imgsz: int = 640,
    device: str = "0",
    margin_frac: float = 0.05,
    pretty: bool = True,
) -> Path:
    """
    Full showcase pipeline: detect, crop, visualise, and save all outputs.

    Creates a timestamped subfolder inside 'output_dir' containing:
      - 'crop_grid.png': the crop visualisation grid
      - 'detections.json': machine-readable detection list

    Returns the path to the output subfolder.
    """
    # Build a timestamped folder so each run has its own outputs.
    run_folder = output_dir / _utc_run_id()
    ensure_dir(run_folder)

    # Run detection and collect crops.
    _image_rgb, crops, _weights = build_detection_crops(
        weights_path=weights_path,
        image_path=image_path,
        bin_mapping_path=bin_mapping_path,
        conf=conf,
        imgsz=imgsz,
        device=device,
        margin_frac=margin_frac,
    )

    # Save the crop grid image.
    showcase_crops_to_file(
        crops,
        run_folder / "crop_grid.png",
        title=image_path.name,
        pretty=pretty,
    )

    # Save a JSON summary of all detections.
    detections_json = [
        {
            "material": c.material_name,
            "bin": c.bin_name,
            "confidence": round(c.confidence, 4),
        }
        for c in crops
    ]
    write_json(run_folder / "detections.json", {
        "image": str(image_path),
        "weights": str(weights_path),
        "detections": detections_json,
    })

    print(f"Showcase complete: {len(crops)} detection(s) in '{run_folder}'")
    return run_folder
