"""Crop detections from an image and visualize them with material and bin labels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from ultralytics import YOLO

from trash_detector.inference.bin_mapping import load_bin_mapping_payload, resolve_bin_for_material


@dataclass(frozen=True, slots=True)
class DetectionCrop:
    """One detection plus the pixel crop taken from the source image."""

    material_name: str
    bin_name: str
    confidence: float
    crop_rgb: np.ndarray


def find_latest_best_weights(train_root: Path) -> Path:
    """Locate the newest 'best.pt' under runs/train across all experiment folders."""
    if not train_root.is_dir():
        raise FileNotFoundError(f"Training directory not found: {train_root}")
    candidates: List[Path] = list(train_root.glob("**/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints named best.pt found under {train_root}")
    # Sort by filesystem modification time so the most recently trained run wins.
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def expand_xyxy(
    xyxy: Sequence[float],
    *,
    margin_frac: float,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """Inflate an axis-aligned box by a fraction of its size, then clip to the image."""
    x1, y1, x2, y2 = xyxy
    box_w = max(float(x2) - float(x1), 1.0)
    box_h = max(float(y2) - float(y1), 1.0)
    pad_x = box_w * margin_frac
    pad_y = box_h * margin_frac
    # Integer pixel bounds keep numpy slicing predictable.
    nx1 = int(max(0.0, math.floor(x1 - pad_x)))
    ny1 = int(max(0.0, math.floor(y1 - pad_y)))
    nx2 = int(min(float(image_width), math.ceil(x2 + pad_x)))
    ny2 = int(min(float(image_height), math.ceil(y2 + pad_y)))
    if nx2 <= nx1:
        # Guarantee at least one horizontal pixel after clipping.
        nx2 = min(image_width, nx1 + 1)
    if ny2 <= ny1:
        # Guarantee at least one vertical pixel after clipping.
        ny2 = min(image_height, ny1 + 1)
    return nx1, ny1, nx2, ny2


def build_detection_crops(
    weights_path: Path | str,
    image_path: Path | str,
    *,
    bin_mapping_path: Path,
    conf: float = 0.25,
    imgsz: int = 640,
    device: str = "0",
    margin_frac: float = 0.1,
    model: Optional[YOLO] = None,
) -> Tuple[np.ndarray, List[DetectionCrop], Path]:
    """Run detection, map materials to bins, and return crops for visualization.

    If ``model`` is provided, it is used for inference and the weights path is only
    used to label the output path in the return value. This avoids loading a
    checkpoint twice when the caller already validated the model.
    """
    # Bin labels come from JSON so sorting rules can change without editing Python code.
    mapping_payload = load_bin_mapping_payload(bin_mapping_path)

    if model is None:
        # Load the checkpoint from disk when the caller does not pass a model instance.
        model = YOLO(str(weights_path))
    results = model.predict(
        source=str(image_path),
        conf=conf,
        save=False,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    if not results:
        raise RuntimeError("Ultralytics returned no results for the given image.")

    primary = results[0]
    if primary.orig_img is None:
        raise RuntimeError("Primary result has no original image array; check your source path.")

    # Predictor results keep the decoded frame in OpenCV channel order (BGR). Matplotlib treats
    # HWC arrays as RGB, so display and saved PNGs must convert to avoid shifted hues.
    image_rgb = primary.orig_img
    if image_rgb.ndim == 3 and image_rgb.shape[2] == 3:
        image_rgb = np.ascontiguousarray(image_rgb[..., ::-1])
    elif image_rgb.ndim == 3 and image_rgb.shape[2] == 4:
        # BGRA: reorder to RGBA for consistent display.
        image_rgb = np.ascontiguousarray(image_rgb[..., (2, 1, 0, 3)])

    height, width = image_rgb.shape[:2]

    crops: List[DetectionCrop] = []
    boxes = primary.boxes
    if boxes is None or len(boxes) == 0:
        return image_rgb, crops, Path(str(weights_path))

    # Materialize tensors once so each box is handled on the CPU side.
    for box in boxes:
        class_id = int(box.cls.item())
        # Ultralytics keeps human-readable names on the loaded model instance.
        material_name = str(model.names[class_id])
        # Household bin rules live in JSON and may include defaults for unseen labels.
        bin_name = resolve_bin_for_material(mapping_payload, material_name)
        xyxy_list = [float(value) for value in box.xyxy[0].tolist()]
        x1, y1, x2, y2 = expand_xyxy(
            xyxy_list,
            margin_frac=margin_frac,
            image_width=width,
            image_height=height,
        )
        crop = image_rgb[y1:y2, x1:x2].copy()
        crops.append(
            DetectionCrop(
                material_name=material_name,
                bin_name=bin_name,
                confidence=float(box.conf.item()),
                crop_rgb=crop,
            )
        )

    return image_rgb, crops, Path(str(weights_path))


def _decorate_crop_axis(axis: Axes, item: DetectionCrop, *, pretty: bool) -> None:
    """Draw one crop panel; ``pretty`` selects production styling versus experiment titles."""
    axis.imshow(item.crop_rgb)
    axis.set_axis_off()

    if pretty:
        # Reserve the bottom band for typography so the photo stays unobstructed above.
        footer_frac = 0.2
        strip = Rectangle(
            (0.0, 0.0),
            1.0,
            footer_frac,
            transform=axis.transAxes,
            facecolor="#f5f6fa",
            edgecolor="#dfe4ea",
            linewidth=1.0,
            clip_on=False,
            zorder=5,
        )
        axis.add_patch(strip)

        # Bin label (primary readout for sorting).
        axis.text(
            0.5,
            footer_frac * 0.62,
            item.bin_name,
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color="#2f3542",
            zorder=6,
        )

        # Confidence as a plain score (two decimals).
        axis.text(
            0.5,
            footer_frac * 0.28,
            f"{item.confidence:.2f}",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="#747d8c",
            zorder=6,
        )
        return

    # Experiment layout: retain material names and explicit prefixes for debugging.
    header = f"{item.material_name}\nBin: {item.bin_name}\nconf={item.confidence:.2f}"
    axis.set_title(header, fontsize=10)


def _build_crop_grid_figure(
    crops: Sequence[DetectionCrop],
    *,
    pretty: bool,
    title: str | None,
) -> Figure:
    """Compose the matplotlib figure shared by interactive display and file export."""
    count = len(crops)
    cols = int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(count / cols))

    # Roomier tiles when typography sits inside each panel.
    figsize = (3.85 * cols, 4.35 * rows) if pretty else (4.0 * cols, 4.2 * rows)
    figure = plt.figure(figsize=figsize, dpi=120 if pretty else 100)

    # Suptitles read as tooling chrome; omit them for the production layout.
    if title and not pretty:
        figure.suptitle(title, fontsize=12)

    # Tighter gutters keep the grid visually unified in production mode.
    hspace = 0.22 if pretty else 0.45
    wspace = 0.18 if pretty else 0.3
    grid = GridSpec(rows, cols, figure=figure, hspace=hspace, wspace=wspace)

    ctx: dict[str, object] = {}
    if pretty:
        ctx = {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "axes.edgecolor": "#dfe4ea",
        }

    def draw_panels() -> None:
        for index, item in enumerate(crops):
            row = index // cols
            col = index % cols
            axis = figure.add_subplot(grid[row, col])
            _decorate_crop_axis(axis, item, pretty=pretty)

    if pretty:
        with plt.rc_context(ctx):
            draw_panels()
    else:
        draw_panels()

    return figure


def showcase_crops(
    crops: Sequence[DetectionCrop],
    *,
    title: str | None = None,
    pretty: bool = False,
) -> None:
    """Display crops in a grid; ``pretty`` swaps in minimal bin-only labeling."""
    count = len(crops)
    if count == 0:
        raise ValueError("No crops to display; nothing was detected above the confidence threshold.")

    figure = _build_crop_grid_figure(crops, pretty=pretty, title=title)
    plt.show()
    plt.close(figure)


def showcase_crops_to_file(
    crops: Sequence[DetectionCrop],
    output_path: Path,
    *,
    title: str | None = None,
    pretty: bool = False,
) -> None:
    """Save a grid image to disk (useful on headless environments)."""
    count = len(crops)
    if count == 0:
        raise ValueError("No crops to save; nothing was detected above the confidence threshold.")

    figure = _build_crop_grid_figure(crops, pretty=pretty, title=title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Higher DPI keeps thin footer rules crisp when ``pretty`` is enabled.
    save_dpi = 175 if pretty else 150
    figure.savefig(output_path, dpi=save_dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
