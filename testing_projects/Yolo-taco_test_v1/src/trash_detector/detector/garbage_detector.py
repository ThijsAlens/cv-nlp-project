"""High-level garbage detection class for downstream consumers.

The class in this file wraps an already-trained YOLO checkpoint and exposes a
single 'detect' call that:
  1. Runs object detection on an input image.
  2. Resolves each detected material to a household bin label.
  3. Saves a cropped preview JPG for every detection.
  4. Writes a JSON file describing all detections (and also returns a dict).

This module does not train models; it only consumes an existing checkpoint.
"""

from __future__ import annotations

# =============================================================================
# Imports
# =============================================================================

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import cv2
import numpy as np
from ultralytics import YOLO

# The bin mapping helpers already validate the JSON shape, so reuse them.
from trash_detector.inference.bin_mapping import (
    load_bin_mapping_payload,
    resolve_bin_for_material,
)


# =============================================================================
# Type aliases
# =============================================================================

# An image can be passed as a filesystem path or as a decoded numpy frame.
ImageInput = Union[str, Path, np.ndarray]


# =============================================================================
# Configuration (edit these defaults to change which model is used)
# =============================================================================

# This file lives at:
#   <repo_root>/src/trash_detector/detector/garbage_detector.py
# So the repository root is three parent folders up.
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Trained model used when no 'weights_path' is passed to 'GarbageDetector'.
# Change this string to point at a different run, then rebuild the path.
MODEL_RELATIVE_PATH = "runs/train/yolo11s_garbage_5c-2/weights/best.pt"
DEFAULT_WEIGHTS_PATH = _REPO_ROOT / MODEL_RELATIVE_PATH

# Default bin mapping JSON shipped with the dataset.
DEFAULT_BIN_MAPPING_PATH = _REPO_ROOT / "data" / "bin_mapping.json"

# Default folder for detector outputs (one subfolder per processed image).
DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "runs" / "detector_inference"


# =============================================================================
# Result containers
# =============================================================================


@dataclass(frozen=True, slots=True)
class Detection:
    """Information about a single detected object."""

    # Stable index inside the parent image (0, 1, 2, ...).
    index: int
    # Numeric class id from the YOLO model.
    class_id: int
    # Human-readable material label (for example 'Cardboard').
    material: str
    # Household bin label resolved from the material via 'bin_mapping.json'.
    bin: str
    # Detection confidence in the 0..1 range.
    confidence: float
    # Bounding box in the source image, format [x1, y1, x2, y2] in pixels.
    bbox_xyxy: List[float]
    # Path to the saved crop JPG (None when 'save_crops=False').
    crop_path: Optional[Path]

    def to_dict(self) -> dict:
        # JSON does not support 'Path', so stringify before dumping.
        return {
            "index": self.index,
            "class_id": self.class_id,
            "material": self.material,
            "bin": self.bin,
            "confidence": round(float(self.confidence), 4),
            "bbox_xyxy": [round(float(v), 2) for v in self.bbox_xyxy],
            "crop_path": str(self.crop_path) if self.crop_path is not None else None,
        }


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """All detections for one input image, plus output paths."""

    # Path to the source image; 'None' when the input was a raw numpy frame.
    image_path: Optional[Path]
    # Pixel size of the source image (width, height).
    image_size: tuple[int, int]
    # Path to the model weights used for inference.
    model_path: Path
    # Output folder where crops and JSON were written (None when nothing saved).
    output_dir: Optional[Path]
    # Path to the JSON summary file (None when 'save_json=False').
    json_path: Optional[Path]
    # The list of individual detections.
    detections: List[Detection] = field(default_factory=list)

    def to_dict(self) -> dict:
        # Compose the canonical JSON payload returned to callers.
        return {
            "image": {
                "path": str(self.image_path) if self.image_path is not None else None,
                "width": int(self.image_size[0]),
                "height": int(self.image_size[1]),
            },
            "model_path": str(self.model_path),
            "output_dir": str(self.output_dir) if self.output_dir is not None else None,
            "json_path": str(self.json_path) if self.json_path is not None else None,
            "num_detections": len(self.detections),
            "detections": [det.to_dict() for det in self.detections],
        }


# =============================================================================
# Helper functions
# =============================================================================


def _expand_and_clip_box(
    xyxy: Sequence[float],
    *,
    margin_frac: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Inflate a bounding box by 'margin_frac' on each side and clip to image bounds."""
    x1, y1, x2, y2 = xyxy
    # Compute box dimensions and pad the borders proportionally.
    box_w = max(float(x2) - float(x1), 1.0)
    box_h = max(float(y2) - float(y1), 1.0)
    pad_x = box_w * margin_frac
    pad_y = box_h * margin_frac
    # Convert to integer pixel coordinates and clamp inside the image.
    nx1 = int(max(0.0, math.floor(x1 - pad_x)))
    ny1 = int(max(0.0, math.floor(y1 - pad_y)))
    nx2 = int(min(float(image_width), math.ceil(x2 + pad_x)))
    ny2 = int(min(float(image_height), math.ceil(y2 + pad_y)))
    # Guarantee at least one pixel of width and height after clipping.
    if nx2 <= nx1:
        nx2 = min(image_width, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(image_height, ny1 + 1)
    return nx1, ny1, nx2, ny2


def _safe_filename_fragment(text: str) -> str:
    """Strip characters that are unsafe in filenames on common platforms."""
    # Whitelist: alphanumerics, dash and underscore. Replace everything else with '_'.
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    fragment = "".join(cleaned).strip("_")
    # Provide a stable fallback so empty or all-special inputs still produce a name.
    return fragment if fragment else "object"


# =============================================================================
# Main detector class
# =============================================================================


class GarbageDetector:
    """Single-entry-point class around a trained YOLO checkpoint.

    Typical usage:

        detector = GarbageDetector()
        result = detector.detect('path/to/image.jpg')
        print(result.to_dict())

    Construction loads the model exactly once, so the same instance can be
    reused for many calls without paying the load cost each time.
    """

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------

    def __init__(
        self,
        weights_path: Optional[Path | str] = None,
        bin_mapping_path: Optional[Path | str] = None,
        *,
        conf: float = 0.25,
        imgsz: int = 640,
        device: Optional[str] = None,
    ) -> None:
        """Load the detector and the bin mapping.

        Parameters:
            weights_path: 'best.pt' for the trained run. Defaults to the latest
                project model under 'runs/train/yolo11s_garbage_5c-2/'.
            bin_mapping_path: 'bin_mapping.json' that maps materials to bins.
            conf: minimum detection confidence kept by YOLO.
            imgsz: inference image size used by YOLO.
            device: 'cpu', '0', '0,1', etc. 'None' lets Ultralytics auto-pick.
        """
        # Resolve the weights path; fall back to the project default checkpoint.
        weights = Path(weights_path) if weights_path is not None else DEFAULT_WEIGHTS_PATH
        if not weights.is_file():
            raise FileNotFoundError(f"Could not find YOLO weights file at: {weights}")

        # Resolve the bin mapping JSON path.
        bin_path = Path(bin_mapping_path) if bin_mapping_path is not None else DEFAULT_BIN_MAPPING_PATH
        if not bin_path.is_file():
            raise FileNotFoundError(f"Could not find bin mapping JSON at: {bin_path}")

        # Load and validate the bin mapping payload up front so a bad JSON fails fast.
        self._bin_mapping_payload = load_bin_mapping_payload(bin_path)

        # Load the YOLO model only once; subsequent 'detect' calls reuse it.
        self._model = YOLO(str(weights))

        # Cache configuration for later 'detect' calls.
        self._weights_path = weights
        self._bin_mapping_path = bin_path
        self._conf = float(conf)
        self._imgsz = int(imgsz)
        self._device = device

    # -------------------------------------------------------------------------
    # Public properties
    # -------------------------------------------------------------------------

    @property
    def weights_path(self) -> Path:
        """Filesystem path of the loaded YOLO checkpoint."""
        return self._weights_path

    @property
    def class_names(self) -> dict[int, str]:
        """Mapping from numeric class id to material name as known by the model."""
        # 'self._model.names' is already a dict of {int: str}; copy it so callers
        # can not mutate the live model state.
        return {int(k): str(v) for k, v in self._model.names.items()}

    # -------------------------------------------------------------------------
    # Public detect entry point
    # -------------------------------------------------------------------------

    def detect(
        self,
        image: ImageInput,
        output_dir: Optional[Path | str] = None,
        *,
        save_crops: bool = True,
        save_json: bool = True,
        margin_frac: float = 0.05,
    ) -> DetectionResult:
        """Run detection on a single image and return all detected objects.

        Parameters:
            image: file path to an image, or a decoded numpy array (BGR or RGB).
            output_dir: where to write crops and the JSON summary. When 'None',
                a folder under 'runs/detector_inference/<image_stem>/' is used.
            save_crops: when 'True', a JPG crop is written for each detection.
            save_json: when 'True', a JSON summary file is written.
            margin_frac: extra padding around each bounding box before cropping,
                expressed as a fraction of the box size on each side.

        Returns:
            A 'DetectionResult' object containing one entry per detected object.
            An image with no detections still returns a valid result object with
            an empty 'detections' list.
        """
        # ---------------------------------------------------------------------
        # Step 1: normalize the input so YOLO and the cropping code agree on it.
        # ---------------------------------------------------------------------
        source_path: Optional[Path] = None
        source_for_yolo: Union[str, np.ndarray]

        if isinstance(image, (str, Path)):
            # Path-like input: validate the file exists and pass the string to YOLO.
            source_path = Path(image)
            if not source_path.is_file():
                raise FileNotFoundError(f"Image file does not exist: {source_path}")
            source_for_yolo = str(source_path)
        elif isinstance(image, np.ndarray):
            # Raw frame input: pass the array directly to YOLO; no source path.
            source_for_yolo = image
        else:
            raise TypeError(
                "Unsupported 'image' type. Expected a path-like or a numpy array, "
                f"got: {type(image).__name__}"
            )

        # ---------------------------------------------------------------------
        # Step 2: run inference. Disable Ultralytics' own saving so this class
        #          stays in full control of where files end up.
        # ---------------------------------------------------------------------
        results = self._model.predict(
            source=source_for_yolo,
            conf=self._conf,
            imgsz=self._imgsz,
            device=self._device,
            save=False,
            verbose=False,
        )
        if not results:
            raise RuntimeError("YOLO returned no results; check the model and the image.")

        # The detector processes a single image at a time, so take the first result.
        primary = results[0]

        # 'orig_img' is always populated for image inputs and is in BGR order.
        if primary.orig_img is None:
            raise RuntimeError("Inference result has no original image array.")
        original_bgr: np.ndarray = primary.orig_img
        height, width = original_bgr.shape[:2]

        # ---------------------------------------------------------------------
        # Step 3: pick the output directory when something will actually be
        #          written; this avoids creating empty folders.
        # ---------------------------------------------------------------------
        output_path: Optional[Path] = None
        if save_crops or save_json:
            if output_dir is not None:
                output_path = Path(output_dir)
            else:
                # Derive a stable subfolder name from the source image stem when
                # available, otherwise fall back to a generic 'array_input' label.
                stem = source_path.stem if source_path is not None else "array_input"
                output_path = DEFAULT_OUTPUT_ROOT / _safe_filename_fragment(stem)
            output_path.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------------------
        # Step 4: walk every detected box, crop it and label it.
        # ---------------------------------------------------------------------
        detections: List[Detection] = []
        boxes = primary.boxes

        if boxes is not None and len(boxes) > 0:
            # Use a stable base name for crop files derived from the source.
            base_name = (
                _safe_filename_fragment(source_path.stem)
                if source_path is not None
                else "array_input"
            )

            for index, box in enumerate(boxes):
                # Pull tensor values to plain Python types for JSON friendliness.
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                xyxy = [float(v) for v in box.xyxy[0].tolist()]

                # Resolve human-readable labels via the model and the JSON map.
                material_name = str(self._model.names[class_id])
                bin_name = resolve_bin_for_material(self._bin_mapping_payload, material_name)

                # Compute the cropping rectangle with a small extra margin so
                # the saved preview shows context around the object.
                x1, y1, x2, y2 = _expand_and_clip_box(
                    xyxy,
                    margin_frac=margin_frac,
                    image_width=width,
                    image_height=height,
                )
                # Numpy slicing keeps the original BGR order, which is what
                # OpenCV expects for 'imwrite'.
                crop_bgr = original_bgr[y1:y2, x1:x2]

                # Optionally write the crop to disk.
                crop_path: Optional[Path] = None
                if save_crops and output_path is not None:
                    safe_material = _safe_filename_fragment(material_name)
                    crop_filename = f"{base_name}_det{index:02d}_{safe_material}.jpg"
                    crop_path = output_path / crop_filename
                    # 'imwrite' silently fails on errors, so check the return.
                    success = cv2.imwrite(str(crop_path), crop_bgr)
                    if not success:
                        raise IOError(f"Failed to write crop image to: {crop_path}")

                # Record the detection in the result list.
                detections.append(
                    Detection(
                        index=index,
                        class_id=class_id,
                        material=material_name,
                        bin=bin_name,
                        confidence=confidence,
                        bbox_xyxy=xyxy,
                        crop_path=crop_path,
                    )
                )

        # ---------------------------------------------------------------------
        # Step 5: assemble the result; optionally write the JSON summary file.
        # ---------------------------------------------------------------------
        json_path: Optional[Path] = None
        if save_json and output_path is not None:
            json_name = (
                f"{_safe_filename_fragment(source_path.stem)}_detections.json"
                if source_path is not None
                else "array_input_detections.json"
            )
            json_path = output_path / json_name

        result = DetectionResult(
            image_path=source_path,
            image_size=(int(width), int(height)),
            model_path=self._weights_path,
            output_dir=output_path,
            json_path=json_path,
            detections=detections,
        )

        # Write the JSON last so 'json_path' inside it points at the actual file.
        if json_path is not None:
            with json_path.open("w", encoding="utf-8") as handle:
                json.dump(result.to_dict(), handle, indent=2, ensure_ascii=False)

        return result


# =============================================================================
# Convenience function
# =============================================================================


def detect_image(
    image: ImageInput,
    weights_path: Optional[Path | str] = None,
    bin_mapping_path: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
    *,
    conf: float = 0.25,
    imgsz: int = 640,
    device: Optional[str] = None,
    save_crops: bool = True,
    save_json: bool = True,
    margin_frac: float = 0.05,
) -> DetectionResult:
    """One-shot helper that builds a 'GarbageDetector' and runs one detection.

    Use this when calling code only needs to process a single image; for many
    images in a row, instantiate 'GarbageDetector' once and call its 'detect'
    method repeatedly to avoid reloading the model each time.
    """
    # Construct the detector with the requested model and threshold settings.
    detector = GarbageDetector(
        weights_path=weights_path,
        bin_mapping_path=bin_mapping_path,
        conf=conf,
        imgsz=imgsz,
        device=device,
    )
    # Delegate to the instance method so behavior stays identical.
    return detector.detect(
        image,
        output_dir=output_dir,
        save_crops=save_crops,
        save_json=save_json,
        margin_frac=margin_frac,
    )


# =============================================================================
# Public exports
# =============================================================================

__all__ = [
    "Detection",
    "DetectionResult",
    "GarbageDetector",
    "detect_image",
    "DEFAULT_WEIGHTS_PATH",
    "DEFAULT_BIN_MAPPING_PATH",
    "DEFAULT_OUTPUT_ROOT",
]
