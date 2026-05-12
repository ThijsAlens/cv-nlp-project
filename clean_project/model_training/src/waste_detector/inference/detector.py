"""
GarbageDetector: the main inference class for the waste material detector.

Runs YOLO on an image, crops each detected object, maps the material name
to its disposal bin, saves crops and a JSON summary, and returns a structured
DetectionResult object.

Typical usage:
  detector = GarbageDetector(weights_path, bin_mapping_path)
  result = detector.detect(image_path, output_dir)
  print(result.to_dict())
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from ultralytics import YOLO

from waste_detector.inference.bin_mapper import load_bin_mapping, resolve_bin
from waste_detector.utils.io import ensure_dir, write_json


# Type alias: an image can be a file path or a NumPy array.
ImageInput = Union[str, Path, np.ndarray]


# ---------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------

@dataclass(frozen=True)
class Detection:
    """A single detected object in one image."""
    index: int               # 0-based index among all detections for this image
    class_id: int            # YOLO class index
    material: str            # Class name (e.g. 'Cardboard')
    bin: str                 # Disposal bin (e.g. 'Paper')
    confidence: float        # Detection confidence in [0, 1]
    bbox_xyxy: List[float]   # Bounding box [x1, y1, x2, y2] in pixel coordinates
    crop_path: Optional[Path] = None  # Path to the saved crop image (if any)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation of this detection."""
        return {
            "index": self.index,
            "class_id": self.class_id,
            "material": self.material,
            "bin": self.bin,
            "confidence": self.confidence,
            "bbox_xyxy": self.bbox_xyxy,
            "crop_path": str(self.crop_path) if self.crop_path else None,
        }


@dataclass(frozen=True)
class DetectionResult:
    """All detections for a single image, plus metadata about the run."""
    image_path: Optional[Path]    # Source image path (None if a NumPy array was passed)
    image_size: tuple             # (width, height) in pixels
    model_path: Path              # Checkpoint used for this inference run
    output_dir: Optional[Path]    # Directory where crops and JSON were saved
    json_path: Optional[Path]     # Path to the saved JSON file (if any)
    detections: List[Detection]   # List of all detected objects

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation of the full result."""
        return {
            "image_path": str(self.image_path) if self.image_path else None,
            "image_size": {"width": self.image_size[0], "height": self.image_size[1]},
            "model_path": str(self.model_path),
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "json_path": str(self.json_path) if self.json_path else None,
            "detections": [d.to_dict() for d in self.detections],
        }


# ---------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------

def _expand_and_clip_box(
    x1: int, y1: int, x2: int, y2: int,
    margin_frac: float,
    img_w: int,
    img_h: int,
) -> tuple:
    """
    Expand a bounding box by 'margin_frac' of its size on each side,
    then clip the result to the image boundaries.
    """
    # Compute margin as a fraction of the shorter box dimension.
    box_w = x2 - x1
    box_h = y2 - y1
    mx = int(box_w * margin_frac)
    my = int(box_h * margin_frac)

    # Expand and clip to image bounds.
    nx1 = max(0, x1 - mx)
    ny1 = max(0, y1 - my)
    nx2 = min(img_w, x2 + mx)
    ny2 = min(img_h, y2 + my)
    return nx1, ny1, nx2, ny2


def _safe_filename(material_name: str) -> str:
    """Return a filesystem-safe version of a material name."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in material_name)


# ---------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------

class GarbageDetector:
    """
    High-level detector that runs YOLO inference and maps results to disposal bins.

    Load once and call 'detect()' for each image.
    """

    def __init__(
        self,
        weights_path: Union[Path, str],
        bin_mapping_path: Union[Path, str],
        *,
        conf: float = 0.25,
        imgsz: int = 640,
        device: Optional[str] = None,
    ) -> None:
        self._weights_path = Path(weights_path)
        self._conf = conf
        self._imgsz = imgsz
        self._device = device

        # Load the YOLO model.
        if not self._weights_path.is_file():
            raise FileNotFoundError(f"Model weights not found: {self._weights_path}")
        self._model = YOLO(str(self._weights_path))

        # Load the bin mapping configuration.
        bin_path = Path(bin_mapping_path)
        if not bin_path.is_file():
            raise FileNotFoundError(f"Bin mapping file not found: {bin_path}")
        self._bin_mapping = load_bin_mapping(bin_path)

    # -----------------------------------------------------------
    # Properties
    # -----------------------------------------------------------

    @property
    def weights_path(self) -> Path:
        """Path to the loaded model checkpoint."""
        return self._weights_path

    @property
    def class_names(self) -> Dict[int, str]:
        """Dict mapping YOLO class index to class name string."""
        return self._model.names

    # -----------------------------------------------------------
    # Main detection method
    # -----------------------------------------------------------

    def detect(
        self,
        image: ImageInput,
        output_dir: Optional[Union[Path, str]] = None,
        *,
        save_crops: bool = True,
        save_json: bool = True,
        margin_frac: float = 0.05,
    ) -> DetectionResult:
        """
        Run detection on 'image' and return a DetectionResult.

        Steps:
          1. Load the image (if a path was given) or use the array directly.
          2. Run YOLO inference.
          3. For each detection: expand the bbox, crop the image, map to a bin.
          4. Optionally save crops to 'output_dir' as individual JPEGs.
          5. Optionally save a detections.json to 'output_dir'.

        Parameters:
          image      - File path, Path object, or RGB NumPy array.
          output_dir - Where to write crops and JSON. Auto-generated if None.
          save_crops - Whether to save individual crop images.
          save_json  - Whether to save a detections.json file.
          margin_frac - Fraction of box size to add as padding around each crop.
        """
        # --- Load image ---
        image_path: Optional[Path] = None
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")
        else:
            # Assume the array is already in RGB order; convert to BGR for OpenCV.
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w = img_bgr.shape[:2]

        # --- Run YOLO inference ---
        predict_kwargs = dict(conf=self._conf, imgsz=self._imgsz, save=False, verbose=False)
        if self._device is not None:
            predict_kwargs["device"] = self._device
        yolo_results = self._model.predict(img_bgr, **predict_kwargs)

        # --- Resolve output directory ---
        out_dir: Optional[Path] = None
        if save_crops or save_json:
            out_dir = Path(output_dir) if output_dir else (
                Path("runs") / "detector_inference" / "latest"
            )
            ensure_dir(out_dir)

        # --- Process each detection box ---
        detections: List[Detection] = []
        for idx, box in enumerate(yolo_results[0].boxes):
            class_id = int(box.cls.item())
            material = self._model.names[class_id]
            confidence = float(box.conf.item())

            # Raw bounding box coordinates in pixels.
            x1, y1, x2, y2 = [int(v) for v in box.xyxy.squeeze().tolist()]

            # Expand the box with a margin and clip to image bounds.
            cx1, cy1, cx2, cy2 = _expand_and_clip_box(
                x1, y1, x2, y2, margin_frac, img_w, img_h
            )

            # Map the detected material to its disposal bin.
            bin_name = resolve_bin(self._bin_mapping, material)

            # Optionally save the crop as a JPEG file.
            crop_path: Optional[Path] = None
            if save_crops and out_dir is not None:
                crop_bgr = img_bgr[cy1:cy2, cx1:cx2]
                safe_name = _safe_filename(material)
                crop_path = out_dir / f"{idx:02d}_{safe_name}.jpg"
                cv2.imwrite(str(crop_path), crop_bgr)

            detections.append(Detection(
                index=idx,
                class_id=class_id,
                material=material,
                bin=bin_name,
                confidence=confidence,
                bbox_xyxy=[x1, y1, x2, y2],
                crop_path=crop_path,
            ))

        # --- Optionally save JSON ---
        json_path: Optional[Path] = None
        if save_json and out_dir is not None:
            json_path = out_dir / "detections.json"
            payload = {
                "image_path": str(image_path) if image_path else None,
                "image_size": {"width": img_w, "height": img_h},
                "model_path": str(self._weights_path),
                "detections": [d.to_dict() for d in detections],
            }
            write_json(json_path, payload)

        return DetectionResult(
            image_path=image_path,
            image_size=(img_w, img_h),
            model_path=self._weights_path,
            output_dir=out_dir,
            json_path=json_path,
            detections=detections,
        )
