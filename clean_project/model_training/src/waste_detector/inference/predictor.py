"""
TrashPredictor: a lightweight wrapper around Ultralytics YOLO for quick inference.

Returns detections as plain Python dicts. Use 'GarbageDetector' from 'detector.py'
for the full pipeline that also crops objects, maps bins, and saves JSON output.
"""

from pathlib import Path
from typing import Any, Dict, List, Union

from ultralytics import YOLO


class TrashPredictor:
    """Runs YOLO inference and returns detections as a list of dicts."""

    def __init__(self, weights_path: Union[Path, str]) -> None:
        # Load the model once at construction time.
        self.model = YOLO(str(weights_path))

    def predict(
        self,
        source: Union[str, Path],
        *,
        conf: float = 0.25,
        imgsz: int = 640,
        device: str = "0",
        save: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Run inference on 'source' (image path, directory, or video).

        Returns a list of dicts, one per detected object:
          {
            'source':     path to the source image,
            'class_id':   integer YOLO class index,
            'class_name': string class label,
            'confidence': float in [0, 1],
            'xyxy':       [x1, y1, x2, y2] bounding box in pixel coordinates,
          }
        """
        results = self.model.predict(
            source=str(source),
            conf=conf,
            imgsz=imgsz,
            device=device,
            save=save,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        for result in results:
            # Each 'result' corresponds to one input image.
            for box in result.boxes:
                detections.append({
                    "source": str(result.path),
                    "class_id": int(box.cls.item()),
                    "class_name": result.names[int(box.cls.item())],
                    "confidence": float(box.conf.item()),
                    "xyxy": box.xyxy.squeeze().tolist(),
                })

        return detections
