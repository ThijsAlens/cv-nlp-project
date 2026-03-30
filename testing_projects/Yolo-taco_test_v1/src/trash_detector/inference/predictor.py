"""Prediction utilities for local images or videos."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from ultralytics import YOLO


class TrashPredictor:
    """Run inference using a trained YOLO checkpoint."""

    def __init__(self, weights_path: Path | str) -> None:
        self.model = YOLO(str(weights_path))

    def predict(
        self,
        source: str | Path,
        conf: float = 0.25,
        save: bool = True,
        imgsz: int = 640,
        device: str = "0",
    ) -> List[dict]:
        """Run inference and convert the results into plain Python dictionaries."""
        results = self.model.predict(source=str(source), conf=conf, save=save, imgsz=imgsz, device=device)
        predictions: List[dict] = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls.item())
                predictions.append(
                    {
                        "source": result.path,
                        "class_id": class_id,
                        "class_name": self.model.names[class_id],
                        "confidence": float(box.conf.item()),
                        "xyxy": [float(value) for value in box.xyxy[0].tolist()],
                    }
                )
        return predictions
