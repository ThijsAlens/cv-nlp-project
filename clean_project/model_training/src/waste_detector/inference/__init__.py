"""Inference subpackage: bin mapping, raw prediction, full detection, and visualisation."""

from waste_detector.inference.detector import Detection, DetectionResult, GarbageDetector
from waste_detector.inference.bin_mapper import BinMappingError, load_bin_mapping, resolve_bin

__all__ = [
    "Detection",
    "DetectionResult",
    "GarbageDetector",
    "BinMappingError",
    "load_bin_mapping",
    "resolve_bin",
]
