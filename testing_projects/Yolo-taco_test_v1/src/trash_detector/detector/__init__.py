"""Easy-to-import inference entry point for the trained garbage detector.

This subpackage is intentionally small and self-contained so other projects
(for example the NLP follow-up step) can import it without pulling in the
training stack.
"""

# Re-export the main entry point so callers can write a short import.
from trash_detector.detector.garbage_detector import (
    Detection,
    DetectionResult,
    GarbageDetector,
    detect_image,
)

__all__ = [
    "Detection",
    "DetectionResult",
    "GarbageDetector",
    "detect_image",
]
