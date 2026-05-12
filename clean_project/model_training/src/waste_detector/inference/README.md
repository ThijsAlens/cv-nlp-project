# inference/

Everything needed to run a trained model and interpret its output.

## Modules

| Module | Purpose |
|--------|---------|
| `bin_mapper.py` | Loads `bin_mapping.json` and resolves material names to disposal bin keys. |
| `predictor.py` | `TrashPredictor` - lightweight YOLO wrapper that returns detections as plain dicts. Use when you only need raw bounding box data. |
| `detector.py` | `GarbageDetector` - full pipeline: runs YOLO, crops objects, maps bins, saves crops and JSON. Use this for the demo and downstream NLP integration. |
| `crop_showcase.py` | Builds a matplotlib PNG grid of detected crops labelled with their bin names. Also provides `run_showcase()` as a one-call convenience. |

## When to use which module

- Need raw detections only -> `predictor.py` (`TrashPredictor`)
- Need structured results + saved crops -> `detector.py` (`GarbageDetector`)
- Need a visual crop grid for inspection -> `crop_showcase.py` (`run_showcase`)
- Need bin lookup without running the model -> `bin_mapper.py`

## GarbageDetector output

`detect()` returns a `DetectionResult` with a list of `Detection` objects.
Each `Detection` has: `material`, `bin`, `confidence`, `bbox_xyxy`, and `crop_path`.
Call `.to_dict()` on either to get a JSON-serialisable representation.
