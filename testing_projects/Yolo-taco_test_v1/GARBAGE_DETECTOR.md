# Garbage detector (`trash_detector.detector`)

A small, importable wrapper around the trained YOLO checkpoint. Built for
downstream consumers (for example the NLP step) that want one image in, and a
JSON-shaped dictionary plus per-object crops out, with no exposure to YOLO,
OpenCV, or matplotlib internals.

This module does **not** train models. It only consumes an existing checkpoint
produced by `scripts/run_train.py`.

## TL;DR

```python
from trash_detector.detector import GarbageDetector

# Load the latest trained model once, then call 'detect' for each image.
detector = GarbageDetector()
result = detector.detect("data/Totaal_dataset/inference_tests_visual/example_6.jpg")

# 'result.to_dict()' is the JSON-shaped Python dictionary.
print(result.to_dict())
```

By default the model is loaded from
`runs/train/yolo11s_garbage_5c-2/weights/best.pt` and the bin mapping comes
from `data/bin_mapping.json`. Crops and a JSON summary are written under
`runs/detector_inference/<image_stem>/`.

## Configuration paths

The module exposes path defaults in a clearly marked block at the top of
[`src/trash_detector/detector/garbage_detector.py`](src/trash_detector/detector/garbage_detector.py).
Edit them once to switch the model or the bin mapping for every caller, or
override per call via constructor arguments.

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_RELATIVE_PATH` | `runs/train/yolo11s_garbage_5c-2/weights/best.pt` | Trained YOLO checkpoint loaded by `GarbageDetector()`. |
| `DEFAULT_WEIGHTS_PATH` | derived from `MODEL_RELATIVE_PATH` | Absolute path passed to YOLO; rebuilt from the relative string above. |
| `DEFAULT_BIN_MAPPING_PATH` | `data/bin_mapping.json` | Material-to-bin JSON used to label detections. |
| `DEFAULT_OUTPUT_ROOT` | `runs/detector_inference` | Folder where each image's crops and JSON are written. |

The demo runner [`garbage_detector_demo.py`](garbage_detector_demo.py)
exposes its own configuration block at the top with `DEMO_IMAGE`,
`MODEL_PATH`, and `DEFAULT_OUTPUT_DIR`. `MODEL_PATH=None` falls back to
`MODEL_RELATIVE_PATH` from the module; setting it to a `Path` swaps the
checkpoint just for the demo without editing the module.

## Where it lives

| Path | Purpose |
| --- | --- |
| `src/trash_detector/detector/garbage_detector.py` | Implementation: `GarbageDetector`, `Detection`, `DetectionResult`, `detect_image`. |
| `src/trash_detector/detector/__init__.py` | Re-exports, so `from trash_detector.detector import GarbageDetector` works. |
| `garbage_detector_demo.py` | Runnable demo at the project root that uses `example_6.jpg`. |

## Public surface

### `GarbageDetector`

| Member | Type | Description |
| --- | --- | --- |
| `GarbageDetector(weights_path=None, bin_mapping_path=None, *, conf=0.25, imgsz=640, device=None)` | constructor | Loads the YOLO checkpoint and validates the bin mapping JSON. The model is loaded **once**; reuse the instance across calls. |
| `weights_path` | property | Path to the loaded weights file. |
| `class_names` | property | Mapping from numeric class id to material name. |
| `detect(image, output_dir=None, *, save_crops=True, save_json=True, margin_frac=0.05)` | method | Runs detection on one image and returns a `DetectionResult`. |

### `detect_image(...)`

A one-shot helper: builds a `GarbageDetector`, runs one detection, and returns
the `DetectionResult`. Convenient for one-off scripts; for many images in a
row, prefer creating the detector once and reusing it (the model load is
expensive).

### `Detection` (dataclass)

One detected object inside an image.

| Field | Type | Notes |
| --- | --- | --- |
| `index` | `int` | Stable index inside the parent image (0, 1, 2, ...). |
| `class_id` | `int` | Numeric YOLO class id. |
| `material` | `str` | Material label, for example `'Cardboard'`. |
| `bin` | `str` | Household bin label resolved through `data/bin_mapping.json`. |
| `confidence` | `float` | Detection confidence in 0..1. |
| `bbox_xyxy` | `list[float]` | Bounding box `[x1, y1, x2, y2]` in source-image pixels. |
| `crop_path` | `Path \| None` | Path to the saved crop JPG, or `None` when `save_crops=False`. |

### `DetectionResult` (dataclass)

The return value of `detect`. Holds all detections plus output paths.

| Field | Type | Notes |
| --- | --- | --- |
| `image_path` | `Path \| None` | Path to the source image; `None` when the input was a numpy array. |
| `image_size` | `tuple[int, int]` | `(width, height)` in pixels. |
| `model_path` | `Path` | Weights file used for inference. |
| `output_dir` | `Path \| None` | Folder where crops and JSON were written. |
| `json_path` | `Path \| None` | Path to the JSON summary file. |
| `detections` | `list[Detection]` | One entry per detected object. |
| `to_dict()` | method | JSON-shaped dictionary of the full result. |

## Inputs accepted

The `image` argument of `detect` accepts:

- A `str` or `pathlib.Path` pointing to an image file.
- A `numpy.ndarray` containing a decoded frame (BGR, as produced by
  `cv2.imread`).

When a numpy array is passed, the `DetectionResult.image_path` is `None` and
output filenames fall back to a generic `array_input` stem.

## Output JSON schema

`DetectionResult.to_dict()` (and the JSON file on disk, when
`save_json=True`) follows this structure:

```json
{
  "image": {
    "path": "data/Totaal_dataset/inference_tests_visual/example_6.jpg",
    "width": 4032,
    "height": 3024
  },
  "model_path": "runs/train/yolo11s_garbage_5c-2/weights/best.pt",
  "output_dir": "runs/detector_inference/example_6",
  "json_path": "runs/detector_inference/example_6/example_6_detections.json",
  "num_detections": 2,
  "detections": [
    {
      "index": 0,
      "class_id": 0,
      "material": "Cardboard",
      "bin": "Paper",
      "confidence": 0.87,
      "bbox_xyxy": [123.4, 56.7, 800.2, 950.1],
      "crop_path": "runs/detector_inference/example_6/example_6_det00_Cardboard.jpg"
    },
    {
      "index": 1,
      "class_id": 4,
      "material": "Plastic",
      "bin": "PMD",
      "confidence": 0.61,
      "bbox_xyxy": [1500.0, 200.0, 2400.0, 1800.0],
      "crop_path": "runs/detector_inference/example_6/example_6_det01_Plastic.jpg"
    }
  ]
}
```

Notes:

- `bin` comes from `data/bin_mapping.json`. Unknown materials fall back to the
  `default_bin` key in that file (`'Rest'`).
- An image with no detections still returns a valid result with
  `num_detections: 0` and an empty `detections` array.

## Output files on disk

Given an image at `data/.../example_6.jpg` and the default settings, the
detector creates:

```text
runs/detector_inference/example_6/
├── example_6_det00_Cardboard.jpg
├── example_6_det01_Plastic.jpg
└── example_6_detections.json
```

Each crop is taken from the source image, expanded by `margin_frac` (default
5 percent on each side) and clipped to the image bounds. JPGs are written via
OpenCV in BGR order (which is the natural format from `cv2.imread`).

## Running the demo

From the project root:

```bash
uv run python garbage_detector_demo.py
```

The demo:

1. Loads the detector with default paths.
2. Runs detection on
   `data/Totaal_dataset/inference_tests_visual/example_6.jpg`.
3. Prints a short summary, the per-detection lines, and the full JSON
   dictionary.
4. Leaves the saved crops and JSON file under
   `runs/detector_inference/example_6/`.

## Integration tips for the NLP side

- Build **one** `GarbageDetector` per process and call `detect` for every
  image. The YOLO checkpoint is large; loading it many times is wasteful.
- Use `result.to_dict()` directly if you want a JSON-friendly Python object,
  or read the saved JSON file with `json.load`. Both contain the same payload.
- If the NLP code only needs the labels (no files on disk), pass
  `save_crops=False` and `save_json=False`. Detections are still returned in
  memory.
- If the NLP code wants its own folder layout, pass an explicit `output_dir`.
- The `device` argument follows Ultralytics conventions (`'cpu'`, `'0'`,
  `'0,1'`, ...). `None` lets Ultralytics auto-select.

## Configuration knobs

| Knob | Default | Effect |
| --- | --- | --- |
| `weights_path` | `runs/train/yolo11s_garbage_5c-2/weights/best.pt` | Which trained checkpoint to use. |
| `bin_mapping_path` | `data/bin_mapping.json` | Which material-to-bin map to use. |
| `conf` | `0.25` | Minimum YOLO confidence kept per detection. |
| `imgsz` | `640` | Inference image size used by YOLO. |
| `device` | `None` | Compute device. `None` means auto. |
| `save_crops` | `True` | When `True`, write a crop JPG per detection. |
| `save_json` | `True` | When `True`, write the JSON summary file. |
| `margin_frac` | `0.05` | Extra padding around each box before cropping, as a fraction of the box size. |
