# vision/

Webcam capture and YOLO inference module.

## Modules

| Module | Purpose |
|--------|---------|
| `vision.py` | `run_vision_thread(model, temp_dir, stop_event)` -- the thread entry point. Captures frames from webcam index 0, centre-crops and resizes to 640x640, runs YOLO inference, and writes results to `temp/`. |

## Temp file outputs

Every inference cycle writes two files to `temp/`:

| File | Contents |
|------|---------|
| `input.png` | The raw 640x640 cropped webcam frame. |
| `output.png` | The same frame with YOLO bounding boxes drawn on it. |
| `results.json` | `{"labels": ["Metal", "Glass"], "image": "<path to output.png>"}` |

The NLP thread reads `results.json` when the user types `start`.
