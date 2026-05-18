# vision/

Webcam capture and YOLO inference module.

## Modules

| Module | Purpose |
|--------|---------|
| `vision.py` | `run_vision_thread(model, temp_dir, stop_event, camera_index, use_rolling_window, rolling_window_size, max_bbox_size)` -- the thread entry point. Captures frames from the configured webcam, centre-crops and resizes to 640x640, runs YOLO inference, and writes results to `temp/`. |

## Rolling prediction window

When `vision.use_rolling_window` is set to `true` in `config/demo_config.yaml`,
the displayed class label and confidence are smoothed across the last
`vision.rolling_window_size` frames:

- The most frequent class id across the window is selected.
- That class's confidences are averaged.
- All current-frame bounding boxes are drawn at their detected positions but
  re-labeled with the smoothed class and average confidence.
- The bounding box geometry is not smoothed because positions are usually stable.

This removes flicker between visually similar classes (for example 'Plastic'
vs 'Metal') without freezing the box location.

When `use_rolling_window` is `false`, the raw per-frame YOLO output is shown.

## Maximum bounding box size filter

`vision.max_bbox_size` (a fraction in `(0, 1]`) drops detections whose
bounding box is too large relative to the frame. A box is removed when
both its width and its height exceed `max_bbox_size * 640`. Set this to
`1.0` to disable the filter.

This is intended to suppress 'phantom' full-frame detections, where YOLO
labels the entire empty scene as some material. The filter is applied
before the rolling window so phantom boxes do not poison the smoothing.

## Temp file outputs

Every inference cycle writes two files to `temp/`:

| File | Contents |
|------|---------|
| `input.png` | The raw 640x640 cropped webcam frame. |
| `output.png` | The same frame with YOLO bounding boxes drawn on it. |
| `results.json` | `{"labels": ["Metal", "Glass"], "image": "<path to output.png>"}` |

When the rolling window is enabled, `results.json` reports the single
smoothed class name (or an empty list if no objects are currently in view).

The NLP thread reads `results.json` when the user types `start`.
