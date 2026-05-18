"""
Vision thread for the demo pipeline.

Provides 'run_vision_thread()', which is the entry point for the vision daemon
thread started by 'demo.py'. The thread continuously:
  1. Captures a frame from the webcam.
  2. Crops and resizes it to 640x640 pixels.
  3. Runs YOLO inference.
  4. Writes 'output.png' (annotated frame) and 'results.json' (detected labels)
     to the shared 'temp/' directory.

When the rolling prediction window is enabled, the displayed class label and
confidence are smoothed across the last N frames (the most frequent class wins
and its confidence is averaged). The bounding box geometry is always taken
from the current frame, since the box positions are already stable.

The NLP thread reads 'results.json' whenever the user types 'start'.
"""

import json
import threading
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


# ---------------------------------------------------------------
# Rolling prediction window
# ---------------------------------------------------------------

class RollingPredictionWindow:
    """
    Keeps the last 'size' frames' detections to smooth out flickering predictions.

    Each entry in the window is the list of (class_id, confidence) pairs detected
    in one frame. 'get_smoothed()' returns the most frequent class across all
    entries together with that class's average confidence.
    """

    def __init__(self, size: int) -> None:
        # 'maxlen' makes the deque automatically drop the oldest frame
        # when a new one is appended once the window is full.
        self._frames: deque[list[tuple[int, float]]] = deque(maxlen=size)

    def add_frame(self, detections: list[tuple[int, float]]) -> None:
        """Append the current frame's detections to the window."""
        # Storing an empty list for frames with no detections is intentional:
        # the empty frame still occupies a slot and pushes old data out.
        self._frames.append(detections)

    def get_smoothed(self) -> tuple[int, float] | None:
        """
        Return (class_id, average_confidence) for the most frequent class in the
        window, or None if the window contains no detections at all.
        """
        # Count how often each class appears and collect its confidences.
        counts: Counter[int] = Counter()
        confs: dict[int, list[float]] = defaultdict(list)
        for frame in self._frames:
            for cls_id, conf in frame:
                counts[cls_id] += 1
                confs[cls_id].append(conf)

        # No detections have been observed in the entire window yet.
        if not counts:
            return None

        # Pick the most frequent class. Counter.most_common breaks ties by
        # insertion order, which is good enough here.
        top_cls = counts.most_common(1)[0][0]
        avg_conf = sum(confs[top_cls]) / len(confs[top_cls])
        return top_cls, avg_conf


# ---------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------

def _draw_smoothed_boxes(
    frame,
    boxes,
    cls_id: int,
    conf: float,
    names: dict[int, str],
):
    """
    Draw the current frame's bounding boxes but label them with the smoothed
    'cls_id' and 'conf' instead of each box's own raw prediction.

    Uses ultralytics' Annotator so the resulting overlay matches the style
    produced by 'results.plot()'.
    """
    # Annotator works on a copy so the raw frame stays untouched.
    annotator = Annotator(frame.copy(), line_width=2)
    label = f"{names[int(cls_id)]} {conf:.2f}"

    # Reuse the same colour as 'results.plot()' would have used for this class.
    box_color = colors(int(cls_id), True)

    # Each box keeps its own geometry; only the label is overridden.
    for box in boxes.xyxy.tolist():
        annotator.box_label(box, label, color=box_color)

    return annotator.result()


# ---------------------------------------------------------------
# Detection filtering
# ---------------------------------------------------------------

def _filter_oversized_boxes(results, frame_h: int, frame_w: int, max_fraction: float):
    """
    Drop bounding boxes that cover almost the entire frame.

    A box is removed when both its width and its height exceed
    'max_fraction * frame_side'. This targets the case where YOLO produces a
    full-frame 'phantom' detection (for example, calling an empty desk
    'cardboard') without filtering out legitimate long, thin objects.

    The filtering is applied in place by replacing 'results.boxes' with the
    kept subset, so that subsequent code (including 'results.plot()') sees
    only the surviving detections.
    """
    # Disabled when the threshold is at or above 1.0; nothing to filter.
    if max_fraction >= 1.0:
        return

    # Nothing detected on this frame, so there is nothing to filter either.
    if len(results.boxes) == 0:
        return

    # Compute per-box width and height directly on the underlying tensor.
    xyxy = results.boxes.xyxy
    box_widths = xyxy[:, 2] - xyxy[:, 0]
    box_heights = xyxy[:, 3] - xyxy[:, 1]

    # Pixel thresholds derived from the frame dimensions.
    max_w = max_fraction * frame_w
    max_h = max_fraction * frame_h

    # Boolean mask: True for boxes that should be kept (not oversized).
    keep_mask = ~((box_widths > max_w) & (box_heights > max_h))

    # Replace the boxes container with the filtered subset. Ultralytics'
    # 'Boxes' class supports boolean-mask indexing via its BaseTensor parent.
    results.boxes = results.boxes[keep_mask]


# ---------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------

def _infer_and_save(
    frame,
    model: YOLO,
    output_dir: Path,
    window: RollingPredictionWindow | None,
    max_bbox_fraction: float,
) -> None:
    """
    Run YOLO inference on one frame and write results to 'output_dir'.

    Writes two files:
      - 'output.png': the input frame with bounding boxes drawn on it.
      - 'results.json': a JSON object with the detected material label names.

    The JSON format is:
      {
        "labels": ["Metal", "Glass"],
        "image": "<path to output.png>"
      }

    If 'window' is provided AND the current frame has at most one detection,
    the rolling window smoothing is applied to the displayed class and
    confidence. With multiple detections in view, the window cannot tell the
    objects apart and would label every box with the same most-frequent class,
    so the raw per-frame YOLO output is used instead.
    """
    # Run YOLO on the in-memory frame. Avoids an extra disk round-trip.
    results = model(frame, save=False, save_txt=False, verbose=False)[0]

    # Drop bounding boxes that cover almost the entire frame. This must happen
    # before the detection list is built so that oversized boxes never reach
    # the rolling window or the JSON output.
    h, w = frame.shape[:2]
    _filter_oversized_boxes(results, h, w, max_bbox_fraction)
    boxes = results.boxes

    # Pull per-detection (class_id, confidence) pairs from the result tensors.
    current_detections: list[tuple[int, float]] = [
        (int(cls), float(conf))
        for cls, conf in zip(boxes.cls.tolist(), boxes.conf.tolist())
    ]

    output_image_path = output_dir / "output.png"

    # ---- Branch 1: rolling window enabled AND at most one object in view. ----
    # The window holds a single class history per object, so it can only smooth
    # a one-object scene. With multiple objects we fall through to Branch 2 so
    # each box keeps its own per-frame class instead of all sharing the most
    # frequent class in the window.
    if window is not None and len(current_detections) <= 1:
        # Update the window with this frame, then read back the smoothed result.
        window.add_frame(current_detections)
        smoothed = window.get_smoothed()

        # If the entire window is empty, there is nothing to draw or report.
        if smoothed is None or len(current_detections) == 0:
            # Save the raw frame so 'output.png' keeps refreshing even when
            # no objects are detected.
            cv2.imwrite(str(output_image_path), frame)
            labels: list[str] = []
        else:
            cls_id, avg_conf = smoothed
            annotated = _draw_smoothed_boxes(frame, boxes, cls_id, avg_conf, results.names)
            cv2.imwrite(str(output_image_path), annotated)
            # The NLP thread only needs the smoothed class name.
            labels = [results.names[int(cls_id)]]

    # ---- Branch 2: rolling window disabled OR multiple objects in view. ----
    else:
        annotated_frame = results.plot()
        cv2.imwrite(str(output_image_path), annotated_frame)
        # Collect unique detected class names from all bounding boxes.
        labels = list({results.names[int(cid)] for cid, _ in current_detections})

    # Write the label list to JSON so the NLP thread can read it.
    with open(output_dir / "results.json", "w") as f:
        json.dump({"labels": labels, "image": str(output_image_path)}, f, indent=4)


# ---------------------------------------------------------------
# Demo thread entry point
# ---------------------------------------------------------------

def run_vision_thread(
    model: YOLO,
    temp_dir: Path,
    stop_event: threading.Event,
    camera_index: int = 0,
    use_rolling_window: bool = False,
    rolling_window_size: int = 20,
    max_bbox_size: float = 1.0,
) -> None:
    """
    Run the vision capture-and-inference loop as a daemon thread.

    Captures frames from the webcam at 'camera_index' continuously until
    'stop_event' is set (triggered by the ESC key in 'demo.py').

    Each frame is centre-cropped to a square and resized to 640x640 before
    being passed to YOLO, matching the model's training resolution.

    When 'use_rolling_window' is True, the displayed class label and confidence
    are smoothed over the last 'rolling_window_size' frames. The bounding box
    positions are never smoothed.

    'max_bbox_size' is a fraction in (0, 1]. Any detection whose width and
    height both exceed that fraction of the frame is dropped before the
    rolling window or the JSON output sees it. A value of 1.0 disables the filter.
    """
    # Short pause to allow other threads to initialise first.
    time.sleep(2)

    # Build the rolling window up-front (or leave it disabled). A size of zero
    # or less would make the deque useless, so it is treated as 'disabled'.
    window: RollingPredictionWindow | None
    if use_rolling_window and rolling_window_size > 0:
        window = RollingPredictionWindow(size=rolling_window_size)
        print(f"Vision: rolling window enabled (size={rolling_window_size}).")
    else:
        window = None

    # Log the bbox-size filter so the user can confirm the configured threshold.
    if max_bbox_size < 1.0:
        print(f"Vision: dropping boxes larger than {max_bbox_size:.2f} of the frame.")

    # Pass cv2.CAP_ANY explicitly so the index->device mapping matches what the
    # 'any' backend produced during probing (0 = USB cam, 1 = internal webcam).
    # Single-arg cv2.VideoCapture(index) can pick a different driver on Windows.
    cap = cv2.VideoCapture(camera_index, cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam (index {camera_index}). "
            "Ensure a webcam is connected and not in use by another application."
        )

    while not stop_event.is_set():
        # Capture one frame from the webcam.
        ret, frame = cap.read()
        if not ret:
            print("Vision: failed to capture frame, retrying...")
            continue

        # Centre-crop the frame to a square, then scale to 640x640.
        h, w = frame.shape[:2]
        crop_side = min(h, w)
        x0 = (w - crop_side) // 2
        y0 = (h - crop_side) // 2
        frame = frame[y0: y0 + crop_side, x0: x0 + crop_side]
        frame = cv2.resize(frame, (640, 640))

        # Save the raw frame for debugging and so other tools can inspect it.
        temp_image_path = temp_dir / "input.png"
        cv2.imwrite(str(temp_image_path), frame)

        # Run inference and write annotated output + results to the temp dir.
        try:
            _infer_and_save(frame, model, temp_dir, window, max_bbox_size)
        except Exception as e:
            print(f"Vision: inference error: {e}")

    # Release the webcam when the stop event fires.
    cap.release()
