"""
Vision thread for the demo pipeline.

Provides 'run_vision_thread()', which is the entry point for the vision daemon
thread started by 'demo.py'. The thread continuously:
  1. Captures a frame from the webcam.
  2. Crops and resizes it to 640x640 pixels.
  3. Runs YOLO inference.
  4. Writes 'output.png' (annotated frame) and 'results.json' (detected labels)
     to the shared 'temp/' directory.

The NLP thread reads 'results.json' whenever the user types 'start'.
"""

import json
import threading
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


# ---------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------

def _infer_and_save(input_image_path: Path, model: YOLO, output_dir: Path) -> None:
    """
    Run YOLO inference on one image and write results to 'output_dir'.

    Writes two files:
      - 'output.png': the input frame with bounding boxes drawn on it.
      - 'results.json': a JSON object with the detected material label names.

    The JSON format is:
      {
        "labels": ["Metal", "Glass"],
        "image": "<path to output.png>"
      }
    """
    results = model(input_image_path, save=False, save_txt=False, verbose=False)[0]

    # Collect unique detected class names from all bounding boxes.
    class_ids = results.boxes.cls.tolist()
    labels = list({results.names[int(cid)] for cid in class_ids})

    # Draw bounding boxes on the frame and save the annotated image.
    annotated_frame = results.plot()
    output_image_path = output_dir / "output.png"
    cv2.imwrite(str(output_image_path), annotated_frame)

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
) -> None:
    """
    Run the vision capture-and-inference loop as a daemon thread.

    Captures frames from webcam index 0 continuously until 'stop_event'
    is set (triggered by the ESC key in 'demo.py').

    Each frame is centre-cropped to a square and resized to 640x640 before
    being passed to YOLO, matching the model's training resolution.
    """
    # Short pause to allow other threads to initialise first.
    time.sleep(2)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam (index 0). "
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

        # Save the raw frame so the inference function can read it as a file.
        temp_image_path = temp_dir / "input.png"
        cv2.imwrite(str(temp_image_path), frame)

        # Run inference and write results to the shared temp directory.
        try:
            _infer_and_save(temp_image_path, model, temp_dir)
        except Exception as e:
            print(f"Vision: inference error: {e}")

    # Release the webcam when the stop event fires.
    cap.release()
