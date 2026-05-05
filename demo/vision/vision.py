import json
import cv2
import time
from pathlib import Path
from ultralytics import YOLO

import config

def _vision_infer_single_image(input_image_path: Path, model: YOLO, output_dir: Path) -> None:
    """
    Run YOLO inference on a single image and save results.
    
    Args:
        input_image_path (Path): Path to the input image.
        model (YOLO): The loaded YOLO model.
        output_dir (Path): Directory where output image and JSON will be saved. Outputs are saved as 'output.png' and 'results.json' within this directory.
            results.json has the following format:
            {
                "labels": [list of detected material names],
                "image": "path to the saved output image with bounding boxes"
            }
        
    Returns:
        None
    """
    res = model(input_image_path, save=False, save_txt=False, save_conf=False, verbose=False)[0]

    ids = res.boxes.cls.tolist()
    names = res.names
    labels = set(names[int(cls_id)] for cls_id in ids)
    labels = list(labels)

    img = res.plot()
    cv2.imwrite(str(output_dir / "output.png"), img)

    with open(output_dir / "results.json", "w") as f:
        json.dump({"labels": labels, "image": str(output_dir / "output.png")}, f, indent=4)
    return

def run_vision() -> None:
    """
    Run the vision part:
        - Capture a frame from the webcam
        - Run YOLO inference on the captured frame
    This function is meant to be called as a deamon that continuously processes webcam input.
    """
    time.sleep(2)  # Wait a bit to ensure all threads are up and running

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    while config.IS_RUNNING:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam, retrying...")
            continue

        # crop / expand the frame to 640x640
        height, width, _ = frame.shape
        crop_size = min(height, width)
        start_x = (width - crop_size) // 2
        start_y = (height - crop_size) // 2
        frame = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]
        frame = cv2.resize(frame, (640, 640))
        
        # Save the captured frame to a temporary file
        temp_image_path = config.TEMP_DIR / "input.png"
        cv2.imwrite(temp_image_path, frame)

        # Run YOLO inference on the captured frame
        try:
            _vision_infer_single_image(temp_image_path, config.VISION_MODEL, config.TEMP_DIR)
        except Exception as e:
            print(f"Error during vision inference: {e}")
            continue