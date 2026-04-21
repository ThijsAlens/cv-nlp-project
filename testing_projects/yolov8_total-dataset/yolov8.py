from ultralytics import YOLO

import os
import sys

import config

def train_model() -> None:
    """
    Trains the YOLO model on the dataset specified in the 'data.yaml' file.

    Args:
        None

    Returns:
        None

    """
    model = YOLO('yolov8n.pt')  # n is the smallest model

    results = model.train(
        data='dataset/data.yaml',
        epochs=50,
        imgsz=640, # (640x640)
        plots=True
    )

def test_model(model, test_folder_path) -> None:
    """
    Goes through all images in the test folder and runs the model on them.
    It saves the results in the '/tests' folder.

    Args:
        model: The trained YOLO model.
        test_folder_path: The path to the folder containing test images.

    Returns:
        None

    """
    for filename in os.listdir(test_folder_path):
        if filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(test_folder_path, filename)
            model(image_path, save=True, project='tests', name='results', exist_ok=True)
    return

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python yolov8.py [train/test]")
        exit(1)
    if sys.argv[1] == 'train':
        train_model()
    if sys.argv[1] == 'test':
        try:
            model = YOLO(config.PATH_TO_BEST_MODEL)
        except Exception as e:
            print(f"Error occurred while loading the model: {e}")
            exit(1)
        test_model(model, config.TEST_DIR_PATH)