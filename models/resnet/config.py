import torch

DATASET_PATH = "path/to/dataset"
MODEL_SAVE_PATH = "path/to/save/model.pth"

NUMBER_OF_CLASSES = 10

LR = 0.001
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"