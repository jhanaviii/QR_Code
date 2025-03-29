import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.resolve()

# Data paths - UPDATE THESE TO MATCH YOUR ACTUAL FOLDER NAMES
DATA_DIR = BASE_DIR / "data"
FIRST_PRINTS_DIR = DATA_DIR / "First Print"  # Note: Space in folder name
SECOND_PRINTS_DIR = DATA_DIR / "Second Print"  # Note: Space in folder name

# Model paths
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
ML_MODEL_PATH = MODEL_DIR / "random_forest.pkl"
CNN_MODEL_PATH = MODEL_DIR / "cnn_model.h5"

# Image parameters
IMG_SIZE = (128, 128)  # Standard size for CNN