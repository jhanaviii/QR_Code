import cv2
import numpy as np
from pathlib import Path
from config import FIRST_PRINTS_DIR, SECOND_PRINTS_DIR, IMG_SIZE


def load_images():
    """Load and preprocess all QR code images"""
    originals, counterfeits = [], []

    # Load originals
    for img_path in FIRST_PRINTS_DIR.glob("*.*"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            originals.append(img)

    # Load counterfeits
    for img_path in SECOND_PRINTS_DIR.glob("*.*"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            counterfeits.append(img)

    return originals, counterfeits


def preprocess_images(images):
    """Resize and normalize images for CNN"""
    processed = []
    for img in images:
        # Resize and normalize
        img_resized = cv2.resize(img, IMG_SIZE)
        img_normalized = img_resized.astype('float32') / 255.0
        processed.append(img_normalized)

    return np.array(processed)