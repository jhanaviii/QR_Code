import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def extract_basic_features(img):
    """Extract basic image statistics"""
    return [
        np.mean(img),        # Average intensity
        np.std(img),         # Contrast
        np.median(img),      # Median value
        np.max(img) - np.min(img)  # Dynamic range
    ]

def extract_texture_features(img, radius=3, n_points=24):
    """Extract texture features using LBP"""
    lbp = local_binary_pattern(img, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp, bins=n_points+2, range=(0, n_points+2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # Normalize
    return hist

def extract_edge_features(img):
    """Extract edge-related features"""
    edges = cv2.Canny(img, 100, 200)
    return [
        np.sum(edges) / img.size,  # Edge density
        np.mean(edges),            # Average edge strength
        len(np.where(edges > 0)[0] / img.size)  # Edge pixel ratio
    ]


def extract_all_features(images):
    """Extract all features for a list of images"""
    if len(images) == 0:
        raise ValueError("No images provided for feature extraction")

    features = []
    for img in images:
        feature_vec = []
        feature_vec.extend(extract_basic_features(img))
        feature_vec.extend(extract_texture_features(img))
        feature_vec.extend(extract_edge_features(img))
        features.append(feature_vec)

    print(f"Extracted features for {len(features)} images")
    return np.array(features)