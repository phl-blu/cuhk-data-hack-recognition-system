# image_feature_utils.py

import os
import cv2
import numpy as np
import joblib
from feature_extractor import FeatureExtractor
from configures import SCALER_FILE

# Image utilities
def load_image_rgb(image_path):
    """Load image from disk and convert to RGB."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Feature extraction utilities
def extract_feature_from_image(image, extractor=None):
    """Extract deep feature vector from a single RGB image."""
    if extractor is None:
        extractor = FeatureExtractor()
    return extractor.extract_features(image)


def extract_features_from_directory(
    images_dir,
    label,
    valid_exts=(".jpg", ".jpeg", ".png")
):
    """
    Extract features from all images in a directory.
    Returns X, y.
    """
    extractor = FeatureExtractor()
    X, y = [], []

    for fname in os.listdir(images_dir):
        if not fname.lower().endswith(valid_exts):
            continue

        fpath = os.path.join(images_dir, fname)
        img = load_image_rgb(fpath)
        if img is None:
            continue

        fv = extractor.extract_features(img)
        X.append(fv)
        y.append(label)

    return np.array(X), np.array(y)


def extract_dataset_features_from_folders(base_dir, labels_map):
    """
    Extract features from a dataset structured as:
    base_dir/class_name/*.jpg
    """
    extractor = FeatureExtractor()
    X, y = [], []

    for class_name, class_id in labels_map.items():
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        print(f"Processing: {class_name}")

        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            img = load_image_rgb(fpath)
            if img is None:
                continue

            fv = extractor.extract_features(img)
            X.append(fv)
            y.append(class_id)

    X = np.array(X)
    y = np.array(y)

    # Shuffle
    idx = np.arange(len(X))
    np.random.shuffle(idx)

    return X[idx], y[idx]


# Scaling utilities
def scale_features(X, scaler_path=SCALER_FILE, fit=False):
    """
    Scale features using StandardScaler.
    If fit=True, fit and save scaler.
    """
    if fit:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, scaler_path)
        return X_scaled
    else:
        scaler = joblib.load(scaler_path)
        return scaler.transform(X)
