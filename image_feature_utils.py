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
