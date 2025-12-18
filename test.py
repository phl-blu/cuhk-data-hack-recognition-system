import os
import cv2
import joblib
import numpy as np

from configures import SCALER_FILE
from feature_extractor import FeatureExtractor

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _list_image_files(folder_path):
    files = []
    for name in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(name)[1].lower()
        if ext in ALLOWED_EXTENSIONS:
            files.append(os.path.join(folder_path, name))
    return files


def predict(dataFilePath, bestModelPath):
    if not os.path.isdir(dataFilePath):
        raise NotADirectoryError(f"Expected a folder of images, got: {dataFilePath}")

    image_paths = _list_image_files(dataFilePath)
    if not image_paths:
        return []

    # load trained KNN model
    model = joblib.load(bestModelPath)

    # Re-attach extractor or scaler if they were not serialized with the model
    if getattr(model, "extractor", None) is None:
        model.extractor = FeatureExtractor()
    if getattr(model, "scaler", None) is None:
        if not os.path.exists(SCALER_FILE):
            raise FileNotFoundError(f"Scaler file not found: {SCALER_FILE}")
        model.scaler = joblib.load(SCALER_FILE)

    predictions = []
    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        feats = model.extractor.extract_features(image)
        feats_scaled = model.scaler.transform(feats.reshape(1, -1))

        pred, _ = model.predict(feats_scaled)
        predictions.append(int(pred[0]))

    return predictions


if __name__ == "__main__":
    sample_folder = "unknown"
    sample_model = "knn_model.pkl"
    print(predict(sample_folder, sample_model))
