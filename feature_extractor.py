import os
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model

class FeatureExtractor:
    def __init__(self):
        # Load ResNet50 without the top layer, with global average pooling
        self.model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
        print("Loaded ResNet50 for feature extraction.")

    def extract_features(self, image):
        # Resize and preprocess image
        img = cv2.resize(image, (224, 224))
        img = preprocess_input(img.astype(np.float32))
        img = np.expand_dims(img, axis=0)  # batch dimension
        features = self.model.predict(img, verbose=0)[0]
        return features
