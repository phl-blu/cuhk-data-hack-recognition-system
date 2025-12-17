import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

class FeatureExtractor:
    def __init__(self):
        self.model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
        print("Loaded ResNet50 for multi-scale feature extraction.")

    def extract_features(self, image):
        # Extract at multiple scales
        scales = [224, 256, 288]  # Different input sizes
        features_list = []
        
        for size in scales:
            img = cv2.resize(image, (size, size))
            img = preprocess_input(img.astype(np.float32))
            img = np.expand_dims(img, axis=0)
            feat = self.model.predict(img, verbose=0)[0]
            features_list.append(feat)
        
        # Concatenate multi-scale features
        features = np.concatenate(features_list)
        return features