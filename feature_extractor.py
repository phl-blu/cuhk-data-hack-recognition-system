import cv2
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import backend as K

class FeatureExtractor:
    def __init__(self):
        # Clear previous TF session to avoid conflicts
        K.clear_session()

        # Force input shape to 3 channels for pretrained weights
        self.model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg", input_shape=(224,224,3))
        print("Loaded EfficientNetB0 for multi-scale feature extraction.")

    def extract_features(self, image):
        scales = [224, 256, 288]
        features_list = []

        for size in scales:
            img = cv2.resize(image, (size, size))
            # Convert grayscale to RGB if needed
            if len(img.shape) == 2 or img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            img = preprocess_input(img.astype(np.float32))
            img = np.expand_dims(img, axis=0)
            feat = self.model.predict(img, verbose=0)[0]
            features_list.append(feat)

        features = np.concatenate(features_list)
        return features
