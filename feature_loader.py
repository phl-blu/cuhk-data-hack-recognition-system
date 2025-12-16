import os 
import numpy as np

class FeatureLoader:
    def __init__(self, features_file="features_train.npy", labels_file="labels_train.npy"):
        self.features_file = features_file
        self.labels_file = labels_file
    
    def load(self):
        """Load the deep features and labels."""

        if not os.path.exists(self.features_file) or not os.path.exists(self.labels_file):
            raise SystemExit("Missing feature files. Run feature_extractor.py first.")

        X = np.load(self.features_file)
        y = np.load(self.labels_file)

        print(f"Loaded features: {X.shape} labels: {y.shape}")
        return X, y
