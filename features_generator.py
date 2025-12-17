import os
import cv2
import numpy as np
import joblib
from feature_extractor import FeatureExtractor 
from sklearn.preprocessing import StandardScaler
from configures import *


# Set random seed for reproducibility
np.random.seed(42)

extractor = FeatureExtractor()

labels_map = {
    "glass": 0,
    "paper": 1,
    "cardboard": 2,
    "plastic": 3,
    "metal": 4,
    "trash": 5,
    "unknown": 6
}

def extract_dataset_features(DATA_DIR):
    X = []
    y = []
    for category in labels_map:
        folder = os.path.join(DATA_DIR, category)
        if not os.path.isdir(folder):
            continue

        print(f"Processing: {category}")
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            image = cv2.imread(filepath)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Use new deep feature extractor
            feature_vector = extractor.extract_features(image)
            X.append(feature_vector)
            y.append(labels_map[category])

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        raise SystemExit("No features extracted")

    # Shuffle dataset
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    return X, y

# Extract Train Features 
print("\n=== Extracting TRAIN features ===")
X_train, y_train = extract_dataset_features(OUTPUT_DIR)

# Scale only on TRAIN set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, SCALER_FILE)

np.save(TRAIN_FEATURES, X_train_scaled)
np.save(TRAIN_LABELS, y_train)

print("Saved:", TRAIN_FEATURES, TRAIN_LABELS)
print("Train shape:", X_train_scaled.shape)

# Extract Test Features
print("\n=== Extracting TEST features ===")
X_test, y_test = extract_dataset_features(TEST_DIR)

scaler = joblib.load(SCALER_FILE)
X_test_scaled = scaler.transform(X_test)

np.save(TEST_FEATURES, X_test_scaled)
np.save(TEST_LABELS, y_test)

print("Saved:", TEST_FEATURES, TEST_LABELS)
print("Test shape:", X_test_scaled.shape)

print("\nFeature extraction complete!")
