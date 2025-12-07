import os
import cv2                   #py -3.11 features_generator.py
import numpy as np
import joblib
from feature_extractor import FeatureExtractor
from sklearn.preprocessing import StandardScaler

DATA_DIR = "dataset_complete"
OUTPUT_FEATURES = "features.npy"
OUTPUT_LABELS = "labels.npy"
OUTPUT_SCALER = "feature_scaler.pkl"

extractor = FeatureExtractor()
X = []   # Features
y = []   # Labels

# Map folder names to class IDs
labels_map = {
    "glass": 0,
    "paper": 1,
    "cardboard": 2,
    "plastic": 3,
    "metal": 4,
    "trash": 5,
    "unknown": 6
}

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
        feature_vector = extractor.extract_features(image)
        X.append(feature_vector)
        y.append(labels_map[category])

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    print("No features extracted")
    raise SystemExit(1)

print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

np.save(OUTPUT_FEATURES, X_scaled)
np.save(OUTPUT_LABELS, y)
joblib.dump(scaler, OUTPUT_SCALER)

print("Feature extraction done!")
print("X shape:", X_scaled.shape)
print("y shape:", y.shape)
