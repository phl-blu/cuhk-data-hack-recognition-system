import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from FeatureLoader import *
import joblib

# Label mapping
labels_map = {
    "glass": 0,
    "paper": 1,
    "cardboard": 2,
    "plastic": 3,
    "metal": 4,
    "trash": 5,
    "unknown": 6
}

# Load features
loader = FeatureLoader()
X, y = loader.load()

### --- MODIFICATION 1: L2-Feature Normalization ---
# Normalize features to unit length. This makes Euclidean distance a measure of angular similarity.
X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)

# Dimensionality reduction
pca = PCA(n_components=512, random_state=42)
X_reduced = pca.fit_transform(X_normalized) # Use normalized features

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42, stratify=y)

# KNN classifier
k = 3
model = KNeighborsClassifier(
    n_neighbors=k,
    metric='euclidean',
    weights='distance',
)
model.fit(X_train, y_train)

# Compute distances for training and test samples
distances_train, _ = model.kneighbors(X_train, n_neighbors=k)
distances_test, _ = model.kneighbors(X_test, n_neighbors=k)

### --- MODIFICATION 2: Use K-th Nearest Neighbor Distance for OOD Score ---
# The OOD score is now the distance to the K-th (3rd) nearest neighbor.
# distances_train[:, k-1] selects the column corresponding to the k-th neighbor distance.
kth_train_distance = distances_train[:, k-1]
kth_test_distance = distances_test[:, k-1]

# Calculate stats on the new OOD score (k-th distance)
mean_train = kth_train_distance.mean()
std_train = kth_train_distance.std()

print(f"Loaded deep features: {X.shape} labels: {y.shape}")
print(f"K-th Distance Stats (K={k}, D=512): Mean={mean_train:.4f}, Std={std_train:.4f}")

# Try multiple thresholds: mean + F*std
print("\nResults with L2-Normalization and K-th Distance Score:")
print("-" * 60)
for factor in [2, 2.5, 3, 3.5, 4, 4.5, 5]:
    threshold = mean_train + factor * std_train
    y_pred = []
    num_unknowns = 0
    
    # Iterate over the K-th distances of the test set
    for i, dist in enumerate(kth_test_distance):
        # OOD Detection: If the k-th distance is too large
        if dist > threshold:
            y_pred.append(labels_map['unknown'])
            num_unknowns += 1
        else:
            # Classification: If it's In-Distribution, predict its class
            y_pred.append(model.predict(X_test[i].reshape(1, -1))[0])
    
    # Calculate accuracy only on the test set (y_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Factor={factor} | Threshold: {threshold:.4f} | Test Accuracy: {acc:.4f} | Unknowns: {num_unknowns}")
# Save the trained model
joblib.dump(model, "knn_model.pkl")
print("\nSaved model to:", "knn_model.pkl")