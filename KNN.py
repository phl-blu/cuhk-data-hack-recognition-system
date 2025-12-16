import os
import numpy as np
import cv2
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from feature_extractor import FeatureExtractor
from feature_loader import FeatureLoader
from configures import *

# Label mappings
labels_map = {
    "glass": 0, "paper": 1, "cardboard": 2,
    "plastic": 3, "metal": 4, "trash": 5, "unknown": 6
}
labels_map_rev = {v: k for k, v in labels_map.items()}


class KNNClassifier:
    def __init__(self, k=3, metric='cosine', weight='distance', pca_components=0.95,
                 mahal_threshold=32.5):
        self.k = k
        self.metric = metric
        self.weight = weight
        self.pca_components = pca_components
        self.mahal_threshold = mahal_threshold  # Mahalanobis distance threshold

        self.model = KNeighborsClassifier(n_neighbors=self.k, metric=self.metric, weights=self.weight)
        self.pca = PCA(n_components=self.pca_components, random_state=42) if self.pca_components else None
        self.scaler = joblib.load(SCALER_FILE)
        self.mean_ = None
        self.inv_cov_ = None

    def fit(self, X_train, y_train):
        # L2-normalize so cosine distance and Mahalanobis operate on unit-length vectors
        X_train = normalize(X_train, norm='l2')
        if self.pca:
            X_train_pca = self.pca.fit_transform(X_train)
        else:
            X_train_pca = X_train
        self.X_train_pca = X_train_pca
        self.y_train = y_train

        # Fit KNN
        self.model.fit(X_train_pca, y_train)

        # Prepare Mahalanobis statistics in the same space
        # Add small regularization to covariance for numerical stability
        cov = np.cov(X_train_pca, rowvar=False) + np.eye(X_train_pca.shape[1]) * 1e-6
        self.inv_cov_ = np.linalg.inv(cov)
        self.mean_ = np.mean(X_train_pca, axis=0)

        print(f"KNN fitted with k={self.k}, metric={self.metric}, weight={self.weight}, pca_components={self.pca_components}")
        print(f"Unknown detection: Mahalanobis with threshold={self.mahal_threshold:.4f}")

    def predict(self, X_test):
        # Match training preprocessing: L2-normalize before PCA/inference
        X_test = normalize(X_test, norm='l2')
        if self.pca:
            X_test_pca = self.pca.transform(X_test)
        else:
            X_test_pca = X_test
        
        # Get KNN predictions with distances
        dist_test, idx_test = self.model.kneighbors(X_test_pca, n_neighbors=self.k)

        # Compute Mahalanobis distances for all test samples
        diff = X_test_pca - self.mean_
        mahal_dist = np.sqrt(np.sum((diff @ self.inv_cov_) * diff, axis=1))

        y_pred = []
        unknown_count = 0

        for i in range(len(X_test_pca)):
            dists = dist_test[i]
            indices = idx_test[i]
            
            # Distance to kth neighbor (last one) – used only for debug/info
            kth_neighbor_distance = dists[-1]
            
            weights = 1 / (dists + 1e-9)
            neighbor_labels = self.y_train[indices]

            class_scores = {}
            for w, lbl in zip(weights, neighbor_labels):
                class_scores[lbl] = class_scores.get(lbl, 0) + w

            best_class = max(class_scores, key=class_scores.get)
            
            # Mahalanobis-based unknown detection
            if mahal_dist[i] > self.mahal_threshold:
                y_pred.append(labels_map["unknown"])
                unknown_count += 1
            else:
                y_pred.append(best_class)

        return np.array(y_pred), unknown_count


    def predict_single_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        extractor = FeatureExtractor()
        fv = extractor.extract_features(image)

        fv_scaled = self.scaler.transform(fv.reshape(1, -1))
        fv_normalized = normalize(fv_scaled, norm='l2')
        if self.pca:
            fv_pca = self.pca.transform(fv_normalized)
        else:
            fv_pca = fv_normalized

        dist, indices = self.model.kneighbors(fv_pca, n_neighbors=self.k)

        print(f"\nKNN distances: {dist[0]}")

        print("\nNearest neighbors:")
        for i, (d, idx) in enumerate(zip(dist[0], indices[0])):
            print(f"  {i+1}. {labels_map_rev[self.y_train[idx]]:<10} (dist: {d:.4f})")
        
        dists = dist[0]
        kth_neighbor_distance = dists[-1]
        neighbor_labels = self.y_train[indices[0]]

        weights = 1 / (dists + 1e-9)
        class_scores = {}

        for w, lbl in zip(weights, neighbor_labels):
            class_scores[lbl] = class_scores.get(lbl, 0) + w

        best_class = max(class_scores, key=class_scores.get)
        
        # Compute Mahalanobis distance for this sample
        diff = fv_pca[0] - self.mean_
        mahal = float(np.sqrt(diff @ self.inv_cov_ @ diff.T))

        # Debug info
        print(f"K-th neighbor distance: {kth_neighbor_distance:.4f}")
        print(f"Mahalanobis distance: {mahal:.4f} (threshold: {self.mahal_threshold:.4f})")
        
        # Final decision based on Mahalanobis distance
        if mahal > self.mahal_threshold:
            pred = "unknown"
            print("\n>>> MARKED AS UNKNOWN (Mahalanobis threshold exceeded)")
        else:
            pred = labels_map_rev[best_class]
            print(f"\n>>> Predicted: {pred}")

        return pred

    def evaluate_unknown_features(self, features_file, labels_file):
        """Evaluate precomputed noisy/unknown features saved as .npy files."""
        if not (os.path.isfile(features_file) and os.path.isfile(labels_file)):
            print(f"Missing features or labels: {features_file}, {labels_file}")
            return {"count": 0, "unknown": 0, "misclassified": []}

        X = np.load(features_file)
        y = np.load(labels_file)

        # Normalize and project to PCA space
        X_norm = normalize(X, norm='l2')
        X_pca = self.pca.transform(X_norm) if self.pca else X_norm

        dist_test, idx_test = self.model.kneighbors(X_pca, n_neighbors=self.k)
        diff = X_pca - self.mean_
        mahal_dist = np.sqrt(np.sum((diff @ self.inv_cov_) * diff, axis=1))

        total = len(X_pca)
        unknown = 0
        misclassified = []

        for i in range(total):
            dists = dist_test[i]
            labels = self.y_train[idx_test[i]]
            weights = 1.0 / (dists + 1e-9)
            scores = {}
            for w, lbl in zip(weights, labels):
                scores[lbl] = scores.get(lbl, 0) + w
            best_class = max(scores, key=scores.get)

            if mahal_dist[i] > self.mahal_threshold:
                unknown += 1
            else:
                misclassified.append({
                    "index": i,
                    "true": int(y[i]) if i < len(y) else None,
                    "pred": labels_map_rev.get(int(best_class), str(best_class)),
                    "mahal": float(mahal_dist[i])
                })

        print("\n" + "-"*60)
        print(f"Unknown features evaluation: {features_file}")
        print(f"Total samples: {total}")
        print(f"Marked unknown: {unknown} ({(unknown/total*100 if total else 0):.1f}%)")
        if misclassified:
            print(f"Known predictions (should be unknown): {len(misclassified)}")
            for m in misclassified[:10]:
                print(f"  idx {m['index']}: true={m['true']} pred={m['pred']} (Mahalanobis {m['mahal']:.2f})")
            if len(misclassified) > 10:
                print(f"  ... and {len(misclassified)-10} more")

        return {"count": total, "unknown": unknown, "misclassified": misclassified}

# Load train and test features
train_loader = FeatureLoader()
X_train, y_train = train_loader.load()

test_loader = FeatureLoader(features_file=TEST_FEATURES, labels_file=TEST_LABELS)
X_test, y_test = test_loader.load()

# X_train = normalize(X_train, norm='l2')
# X_test = normalize(X_test, norm='l2')

# Initialize and train KNN classifier with Mahalanobis unknown detection
# Optimal configuration from search: PCA=0.95 variance, k=3, cosine, distance-weighted, mahal_threshold≈33.2872
knn = KNNClassifier()
knn.fit(X_train, y_train)

# Predict on test set
y_pred, unknown_count = knn.predict(X_test)
print(f"\nMarked as unknown: {unknown_count}/{len(y_test)} ({unknown_count / len(y_test) * 100:.1f}%)")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Single image test
print("\n" + "=" * 60)
print("SINGLE IMAGE TEST")
print("=" * 60)
knn.predict_single_image("unknown/sponge.jpg")

# Evaluate precomputed noisy/unknown features if present
UNKNOWN_FEATURES = "unknown_noisy_features.npy"
UNKNOWN_LABELS = "unknown_noisy_labels.npy"
if os.path.isfile(UNKNOWN_FEATURES) and os.path.isfile(UNKNOWN_LABELS):
    knn.evaluate_unknown_features(UNKNOWN_FEATURES, UNKNOWN_LABELS)


"""
can -> glass (wrong prediction)
can2 -> metal
can3 -> glass (wrong prediction)
can4 -> metal
can5 -> metal

lego -> unknown
remote_control -> unknown
sponge -> unknown
wood -> unknown
"""