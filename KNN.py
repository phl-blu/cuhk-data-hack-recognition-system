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

labels_map = {
    "glass": 0, "paper": 1, "cardboard": 2,
    "plastic": 3, "metal": 4, "trash": 5, "unknown": 6
}
labels_map_rev = {v: k for k, v in labels_map.items()}


class KNNClassifier:
    def __init__(self, k=3, metric='cosine', weight='distance', pca_components=0.95,
                 mahal_threshold=32.5, scaler_file=SCALER_FILE):
        self.k = k
        self.metric = metric
        self.weight = weight
        self.pca_components = pca_components  # % of variance to keep in PCA
        self.mahal_threshold = mahal_threshold  # threshold for unknown detection
        self.scaler_file = scaler_file

        self.model = KNeighborsClassifier(n_neighbors=self.k, metric=self.metric, weights=self.weight)
        
        # PCA for dimensionality reduction (retain pca_components variance)
        self.pca = PCA(n_components=self.pca_components, random_state=42) if self.pca_components else None

        self.scaler = joblib.load(self.scaler_file)
        self.mean_ = None         # mean of training data (for Mahalanobis)
        self.inv_cov_ = None      # inverse covariance (for Mahalanobis)
        self.X_train_pca = None   # training features after PCA
        self.y_train = None
        self.extractor = FeatureExtractor()  # feature extractor for images

    def fit(self, X_train, y_train):
        X_train = normalize(X_train, norm='l2')  # normalize features for cosine distance 

        self.X_train_pca = self.pca.fit_transform(X_train) if self.pca else X_train
        self.y_train = y_train

        # Fit KNN on reduced features
        self.model.fit(self.X_train_pca, y_train)

        # Compute covariance and mean for Mahalanobis distance
        cov = np.cov(self.X_train_pca, rowvar=False) + np.eye(self.X_train_pca.shape[1]) * 1e-6
        self.inv_cov_ = np.linalg.inv(cov)  # inverse covariance
        self.mean_ = np.mean(self.X_train_pca, axis=0)  # mean vector
        print(f"KNN fitted: k={self.k}, metric={self.metric}, weight={self.weight}, PCA={self.pca_components}")
        print(f"Unknown detection: Mahalanobis threshold={self.mahal_threshold}")


    def predict(self, X_test):
        X_test = normalize(X_test, norm='l2')
        X_test_pca = self.pca.transform(X_test) if self.pca else X_test  # project test features to PCA space

        # KNN distances and indices
        dist_test, idx_test = self.model.kneighbors(X_test_pca, n_neighbors=self.k)
        # dist_test: distances to k nearest neighbors (sorted closest first)
        # idx_test: indices of those neighbors in the training set

        # Mahalanobis distance to training data mean
        diff = X_test_pca - self.mean_
        mahal_dist = np.sqrt(np.sum((diff @ self.inv_cov_) * diff, axis=1))

        y_pred, unknown_count = [], 0
        for i, dists in enumerate(dist_test):
            neighbor_labels = self.y_train[idx_test[i]]
            weights = 1 / (dists + 1e-9)  # inverse distance weighting
            scores = {}
            for w, lbl in zip(weights, neighbor_labels):
                scores[lbl] = scores.get(lbl, 0) + w
            best_class = max(scores, key=scores.get)

            # Check if sample is "unknown" using Mahalanobis threshold
            if mahal_dist[i] > self.mahal_threshold:
                y_pred.append(labels_map["unknown"])
                unknown_count += 1
            else:
                y_pred.append(best_class)
        return np.array(y_pred), unknown_count


    def predict_image(self, image_path, verbose=True):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fv = self.extractor.extract_features(image)  

        # Scale, normalize, and apply PCA
        fv_scaled = self.scaler.transform(fv.reshape(1, -1))

        y_pred, _ = self.predict(fv_scaled)

        if verbose:
            print(f"Image: {image_path} -> Predicted: {labels_map_rev[y_pred[0]]}")
        return labels_map_rev[y_pred[0]]


    def predict_directory(self, dir_path, verbose=True):
        results = {}
        files = [f for f in os.listdir(dir_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for f in files:
            full_path = os.path.join(dir_path, f)
            pred = self.predict_image(full_path, verbose=verbose)
            results[f] = pred
        return results


    def save_model(self, path="knn_model.pkl"):
        joblib.dump(self, path)
        print(f"Saved model to {path}")

    @staticmethod
    def load_model(path="knn_model.pkl"):
        return joblib.load(path)

def main():
    # Load features from precomputed file
    train_loader = FeatureLoader()
    X_train, y_train = train_loader.load()

    test_loader = FeatureLoader(features_file=TEST_FEATURES, labels_file=TEST_LABELS)
    X_test, y_test = test_loader.load()

    # Initialize, fit KNN, and save model
    knn = KNNClassifier()
    knn.fit(X_train, y_train)
    knn.save_model("knn_model.pkl")

    # Test on held-out test set
    y_pred, unk_count = knn.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Marked as unknown: {unk_count}/{len(y_test)} ({unk_count/len(y_test)*100:.1f}%)")

    # Predict images in directories
    knn.predict_directory("unknown")
    knn.predict_directory("new_data")


if __name__ == "__main__":
    main()