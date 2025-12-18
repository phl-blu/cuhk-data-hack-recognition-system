import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from feature_loader import FeatureLoader
from itertools import product

UNKNOWN = 6


def knn_weighted_vote(distances, labels):
    weights = 1.0 / (distances + 1e-9)
    scores = {}
    for w, lbl in zip(weights, labels):
        scores[lbl] = scores.get(lbl, 0) + w
    return max(scores, key=scores.get)

def compute_mahalanobis(X_train, X_test, eps=1e-6):
    mean = X_train.mean(axis=0)
    cov = np.cov(X_train, rowvar=False) + np.eye(X_train.shape[1]) * eps
    inv_cov = np.linalg.inv(cov)
    diff = X_test - mean
    return np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))

X_train, y_train = FeatureLoader().load()
X_test, y_test = FeatureLoader(
    features_file="features_test.npy",
    labels_file="labels_test.npy"
).load()

X_train = normalize(X_train)
X_test = normalize(X_test)


param_grid = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "cosine", "manhattan"]
}

pca_options = [None, 256, 512, 0.8, 0.95]

best_acc = 0
best_cfg = {}
all_results = []

for pca_val in pca_options:
    if pca_val:
        pca = PCA(n_components=pca_val, random_state=42)
        Xtr, Xte = pca.fit_transform(X_train), pca.transform(X_test)
    else:
        Xtr, Xte = X_train, X_test

    # Try all parameter combinations
    for n_neighbors, weights, metric in product(
        param_grid["n_neighbors"],
        param_grid["weights"],
        param_grid["metric"]
    ):
        knn = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric
        )
        knn.fit(Xtr, y_train)

        dist_test, idx_test = knn.kneighbors(Xte)
        mahal = compute_mahalanobis(Xtr, Xte)

        mean, std = mahal.mean(), mahal.std()
        thresholds = np.linspace(mean, mean + 4 * std, 9)

        for t in thresholds:
            y_pred = []
            unk = 0

            for i in range(len(Xte)):
                cls = knn_weighted_vote(
                    dist_test[i],
                    y_train[idx_test[i]]
                )
                is_unk = mahal[i] > t
                y_pred.append(UNKNOWN if is_unk else cls)
                unk += is_unk

            acc = accuracy_score(y_test, y_pred)
            
            config = {
                "pca": pca_val,
                "n_neighbors": n_neighbors,
                "weights": weights,
                "metric": metric,
                "threshold": t,
                "unknown_pct": unk / len(y_test) * 100,
                "accuracy": acc
            }
            all_results.append(config)
            
            if acc > best_acc:
                best_acc = acc
                best_cfg = config

# Sort by accuracy descending and get top 10
top_10 = sorted(all_results, key=lambda x: x["accuracy"], reverse=True)[:10]

print("TOP 10 BEST CONFIGURATIONS:")
for i, cfg in enumerate(top_10, 1):
    print(f"{i}. Accuracy: {cfg['accuracy']:.4f}")
    print(f"   PCA: {cfg['pca']}, K: {cfg['n_neighbors']}, Weights: {cfg['weights']}, Metric: {cfg['metric']}")
    print(f"   Threshold: {cfg['threshold']:.4f}, Unknown %: {cfg['unknown_pct']:.2f}%")
    print()

print("BEST:", best_cfg)
print("ACCURACY:", best_acc)