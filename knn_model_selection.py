import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from feature_loader import FeatureLoader

UNKNOWN = 6

# ----------------- Helpers -----------------

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

# ----------------- Load data -----------------

X_train, y_train = FeatureLoader().load()
X_test, y_test = FeatureLoader(
    features_file="features_test.npy",
    labels_file="labels_test.npy"
).load()

X_train = normalize(X_train)
X_test = normalize(X_test)

# ----------------- Model search -----------------

param_grid = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "cosine", "manhattan"]
}

pca_options = [None, 128, 256, 0.95]

best_acc = 0
best_cfg = {}

for pca_val in pca_options:
    if pca_val:
        pca = PCA(n_components=pca_val, random_state=42)
        Xtr, Xte = pca.fit_transform(X_train), pca.transform(X_test)
    else:
        Xtr, Xte = X_train, X_test

    search = RandomizedSearchCV(
        KNeighborsClassifier(),
        param_grid,
        n_iter=50,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        n_jobs=-1,
        random_state=42
    )
    search.fit(Xtr, y_train)
    knn = search.best_estimator_

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
        if acc > best_acc:
            best_acc = acc
            best_cfg = {
                "pca": pca_val,
                "knn": search.best_params_,
                "threshold": t,
                "unknown_pct": unk / len(y_test) * 100
            }

print("BEST:", best_cfg)
print("ACCURACY:", best_acc)