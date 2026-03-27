import os
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from scipy.stats import loguniform

from configures import *
from feature_loader import FeatureLoader

# -------------------------------
# Load deep features
# -------------------------------
loader = FeatureLoader()
X, y = loader.load()
print("Train:", X.shape, y.shape)

# -------------------------------
# Pipeline: SCALER → PCA → SVM
# -------------------------------
pipe = Pipeline([
    ("pca", PCA()), #Reduce dimensionality for faster training and potentially better generalization
    ("svc", SVC(random_state=42, probability=True, class_weight="balanced")) #Help with imbalanced classes
])

# -------------------------------
# Hyperparameter search space
# -------------------------------
param_distributions = {
    "pca__n_components": [128, 256, 512],
    "pca__whiten": [True, False],
    "svc__kernel": ["rbf", "linear"],
    "svc__C": loguniform(1e-1, 1e3),
    "svc__gamma": loguniform(1e-5, 1e-1),
    "svc__tol": [1e-3, 1e-4],
    "svc__max_iter": [10000]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

rs = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_distributions,
    n_iter=20,
    scoring="accuracy",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=2,
    refit=True
)

# -------------------------------
# Train SVM
# -------------------------------
print("\nStarting Randomized Search…\n")
rs.fit(X, y)

best = rs.best_estimator_
print("\nBest Params:", rs.best_params_)
print("Best CV Accuracy:", round(rs.best_score_, 4))

# -------------------------------
# Training Accuracy
# -------------------------------
y_train_pred = best.predict(X)
train_acc = accuracy_score(y, y_train_pred)
print("\nTraining Accuracy:", round(train_acc, 4))

# -------------------------------

# Test Accuracy (before thresholding)
# -------------------------------
loader = FeatureLoader(features_file="features_test.npy", labels_file="labels_test.npy")
X_test, y_test = loader.load()
y_test_pred = best.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", round(test_acc, 4))
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# -------------------------------
# Save Final Model
# -------------------------------
joblib.dump(best, MODEL_FILE)
print("\nSaved model to:", MODEL_FILE)