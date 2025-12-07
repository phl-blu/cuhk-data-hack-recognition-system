import os
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from scipy.stats import loguniform

FEATURES_FILE = "features.npy"
LABELS_FILE = "labels.npy"
MODEL_FILE = "svm_model.pkl"

# -------------------------------
# Load deep features
# -------------------------------
if not os.path.exists(FEATURES_FILE) or not os.path.exists(LABELS_FILE):
    raise SystemExit("Missing feature files. Run deep_feature_extractor.py first.")

X = np.load(FEATURES_FILE)
y = np.load(LABELS_FILE)

print("Loaded deep features:", X.shape, "labels:", y.shape)

# -------------------------------
# Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Train:", X_train.shape, "Test:", X_test.shape)

# -------------------------------
# Pipeline: SCALER → SVM
# -------------------------------
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(random_state=42, probability=False, class_weight="balanced"))
])

# -------------------------------
# Hyperparameter search space
# -------------------------------
param_distributions = {
    "svc__kernel": ["rbf"],
    "svc__C": loguniform(1e-2, 100),
    "svc__gamma": loguniform(1e-4, 1e-1),
    "svc__tol": [1e-3],
    "svc__max_iter": [5000],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rs = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_distributions,
    n_iter=50,            # explore more combinations
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
rs.fit(X_train, y_train)

best = rs.best_estimator_
print("\nBest Params:", rs.best_params_)
print("Best CV Accuracy:", round(rs.best_score_, 4))

# -------------------------------
# Training Accuracy
# -------------------------------
y_train_pred = best.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)
print("\nTraining Accuracy:", round(train_acc, 4))

# -------------------------------
# Test Accuracy
# -------------------------------
y_test_pred = best.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", round(test_acc, 4))

# -------------------------------
# Detailed Metrics
# -------------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# -------------------------------
# Save Final Model
# -------------------------------
joblib.dump(best, MODEL_FILE)
print("\nSaved model to:", MODEL_FILE)
