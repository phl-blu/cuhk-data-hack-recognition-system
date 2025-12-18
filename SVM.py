import os
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from configures import *
from feature_loader import FeatureLoader

loader = FeatureLoader()
X, y = loader.load()
print("Train:", X.shape, y.shape)

pipe = Pipeline([
    ("pca", PCA(n_components=None, whiten=False)),
    ("svc", SVC(
        kernel="poly",
        C=620.1776452596954,
        gamma=0.03392699327922517,
        degree=2,
        coef0=0.0,
        class_weight="balanced",
        probability=True,
        random_state=42,
        max_iter=10000,
        tol=0.001
    ))
])

pipe.fit(X, y)

y_train_pred = pipe.predict(X)
train_acc = accuracy_score(y, y_train_pred)
print("Training Accuracy:", round(train_acc, 4))

loader = FeatureLoader(
    features_file="features_test.npy",
    labels_file="labels_test.npy"
)
X_test, y_test = loader.load()

y_test_pred = pipe.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", round(test_acc, 4))

print(classification_report(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))

joblib.dump(pipe, MODEL_FILE)
print("Saved model to:", MODEL_FILE)
