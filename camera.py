import os
import time
import cv2
import joblib
import numpy as np

from configures import SCALER_FILE
from KNN import KNNClassifier, labels_map_rev
from feature_extractor import FeatureExtractor

MODEL_PATH = "knn_model.pkl"


def _load_knn(model_path):
    model = joblib.load(model_path)

    if getattr(model, "extractor", None) is None:
        model.extractor = FeatureExtractor()
    if getattr(model, "scaler", None) is None:
        if not os.path.exists(SCALER_FILE):
            raise FileNotFoundError(f"Scaler file not found: {SCALER_FILE}")
        model.scaler = joblib.load(SCALER_FILE)
    return model


def run_camera(model_path=MODEL_PATH, camera_index=0):
    knn = _load_knn(model_path)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera")

    fps_list = []
    start_run = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        features = knn.extractor.extract_features(frame_rgb)
        features = knn.scaler.transform(features.reshape(1, -1))
        pred, _ = knn.predict(features)

        t1 = time.time()
        fps = 1.0 / max(t1 - t0, 1e-6)
        fps_list.append(fps)

        label = labels_map_rev[int(pred[0])]
        cv2.putText(frame, f"{label} | FPS: {fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Material Identification", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or 'q'
            break

    cap.release()
    cv2.destroyAllWindows()

    if fps_list:
        print(f"Average FPS: {np.mean(fps_list):.2f}")
        print(f"Min FPS: {np.min(fps_list):.2f}")
        print(f"Max FPS: {np.max(fps_list):.2f}")
    print(f"Runtime: {(time.time() - start_run)/60:.1f} minutes")


if __name__ == "__main__":
    run_camera()
