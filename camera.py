import os
import time
import threading
import cv2
import joblib
import numpy as np
from ultralytics import YOLO

from configures import SCALER_FILE, MODEL_FILE
from feature_extractor import FeatureExtractor

class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash', 'unknown']
THRESHOLD = 0.61
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 0, 0)


class Classifier:
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.scaler = joblib.load(SCALER_FILE)
        self.svm = joblib.load(MODEL_FILE)
        self.lock = threading.Lock()
        self.result = ("...", 0.0, None)
        self.latest_frame = None
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit(self, frame):
        with self.lock:
            self.latest_frame = frame.copy()

    def get_result(self):
        with self.lock:
            return self.result

    def _worker(self):
        while self.running:
            with self.lock:
                frame = self.latest_frame
                self.latest_frame = None
            if frame is None:
                time.sleep(0.001)
                continue
            label, prob, box = self._classify(frame)
            with self.lock:
                self.result = (label, prob, box)

    def _classify(self, frame):
        h, w = frame.shape[:2]

        # Fixed center crop (50% of frame)
        margin_x, margin_y = w // 4, h // 4
        x1, y1 = margin_x, margin_y
        x2, y2 = w - margin_x, h - margin_y
        box = [x1, y1, x2, y2]

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "unknown", 0.0, box

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        features = self.extractor.extract_features(crop_rgb)
        features = self.scaler.transform(features.reshape(1, -1))
        probs = self.svm.predict_proba(features)[0]
        max_prob = probs.max()
        label = "unknown" if max_prob < THRESHOLD else class_names[probs.argmax()]
        return label, float(max_prob), box

    def stop(self):
        self.running = False


def run_camera(camera_index=0):
    print("Loading models...")
    classifier = Classifier()
    print("Models loaded. Press 'q' or ESC to quit.")

    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera")

    cv2.namedWindow("Material Identification", cv2.WINDOW_GUI_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        classifier.submit(frame)
        label, prob, box = classifier.get_result()

        if box:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
            text = f"{label} ({prob:.2f})"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), BOX_COLOR, -1)
            cv2.putText(frame, text, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
        else:
            cv2.putText(frame, f"{label} ({prob:.2f})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, BOX_COLOR, 2)

        cv2.imshow("Material Identification", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    classifier.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera()
