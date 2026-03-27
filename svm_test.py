import cv2
import joblib
import numpy as np
import os
from feature_extractor import FeatureExtractor
from feature_loader import FeatureLoader
from configures import TEST_FEATURES, TEST_LABELS

OUTPUT_SCALER = 'feature_scaler.pkl'
MODEL_FILE = 'svm_model.pkl'

# Only the 6 trained classes
class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']

def test_image(image_path, threshold=0.61, debug=False):
    if not os.path.exists(image_path):
        if debug:
            print(f"[skip] path not found: {image_path}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        if debug:
            print(f"[fail] cv2.imread returned None: {image_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        extractor = FeatureExtractor()
        features = extractor.extract_features(img)
        features = np.array(features)
        scaler = joblib.load(OUTPUT_SCALER)
        features = scaler.transform(features.reshape(1, -1))
    except Exception as e:
        if debug:
            print(f"[fail] feature extraction: {e}")
        return None

    if not os.path.exists(MODEL_FILE):
        if debug:
            print(f"[fail] model file missing: {MODEL_FILE}")
        return None
    try:
        svm_pipeline = joblib.load(MODEL_FILE)
    except Exception as e:
        if debug:
            print(f"[fail] loading model: {e}")
        return None

    try:
        probs = svm_pipeline.predict_proba(features)[0]
        max_prob = probs.max()
        predicted_idx = probs.argmax()
        predicted_name = "unknown" if max_prob < threshold else class_names[predicted_idx]
        return (image_path, predicted_name)
    except Exception as e:
        if debug:
            print(f"[fail] prediction: {e}")
        return None


def test_images_from_folder(folder_path,threshold=0.61, debug=False):
    results = []
    for name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, name)
        if os.path.isfile(image_path):
            res = test_image(image_path, threshold, debug=debug)
            if res is not None:
                results.append(res)
    return results

def evaluate_from_npy(features_file=TEST_FEATURES, labels_file=TEST_LABELS, threshold=0.61):
    loader = FeatureLoader(features_file=features_file, labels_file=labels_file)
    X_test, y_test = loader.load()

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")

    svm_pipeline = joblib.load(MODEL_FILE)
    n = len(y_test)

    correct = 0
    unknown_preds = 0

    # Per-class stats
    per_class_total = {c: 0 for c in range(len(class_names))}
    per_class_correct = {c: 0 for c in range(len(class_names))}

    # Predict probabilities in batches to speed up
    probs_all = svm_pipeline.predict_proba(X_test)

    for i in range(n):
        probs = probs_all[i]
        max_prob = probs.max()
        idx = probs.argmax()
        pred_num = svm_pipeline.classes_[idx]

        gt = int(y_test[i])
        per_class_total[gt] += 1

        if max_prob < threshold:
            unknown_preds += 1
            # Unknown counts as incorrect for dataset_test
        else:
            if pred_num == gt:
                correct += 1
                per_class_correct[gt] += 1

    processed = n
    overall_acc = (correct / processed * 100.0) if processed else 0.0
    unknown_rate = (unknown_preds / processed * 100.0) if processed else 0.0

    print("\n=== SVM Fast Evaluation (NPY) ===")
    print(f"Features: {features_file}")
    print(f"Labels:   {labels_file}")
    print(f"Threshold: {threshold}")
    print("--- Summary ---")
    print(f"Processed:       {processed}")
    print(f"Unknown preds:   {unknown_preds} ({unknown_rate:.2f}%)")
    print(f"Overall accuracy: {overall_acc:.2f}%")

    print("\nPer-class accuracy:")
    for cls_num in range(len(class_names)):
        t = per_class_total[cls_num]
        c = per_class_correct[cls_num]
        acc = (c / t * 100.0) if t else 0.0
        print(f"- {class_names[cls_num]}: {c}/{t} ({acc:.2f}%)")

    return {
        "processed": processed,
        "unknown_preds": unknown_preds,
        "unknown_rate": unknown_rate,
        "overall_accuracy": overall_acc,
        "per_class_total": per_class_total,
        "per_class_correct": per_class_correct,
    }
def main():
    evaluate_from_npy(TEST_FEATURES, TEST_LABELS, threshold=0.61)


if __name__ == "__main__":
    main()
