import cv2
import joblib
import numpy as np
import os
from feature_extractor import FeatureExtractor

OUTPUT_SCALER = 'feature_scaler.pkl'
MODEL_FILE = 'svm_model.pkl'

# Only the 6 trained classes
class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']

def test_image(image_path, threshold=0.61):
    """Test a single image with the SVM model and Unknown rejection"""

    print("=" * 70)
    print("SVM MODEL IMAGE TESTER")
    print("=" * 70)

    # Step 1: Validate image path
    print(f"\n1. Checking image path: {image_path}")
    if not os.path.exists(image_path):
        print(f"   ERROR: Image not found at {image_path}")
        return False
    print(f"   OK: Image found")

    # Step 2: Load image
    print(f"\n2. Loading image...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"   ERROR: Cannot read image. Check if it's a valid image file")
        return False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"   OK: Image loaded")
    print(f"   Shape: {img.shape} (Height x Width x Channels)")

    # Step 3: Extract features
    print(f"\n3. Extracting features using ResNet50...")
    try:
        extractor = FeatureExtractor()
        features = extractor.extract_features(img)
        features = np.array(features)
        scaler = joblib.load(OUTPUT_SCALER)
        features = scaler.transform(features.reshape(1, -1))
        print(f"   OK: Features extracted")
        print(f"   Shape: {features.shape}")
    except Exception as e:
        print(f"   ERROR: Failed to extract features - {e}")
        return False

    # Step 4: Load model
    print(f"\n4. Loading trained SVM model: {MODEL_FILE}")
    if not os.path.exists(MODEL_FILE):
        print(f"   ERROR: Model file not found. Train the model first with SVM.py")
        return False
    try:
        svm_pipeline = joblib.load(MODEL_FILE)
        print(f"   OK: Model loaded successfully")
        print(f"   Pipeline: {list(svm_pipeline.named_steps.keys())}")
    except Exception as e:
        print(f"   ERROR: Failed to load model - {e}")
        return False

    # Step 5: Make prediction with probabilities
    print(f"\n5. Making prediction...")
    try:
        probs = svm_pipeline.predict_proba(features)[0]
        max_prob = probs.max()
        predicted_idx = probs.argmax()

        if max_prob < threshold:
            predicted_name = "unknown"
        else:
            predicted_name = class_names[predicted_idx]

        print(f"   OK: Prediction made")
        print(f"   Predicted class name: {predicted_name.upper()}")
        print(f"   Confidence: {max_prob * 100:.2f}%")
    except Exception as e:
        print(f"   ERROR: Prediction failed - {e}")
        return False

    # Step 6: Display all class probabilities
    print("\n" + "-" * 70)
    print("ALL CLASS PROBABILITIES:")
    print("-" * 70)

    sorted_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    for idx, prob in sorted_probs:
        class_name = class_names[idx]
        confidence = prob * 100
        bar_length = int(confidence / 2)
        bar = "=" * bar_length

        # If threshold triggered Unknown, mark Unknown instead of the top class
        if predicted_name == "unknown" and idx == probs.argmax():
            marker = " <- TOP CLASS (REJECTED, UNKNOWN)"
        elif class_name == predicted_name:
            marker = " <- PREDICTED"
        else:
            marker = ""

        print(f"{class_name.upper():10s} {confidence:6.2f}% |{bar:<50s}{marker}")

    # Add explicit Unknown line if triggered
    if predicted_name == "unknown":
        print(f"{'UNKNOWN':10s} {max_prob*100:6.2f}% |{'='*int(max_prob*50):<50s} <- PREDICTED")

    print("\n" + "=" * 70)
    return True


def main():
    print("\n")
    # image_path= ("unknown/lego.jpg")
    # image_path= ("new_data/OIP.webp")
    image_path= ("unknown_noisy_images/cardboard_3b60ce96-4472-4918-afbc-ae5298852948_blur_heavy.jpg")
    
    success = test_image(image_path)
    
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
