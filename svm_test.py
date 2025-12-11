import cv2
import joblib
import numpy as np
import os
from feature_extractor import FeatureExtractor
OUTPUT_SCALER = 'feature_scaler.pkl'

MODEL_FILE = 'svm_model.pkl'
class_names = ['glass', 'paper','cardboard', 'plastic', 'metal', 'trash','unknown']

def test_image(image_path):
    """Test a single image with the SVM model"""
    
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
    
    # Step 5: Prepare features
    print(f"\n5. Preparing features for prediction...")
    features_reshaped = features.reshape(1, -1)
    print(f"   OK: Features reshaped to {features_reshaped.shape}")
    
    # Step 6: Make prediction
    print(f"\n6. Making prediction...")
    try:
        predicted_idx = svm_pipeline.predict(features)[0]
        predicted_name = class_names[predicted_idx]
        print(f"   OK: Prediction made")
        print(f"   Predicted class index: {predicted_idx}")
        print(f"   Predicted class name: {predicted_name.upper()}")
    except Exception as e:
        print(f"   ERROR: Prediction failed - {e}")
        return False
    
    # Step 7: Get confidence scores
    print(f"\n7. Calculating confidence scores...")
    try:
        decision_scores = svm_pipeline.decision_function(features_reshaped)[0]
        probs = np.exp(decision_scores - decision_scores.max())
        probs /= probs.sum()
        print(f"   OK: Scores calculated")
    except Exception as e:
        print(f"   ERROR: Failed to calculate scores - {e}")
        return False
    
    # Step 8: Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nImage: {image_path}")
    print(f"Predicted Class: {predicted_name.upper()}")
    print(f"Confidence: {probs[predicted_idx] * 100:.2f}%")
    
    print("\n" + "-" * 70)
    print("ALL CLASS PROBABILITIES:")
    print("-" * 70)
    
    # Sort by probability for better visualization
    sorted_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    for idx, prob in sorted_probs:
        class_name = class_names[idx]
        confidence = prob * 100
        bar_length = int(confidence / 2)
        bar = "=" * bar_length
        marker = " <- PREDICTED" if idx == predicted_idx else ""
        print(f"{class_name.upper():10s} {confidence:6.2f}% |{bar:<50s}{marker}")
    
    print("\n" + "=" * 70)
    return True


def main():
    print("\n")
    image_path= ("dataset_test/paper/0d8f0b99-d952-4634-8cfe-2f03ab5bfcfe.jpg")
    
    success = test_image(image_path)
    
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
