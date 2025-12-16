import cv2
import numpy as np
import os
import shutil
from pathlib import Path
import random
import joblib
from feature_extractor import FeatureExtractor
from sklearn.preprocessing import normalize
from configures import *

# Configuration
SOURCE_DATASET = "dataset_test"  # or "dataset_test"
OUTPUT_DIR = "unknown_noisy_images"
# Fewer images per class
IMAGES_PER_CLASS = 10  # Number of source images per class to degrade

# Apply a smaller, stronger subset of degradations
SELECTED_DEGRADATIONS = [
    "blur_heavy",
]

# Create output directory (clean if exists)
if os.path.isdir(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Degradation types
def apply_gaussian_blur(image, kernel_size=25):
    """Apply Gaussian blur to simulate out-of-focus/blurry photos (stronger)."""
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(image, (k, k), 0)

def apply_motion_blur(image, kernel_size=35):
    """Apply motion blur to simulate camera shake (strong linear kernel)."""
    k = kernel_size
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)

def apply_defocus_blur(image, radius=25):
    """Simulate lens defocus blur using a disk kernel."""
    r = max(5, radius)
    size = 2 * r + 1
    y, x = np.ogrid[-r:r+1, -r:r+1]
    mask = x**2 + y**2 <= r**2
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[mask] = 1.0
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)

def apply_zoom_blur(image, steps=8, strength=0.1):
    """Simulate zoom blur by averaging progressively scaled images."""
    h, w = image.shape[:2]
    acc = np.zeros_like(image, dtype=np.float32)
    for i in range(steps):
        scale = 1.0 + strength * (i + 1)
        resized = cv2.resize(image, (int(w * scale), int(h * scale)))
        rh, rw = resized.shape[:2]
        y0 = (rh - h) // 2
        x0 = (rw - w) // 2
        cropped = resized[y0:y0+h, x0:x0+w]
        acc += cropped.astype(np.float32)
    blurred = (acc / steps).clip(0, 255).astype(np.uint8)
    return blurred

def apply_gaussian_noise(image, std=50):
    """Add Gaussian noise to degrade image quality"""
    noise = np.random.normal(0, std, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_brightness_reduction(image, factor=0.8):
    """Reduce brightness significantly (dark/underexposed photo)"""
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def apply_extreme_compression(image, quality=10):
    """Apply heavy JPEG compression artifacts"""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, compressed = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(compressed, 1)

def apply_pixelation(image, block_size=16):
    """Apply pixelation effect (low resolution look)"""
    h, w = image.shape[:2]
    temp = cv2.resize(image, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_extreme_blur_and_noise(image):
    """Combine extreme blur with noise"""
    blurred = cv2.GaussianBlur(image, (25, 25), 0)
    noisy = apply_gaussian_noise(blurred, std=60)
    return noisy

# Dictionary of degradation functions
degradations = {
    "blur_heavy": lambda img: apply_gaussian_blur(img, kernel_size=45)
}

# Process images
processed_count = 0
class_count = {}

for class_folder in os.listdir(SOURCE_DATASET):
    class_path = os.path.join(SOURCE_DATASET, class_folder)
    if not os.path.isdir(class_path):
        continue
    
    class_count[class_folder] = 0
    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Randomly select images to degrade
    selected_images = random.sample(image_files, min(IMAGES_PER_CLASS, len(image_files)))
    
    for img_file in selected_images:
        img_path = os.path.join(class_path, img_file)
        image = cv2.imread(img_path)
        
        if image is None:
            continue
        
        # Apply only selected degradations and save
        for degradation_name in SELECTED_DEGRADATIONS:
            degradation_func = degradations[degradation_name]
            try:
                degraded = degradation_func(image)
                
                # Create output filename
                name_without_ext = os.path.splitext(img_file)[0]
                output_filename = f"{class_folder}_{name_without_ext}_{degradation_name}.jpg"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                
                cv2.imwrite(output_path, degraded)
                processed_count += 1
                class_count[class_folder] += 1
                
                print(f"✓ Created: {output_filename}")
            except Exception as e:
                print(f"✗ Error processing {img_file} with {degradation_name}: {e}")

print("\n" + "="*70)
print("NOISY/BLURRED IMAGE GENERATION COMPLETE")
print("="*70)
print(f"Total images created: {processed_count}")
print(f"Output directory: {OUTPUT_DIR}")
print("\nImages per class (actual saved):")
for class_name, count in class_count.items():
    print(f"  {class_name}: {count} degraded images")

print("\nDegradation types applied:")
for deg_name in SELECTED_DEGRADATIONS:
    print(f"  - {deg_name}")

# ====== Feature extraction for the generated unknown images ======
def extract_and_save_unknown_features(images_dir, features_out="unknown_noisy_features.npy", labels_out="unknown_noisy_labels.npy"):
    files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        print("No images found to extract in:", images_dir)
        return

    extractor = FeatureExtractor()
    X, y = [], []

    for fname in files:
        fpath = os.path.join(images_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fv = extractor.extract_features(img)
        X.append(fv)
        # Label all as unknown (6)
        y.append(6)

    X = np.array(X)
    y = np.array(y)

    # Scale with the same scaler used for train features
    try:
        scaler = joblib.load(SCALER_FILE)
        X_scaled = scaler.transform(X)
    except Exception as e:
        print("Warning: could not load scaler from SCALER_FILE, saving unscaled features. Error:", e)
        X_scaled = X

    np.save(features_out, X_scaled)
    np.save(labels_out, y)
    print("\nSaved unknown noisy features:")
    print("  Features:", features_out, X_scaled.shape)
    print("  Labels:  ", labels_out, y.shape)

# Run extraction for the generated folder
extract_and_save_unknown_features(OUTPUT_DIR)

