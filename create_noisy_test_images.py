import os
import shutil
import random
import numpy as np
from configures import *
from image_feature_utils import load_image_rgb, extract_feature_from_image, scale_features
import cv2

# -------------------------------------------------
# Configuration
# -------------------------------------------------
SOURCE_DATASET = "dataset_test"
OUTPUT_DIR = "unknown_noisy_images"
IMAGES_PER_CLASS = 10
SELECTED_DEGRADATIONS = ["blur_heavy"]

# -------------------------------------------------
# Create/Clean output directory
# -------------------------------------------------
if os.path.isdir(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------
# Degradation functions
# -------------------------------------------------
def apply_gaussian_blur(image, kernel_size=45):
    """Apply strong Gaussian blur."""
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(image, (k, k), 0)

# Dictionary for degradations
degradations = {
    "blur_heavy": apply_gaussian_blur
}

# Process images: degrade and save
processed_count = 0
class_count = {}

for class_folder in os.listdir(SOURCE_DATASET):
    class_path = os.path.join(SOURCE_DATASET, class_folder)
    if not os.path.isdir(class_path):
        continue

    class_count[class_folder] = 0
    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    selected_images = random.sample(image_files, min(IMAGES_PER_CLASS, len(image_files)))

    for img_file in selected_images:
        img_path = os.path.join(class_path, img_file)
        image = load_image_rgb(img_path)
        if image is None:
            continue

        for deg_name in SELECTED_DEGRADATIONS:
            try:
                degraded = degradations[deg_name](image)
                out_fname = f"{class_folder}_{os.path.splitext(img_file)[0]}_{deg_name}.jpg"
                out_path = os.path.join(OUTPUT_DIR, out_fname)
                cv2.imwrite(out_path, cv2.cvtColor(degraded, cv2.COLOR_RGB2BGR))
                processed_count += 1
                class_count[class_folder] += 1
                print(f"✓ Created: {out_fname}")
            except Exception as e:
                print(f"✗ Error processing {img_file} with {deg_name}: {e}")

# Summary
print("\n" + "="*70)
print("NOISY/BLURRED IMAGE GENERATION COMPLETE")
print("="*70)
print(f"Total images created: {processed_count}")
print(f"Output directory: {OUTPUT_DIR}")
print("\nImages per class (actual saved):")
for cls, count in class_count.items():
    print(f"  {cls}: {count} degraded images")
print("\nDegradation types applied:")
for deg_name in SELECTED_DEGRADATIONS:
    print(f"  - {deg_name}")

# Extract features for unknown images
def extract_and_save_unknown_features(images_dir, features_out="unknown_noisy_features.npy", labels_out="unknown_noisy_labels.npy"):
    X, y = [], []
    for fname in os.listdir(images_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        fpath = os.path.join(images_dir, fname)
        img = load_image_rgb(fpath)
        if img is None:
            continue
        fv = extract_feature_from_image(img)
        X.append(fv)
        y.append(6)  # unknown label

    if not X:
        print("No features extracted.")
        return

    X = np.array(X)
    y = np.array(y)
    X_scaled = scale_features(X)
    np.save(features_out, X_scaled)
    np.save(labels_out, y)

    print("\nSaved unknown noisy features:")
    print("  Features:", features_out, X_scaled.shape)
    print("  Labels:  ", labels_out, y.shape)

# Run extraction
extract_and_save_unknown_features(OUTPUT_DIR)
