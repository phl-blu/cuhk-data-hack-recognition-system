import os
import shutil
import random
from pathlib import Path
from PIL import Image

# Configuration
SOURCE_DIR = "dataset"
TRAIN_DIR = "dataset_train"
TEST_DIR = "dataset_test"
TEST_RATIO = 0.3  # 30% for testing, 70% for training
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

@staticmethod
def is_valid_image(file_path):
    """Check if image file is valid and not corrupted"""
    try:
        img = Image.open(file_path)
        img.verify()
        return True
    except Exception:
        return False

def split_dataset():
    """Split dataset into train and test folders"""
    
    # Get all categories
    categories = [d for d in os.listdir(SOURCE_DIR) 
                  if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    print(f"Found {len(categories)} categories: {categories}")
    print(f"\nSplitting ratio: {(1-TEST_RATIO)*100:.0f}% train, {TEST_RATIO*100:.0f}% test")
    print("=" * 70)
    
    # train/test directories
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    total_train = 0
    total_test = 0
    
    for category in sorted(categories):
        source_path = os.path.join(SOURCE_DIR, category)
        train_path = os.path.join(TRAIN_DIR, category)
        test_path = os.path.join(TEST_DIR, category)
        
        # Create category directories
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        
        # get all image files
        all_images = [f for f in os.listdir(source_path) 
                      if os.path.isfile(os.path.join(source_path, f)) and
                      f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.jfif', '.gif', '.webp'))]
        
        # validate images and filter out corrupted ones
        valid_images = []
        corrupted_count = 0
        
        for img_file in all_images:
            img_path = os.path.join(source_path, img_file)
            if is_valid_image(img_path):
                valid_images.append(img_file)
            else:
                corrupted_count += 1
        
        images = valid_images
        
        # shuffle and split
        random.shuffle(images)
        split_idx = int(len(images) * (1 - TEST_RATIO))
        
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # copy train images
        for img in train_images:
            src = os.path.join(source_path, img)
            dst = os.path.join(train_path, img)
            shutil.copy2(src, dst)
        
        # copy test images
        for img in test_images:
            src = os.path.join(source_path, img)
            dst = os.path.join(test_path, img)
            shutil.copy2(src, dst)
        
        total_train += len(train_images)
        total_test += len(test_images)
        
        print(f"{category.upper()}")
        print(f"  Total: {len(images)} images")
        print(f"  Train: {len(train_images)} images → {train_path}")
        print(f"  Test:  {len(test_images)} images → {test_path}")
        if corrupted_count > 0:
            print(f"  Found {corrupted_count} corrupted image(s)")
    
    print("\n" + "=" * 70)
    print(f"SUMMARY")
    print("=" * 70)
    print(f"Total Train Images: {total_train}")
    print(f"Total Test Images:  {total_test}")
    print(f"Total Images:       {total_train + total_test}")
    print(f"\nTrain directory: {TRAIN_DIR}/")
    print(f"Test directory:  {TEST_DIR}/")
    print("=" * 70)

if __name__ == "__main__":
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: '{SOURCE_DIR}' directory not found!")
        exit(1)
    
    split_dataset()
    print("\nDataset split successfully!")
