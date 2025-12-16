import os
from PIL import Image
import imagehash
from tqdm import tqdm
from keras_preprocessing.image import load_img

def remove_duplicate_images(root_folder, hashfunc=imagehash.phash, hash_size=16, threshold=5):

    def is_valid_image(file_path):
        """Check if image file is valid and not corrupted"""
        if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
            return False
        try:
            img = Image.open(file_path)
            img.load()  # fully load image
            return True
        except Exception:
            return False



    for class_folder in os.listdir(root_folder):
        class_path = os.path.join(root_folder, class_folder)
        if not os.path.isdir(class_path):
            continue

        print(f"Checking folder: {class_path}")
        hashes = {}
        files = os.listdir(class_path)

        for fname in files:
            fpath = os.path.join(class_path, fname)
            if not is_valid_image(fpath):
                print(f"Skipping invalid image: {fpath}")
                continue

            try:
                img = Image.open(fpath)
                img_hash = hashfunc(img, hash_size=hash_size)

                duplicate_found = False
                for h in hashes:
                    if img_hash - h <= threshold:
                        print(f"Remove duplicate: {fpath}")
                        os.remove(fpath)
                        duplicate_found = True
                        break
                if not duplicate_found:
                    hashes[img_hash] = fpath
            except Exception as e:
                print(f"Cannot process {fpath}: {e}")


remove_duplicate_images("dataset_train")