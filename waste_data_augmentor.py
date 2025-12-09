import os
import random
import shutil
from math import ceil
import numpy as np
from PIL import Image
from keras_preprocessing.image import ImageDataGenerator, img_to_array, load_img


class WasteDataAugmentor:
    def __init__(self, source_dir, output_dir, target_size=(224, 224)):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.target_size = target_size

        # Categories = each folder
        self.categories = [
            d for d in os.listdir(source_dir)
            if os.path.isdir(os.path.join(source_dir, d))
        ]

        # Prepare output directories
        os.makedirs(output_dir, exist_ok=True)
        for category in self.categories:
            os.makedirs(os.path.join(output_dir, category), exist_ok=True)

    @staticmethod
    def is_valid_image(file_path): # doesn't use self of the class
        try:
            img = Image.open(file_path)
            img.verify() 
            return True
        except Exception:
            return False

    def augment_dataset(self, augmentation_factor=0.3, copy_originals=True,target_class_size = 500):
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.7, 1.3],
            fill_mode='nearest'
        )
        print("\n" + "=" * 70)
        print(f"AUGMENTATION {'WITH ORIGINALS' if copy_originals else 'ONLY'}")
        print("=" * 70)

        stats = {
            'total_original': 0,
            'total_augmented': 0,
            'per_category': {}
        }

        for category in self.categories:
            category_path = os.path.join(self.source_dir, category)
            output_path = os.path.join(self.output_dir, category)

            # load valid images
            image_files = [
                f for f in os.listdir(category_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
                and self.is_valid_image(os.path.join(category_path, f))
            ]

            print(f"\nProcessing {category}: {len(image_files)} images")

            if len(image_files) == 0:
                print("No valid images found — skipping")
                continue

            total_images = len(image_files)
    
            if(total_images >= target_class_size):
                # use the factor
                total_needed = ceil(augmentation_factor * total_images)
                print(f"Using Factor, Need {total_needed} augmentations")
            else:
                # Calculate augmentations needed to reach target_class_size
                if copy_originals:
                    # Augmentations = target - originals (originals will be copied)
                    total_needed = max(0, target_class_size - total_images)
                else:
                    # Generate all images through augmentation
                    total_needed = target_class_size
                print(f"Target size: {target_class_size}, Need {total_needed} augmentations")
            
            base_aug = total_needed // total_images       # equal for all
            extra = total_needed % total_images           # remaining to distribute

            augment_map = {img: base_aug for img in image_files}
            
            # Randomly select which images get +1
            if extra > 0:
                extra_images = random.sample(image_files, extra)
                for img in extra_images:
                    augment_map[img] += 1
                    
            
            original_count = 0
            augmented_count = 0

            for img_file in image_files:
                img_path = os.path.join(category_path, img_file)

                # Copy original
                if copy_originals:
                    try:
                        shutil.copy2(img_path, os.path.join(output_path, img_file))
                        original_count += 1
                    except Exception as e:
                        print(f"Error copying {img_file}: {e}")

                # Number of augmentations for this image
                aug_times = augment_map[img_file]

                # Generate augmentations
                try:
                    img = load_img(img_path, target_size=self.target_size)
                    x = img_to_array(img)
                    x = np.expand_dims(x, axis=0)

                    for i in range(aug_times):
                        batch = next(datagen.flow(x, batch_size=1))

                        aug_filename = f"{os.path.splitext(img_file)[0]}_aug_{i}.jpg"
                        aug_path = os.path.join(output_path, aug_filename)

                        aug_img = Image.fromarray(np.uint8(batch[0]))
                        aug_img.save(aug_path, quality=95)

                        augmented_count += 1

                except Exception as e:
                    print(f"Error augmenting {img_file}: {e}")

            print(f"Copied originals: {original_count}")
            print(f"Created augmentations: {augmented_count}")

            stats['total_original'] += original_count
            stats['total_augmented'] += augmented_count
            stats['per_category'][category] = {
                'original': original_count,
                'augmented': augmented_count,
                'total': original_count + augmented_count
            }
            print("#"*20)
            # print total images in category
            print(f"Total images in {category}: {stats['per_category'][category]['total']}")

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Original images: {stats['total_original']}")
        print(f"Augmented images: {stats['total_augmented']}")
        print(f"Total dataset: {stats['total_original'] + stats['total_augmented']}")
        print("=" * 70)

        return stats
