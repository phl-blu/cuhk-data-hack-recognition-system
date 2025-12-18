from configures import *
from data_augmentor import *

if __name__ == "__main__":
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        print(f"Removed existing directory: {OUTPUT_DIR}")
    augmentor = DataAugmentor(
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR,
        target_size=(224, 224)
    )
    stats = augmentor.augment_dataset(
        augmentation_factor=0.3,
        copy_originals=True 
    )
    print(f"\nComplete dataset saved to: {OUTPUT_DIR}")
