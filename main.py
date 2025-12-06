from configures import *
from waste_data_augmentor import *

if __name__ == "__main__":
    if os.path.exists(OUTPUT_DIR) and any(os.scandir(OUTPUT_DIR)):
        response = input(f"\nDirectory '{OUTPUT_DIR}' already exists! Overwrite? (yes/no): ")
        if response.lower() != 'yes':
            print("Operation cancelled.")
            exit()
        print("Overwriting existing dataset...")
    augmentor = WasteDataAugmentor(
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR,
        target_size=(224, 224)
    )
    stats = augmentor.augment_dataset(
        augmentation_factor=0.3,
        copy_originals=True 
    )
    print(f"\nComplete dataset saved to: {OUTPUT_DIR}")
