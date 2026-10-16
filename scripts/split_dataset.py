from pathlib import Path
import shutil
import random

# ==========================================================
# SETTINGS
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT_DIR / "dataset" / "processed"

TRAIN_DIR = ROOT_DIR / "dataset" / "train"
VAL_DIR = ROOT_DIR / "dataset" / "val"
TEST_DIR = ROOT_DIR / "dataset" / "test"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}

random.seed(RANDOM_SEED)


# ==========================================================
# CLEAR FOLDER
# ==========================================================

def clear_folder(folder):

    if folder.exists():

        for item in folder.iterdir():

            if item.is_file():
                item.unlink()

            elif item.is_dir():
                shutil.rmtree(item)


# ==========================================================
# SPLIT DATASET
# ==========================================================

def split_dataset():

    print("\n===================================")
    print("DATASET SPLITTING")
    print("===================================\n")

    if not PROCESSED_DIR.exists():

        print("ERROR: Processed dataset not found.")
        print("Run prepare_dataset.py first.")
        return

    class_folders = [
        folder
        for folder in PROCESSED_DIR.iterdir()
        if folder.is_dir()
    ]

    for class_folder in class_folders:

        class_name = class_folder.name

        train_class = TRAIN_DIR / class_name
        val_class = VAL_DIR / class_name
        test_class = TEST_DIR / class_name

        train_class.mkdir(parents=True, exist_ok=True)
        val_class.mkdir(parents=True, exist_ok=True)
        test_class.mkdir(parents=True, exist_ok=True)

        # Remove old split files
        clear_folder(train_class)
        clear_folder(val_class)
        clear_folder(test_class)

        images = [
            image
            for image in class_folder.iterdir()
            if image.suffix.lower() in IMAGE_EXTENSIONS
        ]

        random.shuffle(images)

        total = len(images)

        train_count = int(total * TRAIN_RATIO)
        val_count = int(total * VAL_RATIO)

        train_images = images[:train_count]

        val_images = images[
            train_count:
            train_count + val_count
        ]

        test_images = images[
            train_count + val_count:
        ]

        # Copy training images
        for image in train_images:

            shutil.copy2(
                image,
                train_class / image.name
            )

        # Copy validation images
        for image in val_images:

            shutil.copy2(
                image,
                val_class / image.name
            )

        # Copy test images
        for image in test_images:

            shutil.copy2(
                image,
                test_class / image.name
            )

        print(f"\nClass: {class_name}")
        print(f"Total : {total}")
        print(f"Train : {len(train_images)}")
        print(f"Val   : {len(val_images)}")
        print(f"Test  : {len(test_images)}")

    print("\n===================================")
    print("DATASET SPLIT COMPLETED")
    print("===================================")


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    split_dataset()