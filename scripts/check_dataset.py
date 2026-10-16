from pathlib import Path
from PIL import Image

# ==========================================================
# PATHS
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT_DIR / "dataset" / "processed"
TRAIN_DIR = ROOT_DIR / "dataset" / "train"
VAL_DIR = ROOT_DIR / "dataset" / "val"
TEST_DIR = ROOT_DIR / "dataset" / "test"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}


# ==========================================================
# COUNT IMAGES
# ==========================================================

def count_images(folder):

    if not folder.exists():
        return 0

    return len([
        file
        for file in folder.iterdir()
        if file.suffix.lower() in IMAGE_EXTENSIONS
    ])


# ==========================================================
# CHECK DIRECTORY
# ==========================================================

def check_directory(directory, title):

    print("\n===================================")
    print(title)
    print("===================================")

    if not directory.exists():

        print("Folder does not exist:")
        print(directory)
        return

    total = 0

    class_folders = sorted([
        folder
        for folder in directory.iterdir()
        if folder.is_dir()
    ])

    for class_folder in class_folders:

        count = count_images(class_folder)

        total += count

        print(
            f"{class_folder.name:<25} "
            f"{count}"
        )

    print("-----------------------------------")
    print(f"Total images: {total}")


# ==========================================================
# CHECK CORRUPTED IMAGES
# ==========================================================

def check_corrupted_images():

    print("\n===================================")
    print("CHECKING CORRUPTED IMAGES")
    print("===================================")

    corrupted = []

    if not PROCESSED_DIR.exists():
        return

    for class_folder in PROCESSED_DIR.iterdir():

        if not class_folder.is_dir():
            continue

        for image_path in class_folder.iterdir():

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:

                with Image.open(image_path) as img:
                    img.verify()

            except Exception:

                corrupted.append(image_path)

    if len(corrupted) == 0:

        print("No corrupted images found.")

    else:

        print(
            f"Corrupted images found: "
            f"{len(corrupted)}"
        )

        for image in corrupted:
            print(image)


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":

    check_directory(
        PROCESSED_DIR,
        "PROCESSED DATASET"
    )

    check_directory(
        TRAIN_DIR,
        "TRAIN DATASET"
    )

    check_directory(
        VAL_DIR,
        "VALIDATION DATASET"
    )

    check_directory(
        TEST_DIR,
        "TEST DATASET"
    )

    check_corrupted_images()

    print("\n===================================")
    print("DATASET CHECK COMPLETED")
    print("===================================")