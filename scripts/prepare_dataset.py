from pathlib import Path
import shutil

# ==========================================================
# PROJECT PATHS
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT_DIR / "dataset" / "raw"
PROCESSED_DIR = ROOT_DIR / "dataset" / "processed"

# ==========================================================
# CLASS MAPPING
# ==========================================================

CLASS_MAPPING = {
    "normal": ["c0"],

    "using_phone": [
        "c1",
        "c2",
        "c3",
        "c4"
    ],

    "eating_drinking": ["c6"],

    "reaching_behind": ["c7"],

    "talking_passenger": ["c9"]
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}


# ==========================================================
# CLEAR OLD PROCESSED DATA
# ==========================================================

def clear_folder(folder):
    if folder.exists():

        for item in folder.iterdir():

            if item.is_file():
                item.unlink()

            elif item.is_dir():
                shutil.rmtree(item)


# ==========================================================
# PREPARE DATASET
# ==========================================================

def prepare_dataset():

    print("\n========================================")
    print("DRIVER DISTRACTION DATASET PREPARATION")
    print("========================================\n")

    if not RAW_DIR.exists():
        print("ERROR: Raw dataset folder not found:")
        print(RAW_DIR)
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for new_class, source_classes in CLASS_MAPPING.items():

        destination = PROCESSED_DIR / new_class

        destination.mkdir(parents=True, exist_ok=True)

        # Remove old files
        clear_folder(destination)

        total_copied = 0

        print(f"\nPreparing class: {new_class}")
        print("--------------------------------")

        for source_class in source_classes:

            source_folder = RAW_DIR / source_class

            if not source_folder.exists():

                print(f"WARNING: {source_class} not found.")
                continue

            source_count = 0

            for image_file in source_folder.iterdir():

                if image_file.suffix.lower() in IMAGE_EXTENSIONS:

                    # Add c0/c1/etc. to filename
                    # This prevents duplicate names
                    new_name = f"{source_class}_{image_file.name}"

                    destination_file = destination / new_name

                    shutil.copy2(
                        image_file,
                        destination_file
                    )

                    source_count += 1
                    total_copied += 1

            print(
                f"{source_class}: "
                f"{source_count} images copied"
            )

        print(
            f"Total {new_class}: "
            f"{total_copied} images"
        )

    print("\n========================================")
    print("DATASET PREPARATION COMPLETED")
    print("========================================")


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    prepare_dataset()