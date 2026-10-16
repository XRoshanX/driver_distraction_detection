from pathlib import Path
import sys

import numpy as np
import cv2

from tensorflow import keras


# ==========================================================
# SETTINGS
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT_DIR /
    "models" /
    "best_model.keras"
)

CLASS_NAMES_FILE = (
    ROOT_DIR /
    "class_names.txt"
)

OUTPUT_DIR = (
    ROOT_DIR /
    "outputs" /
    "predictions"
)

IMAGE_SIZE = (150, 150)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# LOAD CLASS NAMES
# ==========================================================

with open(
    CLASS_NAMES_FILE,
    "r",
    encoding="utf-8"
) as file:

    class_names = [
        line.strip()
        for line in file
        if line.strip()
    ]


# ==========================================================
# LOAD MODEL
# ==========================================================

print("Loading model...")

model = keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ==========================================================
# GET IMAGE PATH
# ==========================================================

if len(sys.argv) > 1:

    image_path = Path(
        sys.argv[1]
    )

else:

    image_path = Path(
        input(
            "\nEnter image path: "
        ).strip()
    )


if not image_path.exists():

    print(
        "\nERROR: Image not found:"
    )

    print(image_path)

    sys.exit()


# ==========================================================
# LOAD IMAGE FOR MODEL
# ==========================================================

img = keras.utils.load_img(

    image_path,

    target_size=IMAGE_SIZE
)

img_array = keras.utils.img_to_array(
    img
)

img_array = np.expand_dims(
    img_array,
    axis=0
)


# ==========================================================
# PREDICT
# ==========================================================

predictions = model.predict(
    img_array,
    verbose=0
)

predicted_index = np.argmax(
    predictions[0]
)

predicted_class = class_names[
    predicted_index
]

confidence = predictions[0][
    predicted_index
] * 100


# ==========================================================
# SHOW RESULT
# ==========================================================

print("\n===================================")
print("PREDICTION RESULT")
print("===================================")

print(
    f"Prediction : "
    f"{predicted_class}"
)

print(
    f"Confidence : "
    f"{confidence:.2f}%"
)


# ==========================================================
# SHOW ALL CLASS PROBABILITIES
# ==========================================================

print("\nClass probabilities:")

for class_name, probability in zip(
    class_names,
    predictions[0]
):

    print(
        f"{class_name:<25} "
        f"{probability * 100:.2f}%"
    )


# ==========================================================
# SAVE ANNOTATED IMAGE
# ==========================================================

original = cv2.imread(
    str(image_path)
)

if original is not None:

    text = (
        f"{predicted_class}: "
        f"{confidence:.1f}%"
    )

    cv2.putText(

        original,

        text,

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        (0, 255, 0),

        2
    )

    output_path = (
        OUTPUT_DIR /
        f"predicted_{image_path.name}"
    )

    cv2.imwrite(
        str(output_path),
        original
    )

    print(
        "\nAnnotated image saved to:"
    )

    print(output_path)