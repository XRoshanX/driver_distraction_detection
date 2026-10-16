from pathlib import Path
from collections import deque

import cv2
import numpy as np

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

IMAGE_SIZE = (150, 150)

SMOOTHING_FRAMES = 10


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

print("\nLoading model...")

model = keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ==========================================================
# PREDICTION BUFFER
# ==========================================================

prediction_buffer = deque(
    maxlen=SMOOTHING_FRAMES
)


# ==========================================================
# OPEN WEBCAM
# ==========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print(
        "ERROR: Cannot access webcam."
    )

    exit()


print("\n===================================")
print("WEBCAM STARTED")
print("===================================")

print(
    "Press Q to exit."
)


# ==========================================================
# MAIN LOOP
# ==========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # --------------------------------
    # Mirror view
    # --------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------
    # Convert BGR -> RGB
    # --------------------------------

    rgb_frame = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2RGB
    )


    # --------------------------------
    # Resize
    # --------------------------------

    resized = cv2.resize(

        rgb_frame,

        IMAGE_SIZE
    )


    # --------------------------------
    # Prepare for CNN
    # --------------------------------

    input_image = np.expand_dims(

        resized.astype(
            "float32"
        ),

        axis=0
    )


    # --------------------------------
    # Prediction
    # --------------------------------

    prediction = model.predict(

        input_image,

        verbose=0
    )[0]


    # --------------------------------
    # Smooth predictions
    # --------------------------------

    prediction_buffer.append(
        prediction
    )

    average_prediction = np.mean(

        prediction_buffer,

        axis=0
    )


    predicted_index = np.argmax(
        average_prediction
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = (
        average_prediction[
            predicted_index
        ] * 100
    )


    # --------------------------------
    # Determine status
    # --------------------------------

    if predicted_class == "normal":

        status = "SAFE DRIVING"

        color = (
            0,
            255,
            0
        )

    else:

        status = "WARNING: DISTRACTED"

        color = (
            0,
            0,
            255
        )


    # --------------------------------
    # Display prediction
    # --------------------------------

    cv2.putText(

        frame,

        f"Action: {predicted_class}",

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.9,

        color,

        2
    )


    cv2.putText(

        frame,

        f"Confidence: {confidence:.1f}%",

        (20, 80),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        color,

        2
    )


    cv2.putText(

        frame,

        status,

        (20, 125),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        color,

        3
    )


    # --------------------------------
    # Show webcam
    # --------------------------------

    cv2.imshow(

        "Driver Distraction Detection",

        frame
    )


    # --------------------------------
    # Press Q to exit
    # --------------------------------

    if (
        cv2.waitKey(1) & 0xFF
    ) == ord("q"):

        break


# ==========================================================
# CLEANUP
# ==========================================================

cap.release()

cv2.destroyAllWindows()

print(
    "\nWebcam closed successfully."
)