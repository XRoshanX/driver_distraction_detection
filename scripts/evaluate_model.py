from pathlib import Path

import numpy as np

import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# ==========================================================
# SETTINGS
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

TEST_DIR = ROOT_DIR / "dataset" / "test"

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
    "outputs"
)

CONFUSION_MATRIX_PATH = (
    OUTPUT_DIR /
    "confusion_matrix.png"
)

REPORT_PATH = (
    OUTPUT_DIR /
    "classification_report.txt"
)

IMAGE_SIZE = (150, 150)

BATCH_SIZE = 32


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
# LOAD TEST DATASET
# ==========================================================

test_ds = keras.utils.image_dataset_from_directory(

    TEST_DIR,

    class_names=class_names,

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="categorical",

    shuffle=False
)


# ==========================================================
# LOAD MODEL
# ==========================================================

print("\nLoading model...")

model = keras.models.load_model(
    MODEL_PATH
)


# ==========================================================
# TEST LOSS AND ACCURACY
# ==========================================================

print("\n===================================")
print("MODEL EVALUATION")
print("===================================")

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print(
    f"\nTest Loss     : "
    f"{test_loss:.4f}"
)

print(
    f"Test Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)


# ==========================================================
# GET TRUE LABELS
# ==========================================================

y_true = []

for images, labels in test_ds:

    labels = np.argmax(
        labels.numpy(),
        axis=1
    )

    y_true.extend(labels)

y_true = np.array(y_true)


# ==========================================================
# MODEL PREDICTIONS
# ==========================================================

predictions = model.predict(
    test_ds,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

report = classification_report(

    y_true,

    y_pred,

    target_names=class_names,

    zero_division=0
)

print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================\n")

print(report)


# Save report

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        f"Test Loss: {test_loss:.4f}\n"
    )

    file.write(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%\n\n"
    )

    file.write(report)


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

fig, ax = plt.subplots(
    figsize=(10, 8)
)

display = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=class_names
)

display.plot(
    ax=ax,
    xticks_rotation=45,
    values_format="d"
)

plt.title(
    "Driver Distraction Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH
)

plt.close()


print("\n===================================")
print("EVALUATION COMPLETED")
print("===================================")

print(
    "\nReport saved to:"
)

print(
    REPORT_PATH
)

print(
    "\nConfusion matrix saved to:"
)

print(
    CONFUSION_MATRIX_PATH
)