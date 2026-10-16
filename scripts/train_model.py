from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt


# ==========================================================
# SETTINGS
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT_DIR / "dataset" / "train"
VAL_DIR = ROOT_DIR / "dataset" / "val"

MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
GRAPH_DIR = OUTPUT_DIR / "graphs"

BEST_MODEL_PATH = MODEL_DIR / "best_model.keras"

FINAL_MODEL_PATH = (
    MODEL_DIR /
    "driver_distraction_cnn.keras"
)

CLASS_NAMES_FILE = ROOT_DIR / "class_names.txt"

TRAINING_LOG = OUTPUT_DIR / "training_log.csv"


IMAGE_SIZE = (150, 150)

BATCH_SIZE = 32

EPOCHS = 30

SEED = 42


# ==========================================================
# CREATE OUTPUT FOLDERS
# ==========================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

GRAPH_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# CHECK GPU
# ==========================================================

print("\n===================================")
print("GPU CHECK")
print("===================================")

gpus = tf.config.list_physical_devices("GPU")

if gpus:

    print("GPU detected:")

    for gpu in gpus:
        print(gpu)

else:

    print("No TensorFlow GPU detected.")
    print("Training will use CPU.")


# ==========================================================
# RANDOM SEED
# ==========================================================

tf.random.set_seed(SEED)


# ==========================================================
# LOAD TRAINING DATA
# ==========================================================

print("\nLoading training dataset...")

train_ds = keras.utils.image_dataset_from_directory(

    TRAIN_DIR,

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="categorical",

    shuffle=True,

    seed=SEED
)


# ==========================================================
# LOAD VALIDATION DATA
# ==========================================================

print("\nLoading validation dataset...")

val_ds = keras.utils.image_dataset_from_directory(

    VAL_DIR,

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="categorical",

    shuffle=False
)


# ==========================================================
# CLASS NAMES
# ==========================================================

class_names = train_ds.class_names

num_classes = len(class_names)

print("\nClasses:")

for index, class_name in enumerate(class_names):

    print(
        f"{index} -> {class_name}"
    )


# Save class order

with open(
    CLASS_NAMES_FILE,
    "w",
    encoding="utf-8"
) as file:

    for class_name in class_names:

        file.write(
            class_name + "\n"
        )


# ==========================================================
# PERFORMANCE OPTIMIZATION
# ==========================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(
    buffer_size=AUTOTUNE
)

val_ds = val_ds.prefetch(
    buffer_size=AUTOTUNE
)


# ==========================================================
# DATA AUGMENTATION
# ==========================================================

data_augmentation = keras.Sequential(
    [

        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.05
        ),

        layers.RandomZoom(
            0.10
        )

    ],
    name="data_augmentation"
)


# ==========================================================
# BUILD CNN MODEL
# ==========================================================

model = keras.Sequential([

    layers.Input(
        shape=(
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        )
    ),

    data_augmentation,

    layers.Rescaling(
        1.0 / 255
    ),

    # --------------------------------
    # Convolution Block 1
    # --------------------------------

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------
    # Convolution Block 2
    # --------------------------------

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------
    # Convolution Block 3
    # --------------------------------

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------
    # Convolution Block 4
    # --------------------------------

    layers.Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------
    # Fully Connected Layers
    # --------------------------------

    layers.Flatten(),

    layers.Dense(
        256,
        activation="relu"
    ),

    layers.Dropout(
        0.5
    ),

    # --------------------------------
    # Output Layer
    # --------------------------------

    layers.Dense(
        num_classes,
        activation="softmax"
    )
])


# ==========================================================
# COMPILE MODEL
# ==========================================================

model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


# ==========================================================
# MODEL SUMMARY
# ==========================================================

print("\n===================================")
print("CNN MODEL")
print("===================================")

model.summary()


# ==========================================================
# CALLBACKS
# ==========================================================

callbacks = [

    keras.callbacks.ModelCheckpoint(

        filepath=BEST_MODEL_PATH,

        monitor="val_accuracy",

        save_best_only=True,

        mode="max",

        verbose=1
    ),

    keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=5,

        restore_best_weights=True,

        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=2,

        min_lr=0.000001,

        verbose=1
    ),

    keras.callbacks.CSVLogger(
        TRAINING_LOG
    )
]


# ==========================================================
# TRAIN MODEL
# ==========================================================

print("\n===================================")
print("TRAINING STARTED")
print("===================================\n")

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=callbacks
)


# ==========================================================
# SAVE FINAL MODEL
# ==========================================================

model.save(
    FINAL_MODEL_PATH
)

print(
    "\nFinal model saved to:"
)

print(
    FINAL_MODEL_PATH
)


# ==========================================================
# ACCURACY GRAPH
# ==========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "Training and Validation Accuracy"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "accuracy.png"
)

plt.close()


# ==========================================================
# LOSS GRAPH
# ==========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Training and Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "loss.png"
)

plt.close()


print("\n===================================")
print("TRAINING COMPLETED")
print("===================================")

print(
    "\nBest model:"
)

print(
    BEST_MODEL_PATH
)

print(
    "\nAccuracy graph:"
)

print(
    GRAPH_DIR / "accuracy.png"
)

print(
    "\nLoss graph:"
)

print(
    GRAPH_DIR / "loss.png"
)