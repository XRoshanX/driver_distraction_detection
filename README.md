# Driver Distraction Detection Using Deep Learning

A real-time **Driver Distraction Detection System** built using **Deep Learning, TensorFlow/Keras, CNN, and OpenCV**.

The project trains a **Convolutional Neural Network (CNN) from scratch** to identify different driver behaviors and detect whether the driver is driving normally or is distracted.

---

## Project Objective

Driver distraction is one of the major causes of road accidents.

The objective of this project is to develop a deep-learning system that can classify driver behavior into multiple categories and provide a warning when distracted driving is detected.

The trained model can also be connected to a webcam for real-time driver behavior prediction.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Pillow
- CNN — Convolutional Neural Network

---

## Dataset

This project uses the **State Farm Distracted Driver Detection Dataset** available on Kaggle.

Original dataset classes:

```text
c0 → Safe driving
c1 → Texting - right hand
c2 → Talking on phone - right hand
c3 → Texting - left hand
c4 → Talking on phone - left hand
c5 → Operating radio
c6 → Drinking
c7 → Reaching behind
c8 → Hair and makeup
c9 → Talking to passenger
```

For this project, the original dataset is converted into five classes:

```text
normal
using_phone
eating_drinking
reaching_behind
talking_passenger
```

Class mapping:

```text
c0              → normal

c1 + c2 + c3 + c4
                → using_phone

c6              → eating_drinking

c7              → reaching_behind

c9              → talking_passenger
```

`c5` and `c8` are currently not used.

> Note: The `eating_drinking` class currently mainly contains drinking images from `c6`. Eating images can be added later to improve the class.

---

## Project Folder Structure

```text
driver_distraction_detection/
│
├── dataset/
│   │
│   ├── raw/
│   │   ├── c0/
│   │   ├── c1/
│   │   ├── c2/
│   │   ├── c3/
│   │   ├── c4/
│   │   ├── c5/
│   │   ├── c6/
│   │   ├── c7/
│   │   ├── c8/
│   │   └── c9/
│   │
│   ├── processed/
│   │   ├── normal/
│   │   ├── using_phone/
│   │   ├── eating_drinking/
│   │   ├── reaching_behind/
│   │   └── talking_passenger/
│   │
│   ├── train/
│   │   ├── normal/
│   │   ├── using_phone/
│   │   ├── eating_drinking/
│   │   ├── reaching_behind/
│   │   └── talking_passenger/
│   │
│   ├── val/
│   │   ├── normal/
│   │   ├── using_phone/
│   │   ├── eating_drinking/
│   │   ├── reaching_behind/
│   │   └── talking_passenger/
│   │
│   └── test/
│       ├── normal/
│       ├── using_phone/
│       ├── eating_drinking/
│       ├── reaching_behind/
│       └── talking_passenger/
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── split_dataset.py
│   ├── check_dataset.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict_image.py
│   └── webcam_detection.py
│
├── models/
│   ├── best_model.keras
│   └── driver_distraction_cnn.keras
│
├── test_images/
│
├── outputs/
│   ├── graphs/
│   │   ├── accuracy.png
│   │   └── loss.png
│   │
│   ├── predictions/
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── training_log.csv
│
├── class_names.txt
├── requirements.txt
├── .gitignore
└── README.md
```

---

# CNN Architecture

The model is trained from scratch using a custom CNN architecture.

```text
Input Image
    ↓
Resize to 150 × 150
    ↓
Data Augmentation
    ↓
Rescaling
    ↓
Conv2D - 32 filters
    ↓
MaxPooling2D
    ↓
Conv2D - 64 filters
    ↓
MaxPooling2D
    ↓
Conv2D - 128 filters
    ↓
MaxPooling2D
    ↓
Conv2D - 256 filters
    ↓
MaxPooling2D
    ↓
Flatten
    ↓
Dense - 256 neurons
    ↓
Dropout
    ↓
Dense - 5 classes
    ↓
Softmax
```

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/driver-distraction-detection.git
```

Move into the project directory:

```bash
cd driver-distraction-detection
```

---

## 2. Create a Virtual Environment

Windows:

```bash
python -m venv .venv
```

Activate it using CMD:

```bash
.venv\Scripts\activate
```

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

The required libraries include:

```text
tensorflow
opencv-python
numpy
matplotlib
scikit-learn
pillow
```

---

# Dataset Setup

The dataset is not included in this GitHub repository because of its large size.

Download the State Farm Distracted Driver Detection dataset from Kaggle.

After downloading and extracting it, place the original folders inside:

```text
dataset/raw/
```

The result should look like:

```text
dataset/raw/c0/
dataset/raw/c1/
dataset/raw/c2/
dataset/raw/c3/
dataset/raw/c4/
dataset/raw/c5/
dataset/raw/c6/
dataset/raw/c7/
dataset/raw/c8/
dataset/raw/c9/
```

---

# Running the Project

Run the following programs in order.

## Step 1 — Prepare the Dataset

```bash
python scripts/prepare_dataset.py
```

This converts the original `c0-c9` classes into the five project classes.

---

## Step 2 — Split the Dataset

```bash
python scripts/split_dataset.py
```

The dataset is divided into:

```text
70% → Training
15% → Validation
15% → Testing
```

---

## Step 3 — Check the Dataset

```bash
python scripts/check_dataset.py
```

This program checks:

- Number of images
- Class folders
- Training images
- Validation images
- Testing images
- Corrupted images

---

## Step 4 — Train the CNN

```bash
python scripts/train_model.py
```

During training, the model learns driver behavior from the training images.

The best model is saved as:

```text
models/best_model.keras
```

The final model is saved as:

```text
models/driver_distraction_cnn.keras
```

Training graphs are stored inside:

```text
outputs/graphs/
```

---

# Model Evaluation

Run:

```bash
python scripts/evaluate_model.py
```

The program calculates:

- Test Accuracy
- Test Loss
- Precision
- Recall
- F1-score
- Confusion Matrix

The results are stored inside:

```text
outputs/
```

---

# Predict a Single Image

Place a test image inside:

```text
test_images/
```

Example:

```text
test_images/driver1.jpg
```

Run:

```bash
python scripts/predict_image.py test_images/driver1.jpg
```

Example output:

```text
Prediction : using_phone
Confidence : 95.42%
```

---

# Real-Time Webcam Detection

Run:

```bash
python scripts/webcam_detection.py
```

The application opens the webcam and continuously predicts the driver's behavior.

Example:

```text
Action: normal

Confidence: 96.4%

SAFE DRIVING
```

When distraction is detected:

```text
Action: using_phone

Confidence: 94.1%

WARNING: DISTRACTED
```

Press:

```text
Q
```

to close the webcam.

---

# Project Workflow

```text
Kaggle Dataset
      ↓
dataset/raw/
      ↓
prepare_dataset.py
      ↓
dataset/processed/
      ↓
split_dataset.py
      ↓
Train / Validation / Test
      ↓
train_model.py
      ↓
CNN Model
      ↓
best_model.keras
      ↓
evaluate_model.py
      ↓
Model Evaluation
      ↓
predict_image.py
      ↓
Single Image Prediction
      ↓
webcam_detection.py
      ↓
Real-Time Driver Distraction Detection
```

---

# Output Classes

The model predicts one of the following five classes:

```text
normal
using_phone
eating_drinking
reaching_behind
talking_passenger
```

All classes other than `normal` are considered distracted-driving behavior in the current real-time application.

---

# Future Improvements

The project can be improved by adding:

- Driver-wise train/test splitting
- More driver identities
- More real-world images
- Eating detection
- Looking-left detection
- Looking-right detection
- Drowsiness detection
- Seatbelt detection
- Face detection
- Eye tracking
- Alarm sound when distraction is detected
- Dashboard/mobile application
- Video-file detection
- More robust nighttime detection

---

# Important Dataset Note

The original Kaggle dataset is not stored in this GitHub repository.

Users who want to run the project should download the dataset separately and place the `c0-c9` folders inside:

```text
dataset/raw/
```

This repository mainly contains the source code required to prepare, train, evaluate, and run the deep-learning model.

---

# Author

**Syed Roshan Ali**

Deep Learning Project

**Driver Distraction Detection Using CNN and Computer Vision**