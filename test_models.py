import os
import tensorflow as tf
from ultralytics import YOLO


# ============================================================
# MODEL PATHS
# ============================================================

CLASSIFIER_PATH = "models/brain_tumor_classifier.keras"
YOLO_PATH = "models/brain_tumor_yolo_seg.pt"


print("=" * 60)
print("       TESTING BRAIN TUMOR AI MODELS")
print("=" * 60)


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking model files...")

print(
    "Classifier exists:",
    os.path.exists(CLASSIFIER_PATH)
)

print(
    "YOLO exists:",
    os.path.exists(YOLO_PATH)
)


if not os.path.exists(CLASSIFIER_PATH):
    raise FileNotFoundError(
        f"Classifier not found: {CLASSIFIER_PATH}"
    )

if not os.path.exists(YOLO_PATH):
    raise FileNotFoundError(
        f"YOLO model not found: {YOLO_PATH}"
    )


# ============================================================
# LOAD CLASSIFIER
# ============================================================

print("\nLoading classifier...")

classifier = tf.keras.models.load_model(
    CLASSIFIER_PATH,
    compile=False
)

print("✅ CLASSIFIER LOADED SUCCESSFULLY")

print(
    "Input shape :",
    classifier.input_shape
)

print(
    "Output shape:",
    classifier.output_shape
)


# ============================================================
# LOAD YOLO
# ============================================================

print("\nLoading YOLO segmentation model...")

yolo_model = YOLO(YOLO_PATH)

print("✅ YOLO LOADED SUCCESSFULLY")

print(
    "Task    :",
    yolo_model.task
)

print(
    "Classes :",
    yolo_model.names
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 60)
print("       ALL MODELS LOADED SUCCESSFULLY")
print("=" * 60)