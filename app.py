import os
import numpy as np
from PIL import Image

from flask import Flask, render_template, request, send_from_directory, send_file

import tensorflow as tf
from ultralytics import YOLO
from services.pdf_service import generate_pdf_report
# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# FOLDERS
# ============================================================

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# ============================================================
# MODEL PATHS
# ============================================================

CLASSIFIER_PATH = os.path.join(
    "models",
    "brain_tumor_classifier.keras"
)

YOLO_PATH = os.path.join(
    "models",
    "brain_tumor_yolo_seg.pt"
)


# ============================================================
# LOAD CLASSIFIER
# ============================================================

print("\nLoading brain tumor classifier...")

classifier = tf.keras.models.load_model(
    CLASSIFIER_PATH,
    compile=False
)

print("✅ Classifier loaded")
print("Input :", classifier.input_shape)
print("Output:", classifier.output_shape)


# ============================================================
# LOAD YOLO
# ============================================================

print("\nLoading YOLO segmentation model...")

yolo_model = YOLO(YOLO_PATH)

print("✅ YOLO loaded")
print("Task   :", yolo_model.task)
print("Classes:", yolo_model.names)


CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]

CLASSIFICATION_THRESHOLD = 0.70
YOLO_CONFIDENCE_THRESHOLD = 0.25

# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# MRI UPLOAD
# ============================================================

def run_yolo_segmentation(image_path):

    print("\n" + "=" * 60)
    print("YOLO SEGMENTATION")
    print("=" * 60)

    print("Running YOLO...")

    results = yolo_model.predict(
        source=image_path,
        conf=YOLO_CONFIDENCE_THRESHOLD,
        verbose=False
    )

    result = results[0]

    # --------------------------------------------------
    # No segmentation mask
    # --------------------------------------------------

    if result.masks is None:

        print("❌ No tumor mask detected.")

        return {
            "detected": False,
            "yolo_confidence": 0.0,
            "number_of_masks": 0,
            "tumor_pixels": 0,
            "tumor_area_percent": 0.0,
            "result_image": None
        }

    # --------------------------------------------------
    # YOLO confidence
    # --------------------------------------------------

    confidences = result.boxes.conf.cpu().numpy()

    if len(confidences) == 0:

        print("❌ No YOLO detections.")

        return {
            "detected": False,
            "yolo_confidence": 0.0,
            "number_of_masks": 0,
            "tumor_pixels": 0,
            "tumor_area_percent": 0.0,
            "result_image": None
        }

    best_confidence = float(np.max(confidences))

    # --------------------------------------------------
    # Combine masks
    # --------------------------------------------------

    masks = result.masks.data

    combined_mask = masks.any(dim=0).cpu().numpy()

    tumor_pixels = int(combined_mask.sum())

    mask_height, mask_width = combined_mask.shape

    total_pixels = mask_height * mask_width

    tumor_area_percent = (
        tumor_pixels / total_pixels
    ) * 100

    number_of_masks = len(masks)

    print("Number of masks :", number_of_masks)
    print(
        "YOLO confidence :",
        f"{best_confidence * 100:.2f}%"
    )

    print("Tumor pixels    :", tumor_pixels)

    print(
        "Tumor area      :",
        f"{tumor_area_percent:.2f}%"
    )

    # --------------------------------------------------
    # Create visualization
    # --------------------------------------------------

    image = Image.open(image_path).convert("RGB")

    original = np.array(image)

    mask_image = Image.fromarray(
        (combined_mask * 255).astype(np.uint8)
    )

    mask_image = mask_image.resize(
        image.size
    )

    mask = np.array(mask_image) > 127

    # Red overlay
    overlay = original.copy()

    overlay[mask] = (
        0.5 * overlay[mask]
        + 0.5 * np.array([255, 0, 0])
    )

    overlay = overlay.astype(np.uint8)

    # --------------------------------------------------
    # Save visualization
    # --------------------------------------------------

    result_filename = (
        "result_" +
        os.path.splitext(
            os.path.basename(image_path)
        )[0] +
        ".jpg"
    )

    result_path = os.path.join(
        "results",
        result_filename
    )

    Image.fromarray(overlay).save(
        result_path
    )

    print(
        "Result saved:",
        result_path
    )

    return {
        "detected": True,
        "yolo_confidence": best_confidence,
        "number_of_masks": number_of_masks,
        "tumor_pixels": tumor_pixels,
        "tumor_area_percent": tumor_area_percent,
        "result_image": result_filename
    }

@app.route("/predict", methods=["POST"])
def predict():

    # ==================================================
    # 1. CHECK UPLOAD
    # ==================================================

    if "mri" not in request.files:
        return "No MRI uploaded."

    file = request.files["mri"]

    if file.filename == "":
        return "No MRI selected."

    # ==================================================
    # 2. SAVE MRI
    # ==================================================

    filename = file.filename

    upload_path = os.path.join(
        "uploads",
        filename
    )

    file.save(upload_path)

    print("\n" + "=" * 60)
    print("MRI UPLOADED")
    print("=" * 60)

    print("Filename:", filename)
    print("Path:", upload_path)

    # ==================================================
    # 3. CLASSIFIER
    # ==================================================

    print("\nRunning classifier...")

    image = Image.open(upload_path).convert("RGB")

    print("Original size:", image.size)

    resized = image.resize((224, 224))

    img_array = np.array(
        resized,
        dtype=np.float32
    )

    # IMPORTANT:
    # Your trained classifier expects 0-255
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = classifier.predict(
        img_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(prediction)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    classification_confidence = float(
        prediction[predicted_index]
    )

    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULT")
    print("=" * 60)

    print(
        "Prediction:",
        predicted_class
    )

    print(
        "Confidence:",
        f"{classification_confidence * 100:.2f}%"
    )

    print("\nClass probabilities:")

    print(
        "glioma     :",
        f"{prediction[0] * 100:.2f}%"
    )

    print(
        "meningioma :",
        f"{prediction[1] * 100:.2f}%"
    )

    print(
        "no_tumor   :",
        f"{prediction[2] * 100:.2f}%"
    )

    print(
        "pituitary  :",
        f"{prediction[3] * 100:.2f}%"
    )
    # ==================================================
    # 4. CONFIDENCE CHECK
    # ==================================================

    if classification_confidence < CLASSIFICATION_THRESHOLD:

      report_filename = (
        "report_" +
        os.path.splitext(filename)[0] +
        ".pdf"
    )

      report_path = os.path.join(
        "reports",
        report_filename
    )

      generate_pdf_report(
        output_path=report_path,
        filename=filename,
        prediction="Uncertain",
        classification_confidence=
            classification_confidence * 100,
        probabilities=prediction * 100
    )

      return render_template(
        "results.html",
        filename=filename,
        prediction="Uncertain",
        confidence=classification_confidence * 100,
        probabilities=prediction * 100,
        segmentation_detected=False,
        yolo_confidence=0,
        tumor_area=0,
        number_of_masks=0,
        result_image=None,
        report_filename=report_filename
    )               

    # ==================================================
    # 5. NO TUMOR
    # ==================================================

    if predicted_class == "no_tumor":

        print("\nNo tumor detected.")
        print("YOLO segmentation skipped.")
        report_filename = (
            "report_" +
            os.path.splitext(filename)[0] +
            ".pdf"
        )

        report_path = os.path.join(
            "reports",
            report_filename
        )

        generate_pdf_report(
            output_path=report_path,
            filename=filename,
            prediction=predicted_class,
            classification_confidence=
                classification_confidence * 100,
            probabilities=prediction * 100
        )
        return render_template(
            "results.html",
            filename=filename,
            prediction=predicted_class,
            confidence=classification_confidence * 100,
            probabilities=prediction * 100,
            segmentation_detected=False,
            yolo_confidence=0,
            tumor_area=0,
            number_of_masks=0,
            result_image=None,
            report_filename=report_filename
        )

    # ==================================================
    # 6. YOLO SEGMENTATION
    # ==================================================

    print("\nTumor detected by classifier.")
    print("Running YOLO segmentation...")

    yolo_result = run_yolo_segmentation(
        upload_path
    )

    # ==================================================
    # 7. FINAL RESULT
    # ==================================================

       # ==================================================
    # 7. GENERATE PDF REPORT
    # ==================================================

    report_filename = (
        "report_" +
        os.path.splitext(filename)[0] +
        ".pdf"
    )

    report_path = os.path.join(
        "reports",
        report_filename
    )

    generate_pdf_report(
        output_path=report_path,
        filename=filename,
        prediction=predicted_class,
        classification_confidence=
            classification_confidence * 100,
        probabilities=prediction * 100,
        segmentation_detected=
            yolo_result["detected"],
        yolo_confidence=
            yolo_result["yolo_confidence"] * 100,
        tumor_area=
            yolo_result["tumor_area_percent"],
        number_of_masks=
            yolo_result["number_of_masks"],
        tumor_pixels=
            yolo_result["tumor_pixels"],
        result_image=
            yolo_result["result_image"]
    )

    print(
        "PDF report generated:",
        report_path
    )

    # ==================================================
    # 8. FINAL RESULT
    # ==================================================

    return render_template(
        "results.html",

        filename=filename,

        prediction=predicted_class,

        confidence=classification_confidence * 100,

        probabilities=prediction * 100,

        segmentation_detected=
            yolo_result["detected"],

        yolo_confidence=
            yolo_result["yolo_confidence"] * 100,

        tumor_area=
            yolo_result["tumor_area_percent"],

        tumor_pixels=
            yolo_result["tumor_pixels"],

        number_of_masks=
            yolo_result["number_of_masks"],

        result_image=
            yolo_result["result_image"],

        report_filename=
            report_filename
    )

@app.route("/results/<filename>")
def results_file(filename):
    return send_from_directory(
        "results",
        filename
    )
@app.route("/download-report/<filename>")
def download_report(filename):
    report_path = os.path.join(
        "reports",
        filename
    )

    if not os.path.exists(report_path):
        return "Report not found.", 404

    return send_file(
        report_path,
        as_attachment=True
    )

# ============================================================
# START FLASK
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )