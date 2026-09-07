import os
import numpy as np
from PIL import Image

from flask import (
    Flask, render_template, request,
    send_from_directory, send_file, jsonify
)
from werkzeug.utils import secure_filename

import tensorflow as tf
from ultralytics import YOLO
from services.pdf_service import generate_pdf_report
from services.ocr_service import extract_report
from services.ai_service import analyze_report

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# FOLDERS
# ============================================================

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
REPORT_UPLOAD_FOLDER = "report_uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(REPORT_UPLOAD_FOLDER, exist_ok=True)
os.makedirs("reports", exist_ok=True)


# ============================================================
# ALLOWED EXTENSIONS
# ============================================================

MRI_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}
REPORT_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


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
# HELPERS
# ============================================================

def allowed_file(filename, allowed_extensions):
    """Check if a filename has an allowed extension."""
    if not filename:
        return False
    ext = os.path.splitext(filename)[1].lower()
    return ext in allowed_extensions


def extract_image_from_pdf(pdf_path):
    """
    Extract a single MRI image from a PDF.

    Strategy:
        1. Try to extract embedded raster images.
        2. If exactly one usable image found, return it.
        3. If multiple images found, return None with a message.
        4. If no images found, render the first page as an image.

    Returns:
        (PIL.Image or None, error_message or None)
    """
    import fitz  # pymupdf

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return None, "Unable to read the PDF file."

    if doc.page_count == 0:
        doc.close()
        return None, "The PDF file contains no pages."

    # --------------------------------------------------
    # Try extracting embedded raster images
    # --------------------------------------------------
    extracted_images = []

    for page_num in range(doc.page_count):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_info in image_list:
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                if base_image and base_image.get("image"):
                    img_bytes = base_image["image"]
                    from io import BytesIO
                    pil_img = Image.open(BytesIO(img_bytes))

                    # Only consider images large enough to be MRI
                    w, h = pil_img.size
                    if w >= 64 and h >= 64:
                        extracted_images.append(pil_img)
            except Exception:
                continue

    doc_for_render = doc  # keep open for potential render

    # --------------------------------------------------
    # Evaluate extracted images
    # --------------------------------------------------
    if len(extracted_images) == 1:
        doc_for_render.close()
        return extracted_images[0].convert("RGB"), None

    if len(extracted_images) > 1:
        doc_for_render.close()
        return None, (
            "This PDF contains multiple image candidates. "
            "Please upload the MRI image directly as "
            "JPG, JPEG, or PNG."
        )

    # --------------------------------------------------
    # No embedded images — render first page
    # --------------------------------------------------
    try:
        page = doc_for_render[0]
        matrix = fitz.Matrix(3, 3)  # 3x resolution
        pixmap = page.get_pixmap(matrix=matrix)

        img_data = np.frombuffer(
            pixmap.samples,
            dtype=np.uint8
        )

        if pixmap.n == 4:
            img_data = img_data.reshape(
                pixmap.height, pixmap.width, 4
            )
            pil_img = Image.fromarray(img_data[:, :, :3])
        else:
            img_data = img_data.reshape(
                pixmap.height, pixmap.width, 3
            )
            pil_img = Image.fromarray(img_data)

        doc_for_render.close()
        return pil_img.convert("RGB"), None

    except Exception:
        doc_for_render.close()
        return None, "Unable to render MRI image from the PDF."


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# YOLO SEGMENTATION (existing — unchanged)
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


# ============================================================
# MRI ANALYSIS — POST /predict (existing pipeline preserved)
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    # ==================================================
    # 1. CHECK UPLOAD
    # ==================================================

    if "mri" not in request.files:
        return "Please upload a supported MRI file: JPG, JPEG, PNG, or PDF."

    file = request.files["mri"]

    if file.filename == "":
        return "No MRI selected."

    # ==================================================
    # 1.5 VALIDATE FILE
    # ==================================================

    original_filename = file.filename
    filename = secure_filename(original_filename)

    if not filename:
        return "Invalid filename."

    if not allowed_file(filename, MRI_ALLOWED_EXTENSIONS):
        return "Please upload a supported MRI file: JPG, JPEG, PNG, or PDF."

    # ==================================================
    # 2. SAVE MRI
    # ==================================================

    upload_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    file.save(upload_path)

    print("\n" + "=" * 60)
    print("MRI UPLOADED")
    print("=" * 60)

    print("Filename:", filename)
    print("Path:", upload_path)

    # ==================================================
    # 2.5 PDF — EXTRACT MRI IMAGE
    # ==================================================

    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        print("\nPDF detected — extracting MRI image...")

        pil_image, error_msg = extract_image_from_pdf(
            upload_path
        )

        if pil_image is None:
            return error_msg or (
                "Unable to extract an MRI image from the PDF."
            )

        # Save extracted image as PNG for the pipeline
        extracted_filename = (
            os.path.splitext(filename)[0] + "_extracted.png"
        )
        extracted_path = os.path.join(
            UPLOAD_FOLDER,
            extracted_filename
        )
        pil_image.save(extracted_path)

        # Use extracted image for the rest of the pipeline
        upload_path = extracted_path
        filename = extracted_filename

        print("Extracted MRI image:", extracted_path)

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


# ============================================================
# REPORT ANALYSIS — POST /analyze-report (independent)
# ============================================================

@app.route("/analyze-report", methods=["POST"])
def analyze_report_route():

    # ==================================================
    # 1. CHECK UPLOAD
    # ==================================================

    if "report" not in request.files:
        return render_template(
            "report_results.html",
            error="Please upload a supported medical report: PDF, JPG, JPEG, or PNG."
        )

    file = request.files["report"]

    if file.filename == "":
        return render_template(
            "report_results.html",
            error="No file selected."
        )

    # ==================================================
    # 2. VALIDATE FILE
    # ==================================================

    original_filename = file.filename
    filename = secure_filename(original_filename)

    if not filename:
        return render_template(
            "report_results.html",
            error="Invalid filename."
        )

    if not allowed_file(filename, REPORT_ALLOWED_EXTENSIONS):
        return render_template(
            "report_results.html",
            error="Please upload a supported medical report: PDF, JPG, JPEG, or PNG."
        )

    # ==================================================
    # 3. SAVE FILE
    # ==================================================

    upload_path = os.path.join(
        REPORT_UPLOAD_FOLDER,
        filename
    )

    file.save(upload_path)

    print("\n" + "=" * 60)
    print("REPORT UPLOADED")
    print("=" * 60)

    print("Filename:", filename)
    print("Path:", upload_path)

    # ==================================================
    # 4. OCR + NLP ANALYSIS
    # ==================================================

    try:
        print("\nExtracting report text...")

        report_data = extract_report(upload_path)

        raw_text = report_data.get("raw_text", "")

        print("OCR text extracted.")
        print("Characters:", len(raw_text))

        if not raw_text or len(raw_text.strip()) < 10:
            return render_template(
                "report_results.html",
                filename=filename,
                error="We could not extract readable text from this document. Please try a clearer scan."
            )

        print("\nRunning NLP analysis...")

        analysis = analyze_report(report_data)

        print("Document type:", analysis.get("document_type"))
        print("Diagnosis:", analysis.get("diagnosis"))
        print("Report analysis completed.")

    except Exception as e:
        print("Report analysis failed:", str(e))

        return render_template(
            "report_results.html",
            filename=filename,
            error=f"An error occurred while processing the document: {str(e)}"
        )

    # ==================================================
    # 5. CLEAN UP TEMPORARY FILES
    # ==================================================

    # Report uploads are kept until the response is sent.
    # They are not model files and can be cleaned up.
    # For now, we keep them for debugging.

    # ==================================================
    # 6. RENDER RESULTS
    # ==================================================

    return render_template(
        "report_results.html",
        filename=filename,
        document_type=analysis.get("document_type", "unknown"),
        raw_text=raw_text,
        analysis=analysis
    )


# ============================================================
# EXISTING ROUTES (preserved)
# ============================================================

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