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
# EDUCATION HUB
# ============================================================

@app.route("/education")
def education():

    return render_template("education.html")


# ============================================================
# EDUCATION — TUMOR TYPE DETAIL
# ============================================================

TUMOR_EDUCATION = {

    "glioma": {
        "name": "Glioma",
        "tagline": "Understanding Your Glioma Diagnosis",
        "color_class": "glioma",
        "who_grade": "WHO Grade I–IV",
        "overview": (
            "Gliomas are tumors that arise from glial cells — the supportive cells of the nervous system. "
            "They are the most common type of primary brain tumor, accounting for about 33% of all brain tumors. "
            "Gliomas range from slow-growing, low-grade tumors (Grade I–II) to highly aggressive, "
            "fast-growing tumors like Glioblastoma Multiforme (Grade IV)."
        ),
        "subtypes": [
            {"name": "Astrocytoma", "desc": "Arises from astrocytes. Can be low-grade (Grade I–II) or high-grade (Grade III–IV)."},
            {"name": "Oligodendroglioma", "desc": "Develops from oligodendrocytes. Often responds well to chemotherapy."},
            {"name": "Glioblastoma (GBM)", "desc": "The most aggressive form (Grade IV). Most common malignant brain tumor in adults."},
            {"name": "Ependymoma", "desc": "Arises from ependymal cells lining the ventricles. More common in children."},
        ],
        "symptoms": [
            {"name": "Persistent Headaches", "detail": "Often worse in the morning or after lying down. May be different from usual headaches."},
            {"name": "Seizures", "detail": "New-onset seizures are a common first symptom, especially in low-grade gliomas."},
            {"name": "Cognitive Changes", "detail": "Memory loss, confusion, difficulty concentrating, personality changes."},
            {"name": "Weakness or Numbness", "detail": "Often on one side of the body, depending on tumor location."},
            {"name": "Speech Difficulties", "detail": "Trouble speaking, understanding, or finding words (if tumor is in language areas)."},
            {"name": "Vision Problems", "detail": "Blurred vision, double vision, or visual field loss."},
            {"name": "Nausea & Vomiting", "detail": "Caused by increased intracranial pressure from the growing tumor."},
            {"name": "Balance Problems", "detail": "Difficulty walking, dizziness, or loss of coordination."},
        ],
        "precautions": [
            "Attend all scheduled follow-up MRI scans — early detection of changes is critical",
            "Take prescribed anti-seizure medications exactly as directed, even if seizure-free",
            "Keep a symptom diary to track any new or worsening symptoms between appointments",
            "Avoid activities that could cause head injury (contact sports, extreme sports)",
            "Inform your medical team immediately if you experience new seizures, sudden headaches, or neurological changes",
            "Carry a medical ID card or bracelet noting your condition and medications",
        ],
        "dos": [
            "Follow your neurosurgeon's and oncologist's treatment plan consistently",
            "Maintain a balanced diet rich in antioxidants (fruits, vegetables, whole grains)",
            "Stay physically active within your doctor's guidelines — gentle exercise can improve mood and energy",
            "Get adequate sleep (7–9 hours) — the brain heals during rest",
            "Seek support from brain tumor support groups and counseling services",
            "Ask your medical team about clinical trials — new treatments are constantly being developed",
            "Practice stress-reduction techniques (meditation, deep breathing, yoga)",
            "Stay hydrated and maintain a healthy weight",
        ],
        "donts": [
            "Do NOT skip medications or adjust doses without consulting your doctor",
            "Do NOT ignore new or worsening symptoms — report them immediately",
            "Do NOT consume alcohol excessively — it can interact with medications and affect brain function",
            "Do NOT drive if you are experiencing seizures (follow local driving regulations)",
            "Do NOT rely solely on alternative/complementary treatments — use them alongside conventional medicine",
            "Do NOT isolate yourself — maintain social connections for mental health",
            "Do NOT make major decisions while on high-dose steroids (they can affect judgment and mood)",
        ],
        "treatment_overview": (
            "Treatment depends on the tumor's grade, size, location, and molecular markers. "
            "Options include surgery (maximum safe resection), radiation therapy, and chemotherapy "
            "(Temozolomide is standard for high-grade gliomas). Emerging treatments include "
            "tumor treating fields (TTFields), immunotherapy, and targeted molecular therapies."
        ),
    },

    "meningioma": {
        "name": "Meningioma",
        "tagline": "Understanding Your Meningioma Diagnosis",
        "color_class": "meningioma",
        "who_grade": "WHO Grade I–III (mostly Grade I)",
        "overview": (
            "Meningiomas develop from the meninges — the protective membranes surrounding the brain and spinal cord. "
            "They are the most common benign brain tumor, representing about 37% of all primary brain tumors. "
            "The vast majority (~80%) are Grade I (benign) and grow slowly over years. "
            "They are more common in women and typically diagnosed between ages 40–70."
        ),
        "subtypes": [
            {"name": "Grade I (Benign)", "desc": "Slow-growing, well-defined borders. Most common type (~80%). Often curable with surgery."},
            {"name": "Grade II (Atypical)", "desc": "Faster growth, higher recurrence rate. Accounts for ~15–20% of meningiomas."},
            {"name": "Grade III (Malignant)", "desc": "Rare (~2%), aggressive, can invade brain tissue. Requires aggressive treatment."},
        ],
        "symptoms": [
            {"name": "Gradual Headaches", "detail": "Slowly worsening headaches over months or years as the tumor grows."},
            {"name": "Vision Changes", "detail": "Blurred or double vision, especially if the tumor is near the optic nerves."},
            {"name": "Hearing Loss", "detail": "Ringing in ears or hearing loss if tumor is near the auditory structures."},
            {"name": "Memory Difficulties", "detail": "Subtle cognitive changes that may develop gradually."},
            {"name": "Weakness in Limbs", "detail": "Progressive weakness, usually on one side, if tumor compresses motor areas."},
            {"name": "Seizures", "detail": "Can occur when the tumor irritates the brain surface."},
            {"name": "Loss of Smell", "detail": "If the meningioma is located near the olfactory groove."},
        ],
        "precautions": [
            "Follow up with regular MRI scans as recommended by your neurosurgeon",
            "Small, asymptomatic meningiomas may only need monitoring ('watch and wait')",
            "Report any new or progressive neurological symptoms promptly",
            "Discuss hormone therapy and birth control with your doctor (some meningiomas have hormone receptors)",
            "Protect your head from injury — wear helmets during cycling or similar activities",
        ],
        "dos": [
            "Attend all scheduled imaging and follow-up appointments consistently",
            "Maintain a healthy lifestyle with regular exercise and balanced nutrition",
            "Discuss surgical options thoroughly with your neurosurgeon — most Grade I tumors are curable",
            "Seek a second opinion if unsure about the treatment plan",
            "Join meningioma-specific support communities for shared experiences",
            "Practice mindfulness and relaxation techniques to manage anxiety",
            "Stay informed about your condition but use reputable medical sources",
        ],
        "donts": [
            "Do NOT panic — most meningiomas are benign and treatable",
            "Do NOT skip follow-up MRI scans, even if you feel fine",
            "Do NOT take hormone supplements or birth control pills without discussing with your neuro-oncologist",
            "Do NOT ignore gradually worsening symptoms — 'slow' doesn't mean 'not important'",
            "Do NOT self-diagnose using the internet — always consult qualified professionals",
            "Do NOT delay seeking medical attention if you develop sudden severe headache or seizures",
        ],
        "treatment_overview": (
            "Many meningiomas can be cured with complete surgical removal. Small, asymptomatic tumors "
            "may be monitored with regular MRI scans. Stereotactic radiosurgery (Gamma Knife) is an option "
            "for small tumors or surgical remnants. Atypical and malignant meningiomas may require "
            "radiation therapy after surgery. Recurrence is possible, making long-term follow-up essential."
        ),
    },

    "pituitary": {
        "name": "Pituitary Tumor",
        "tagline": "Understanding Your Pituitary Tumor Diagnosis",
        "color_class": "pituitary",
        "who_grade": "Usually benign (adenoma)",
        "overview": (
            "Pituitary tumors develop in the pituitary gland — a pea-sized gland at the base of the brain "
            "that controls many important hormones. Pituitary adenomas account for about 16% of primary brain tumors. "
            "The vast majority are benign and treatable. They are classified by size "
            "(microadenoma <10mm, macroadenoma ≥10mm) and whether they produce excess hormones."
        ),
        "subtypes": [
            {"name": "Prolactinoma", "desc": "Most common type. Produces excess prolactin. Often treated with medication alone."},
            {"name": "Growth Hormone Adenoma", "desc": "Causes acromegaly in adults or gigantism in children."},
            {"name": "ACTH-producing Adenoma", "desc": "Causes Cushing's disease (excess cortisol production)."},
            {"name": "Non-functioning Adenoma", "desc": "Does not produce hormones but can cause symptoms by pressing on nearby structures."},
            {"name": "TSH-producing Adenoma", "desc": "Rare. Causes hyperthyroidism by producing excess TSH."},
        ],
        "symptoms": [
            {"name": "Vision Changes", "detail": "Loss of peripheral vision (bitemporal hemianopia) from pressure on the optic chiasm."},
            {"name": "Persistent Headaches", "detail": "Caused by the expanding tumor pressing on surrounding structures."},
            {"name": "Hormonal Imbalances", "detail": "Irregular periods, infertility, unexpected lactation, weight changes, fatigue."},
            {"name": "Fatigue & Weakness", "detail": "Low energy levels, often caused by hormonal deficiencies."},
            {"name": "Mood Changes", "detail": "Depression, anxiety, or irritability related to hormonal imbalances."},
            {"name": "Sexual Dysfunction", "detail": "Low libido, erectile dysfunction, or menstrual irregularities."},
        ],
        "precautions": [
            "Have complete hormonal blood tests before and after treatment",
            "Monitor vision regularly with an ophthalmologist (visual field testing)",
            "Take hormone replacement medications exactly as prescribed if your pituitary function is affected",
            "Carry a steroid emergency card if you are on cortisol replacement (adrenal insufficiency risk)",
            "Inform all your doctors (dentist, surgeon) about your pituitary condition before any procedure",
            "Women: discuss fertility plans with your endocrinologist before treatment",
        ],
        "dos": [
            "Work closely with an endocrinologist for ongoing hormone management",
            "Take hormone replacement medications at the same time every day",
            "Get regular eye exams to monitor vision, especially peripheral vision",
            "Learn about your specific tumor type and its hormonal effects",
            "Maintain consistent meal schedules if you have cortisol-related issues",
            "Wear a medical alert bracelet if you have adrenal insufficiency",
            "Stay physically active — exercise helps manage weight and mood changes from hormonal shifts",
            "Keep all follow-up endocrinology and imaging appointments",
        ],
        "donts": [
            "Do NOT stop hormone medications suddenly — this can cause life-threatening adrenal crisis",
            "Do NOT ignore changes in vision, even subtle ones — report them immediately",
            "Do NOT skip blood tests — hormonal monitoring is essential for safe treatment",
            "Do NOT assume all symptoms are from the tumor — some may be medication side effects",
            "Do NOT delay seeking emergency care if you experience sudden severe headache with vision loss (pituitary apoplexy)",
            "Do NOT take over-the-counter supplements without consulting your endocrinologist",
        ],
        "treatment_overview": (
            "Treatment depends on the tumor type. Prolactinomas often respond to medication alone "
            "(cabergoline or bromocriptine). Other types may require transsphenoidal surgery (through the nose). "
            "Radiation therapy is reserved for residual or recurrent tumors. Hormone replacement therapy "
            "may be needed lifelong if pituitary function is compromised."
        ),
    },

    "no_tumor": {
        "name": "No Tumor Detected",
        "tagline": "Understanding Your Normal Brain MRI Result",
        "color_class": "notumor",
        "who_grade": "No abnormality detected by AI",
        "overview": (
            "The AI classifier did not detect patterns consistent with glioma, meningioma, or pituitary tumors "
            "in your MRI scan. This is a positive finding, but it is important to understand that this AI system "
            "only screens for three specific tumor types. Other neurological conditions, smaller lesions, "
            "or rare tumor types may not be detected by this model."
        ),
        "subtypes": [],
        "symptoms": [
            {"name": "If You Still Have Symptoms", "detail": "A 'no tumor' AI result does NOT rule out all conditions. Consult your neurologist for thorough evaluation."},
            {"name": "Persistent Headaches", "detail": "Chronic headaches have many causes (migraine, tension, cluster). A specialist can help identify the cause."},
            {"name": "Other Neurological Symptoms", "detail": "Dizziness, tingling, weakness can indicate various conditions beyond tumors."},
        ],
        "precautions": [
            "This AI result is preliminary — always confirm with a qualified radiologist and neurologist",
            "The system only detects glioma, meningioma, and pituitary tumors — other conditions may exist",
            "Continue investigating symptoms with your healthcare provider even with a 'no tumor' AI result",
            "Follow up with your doctor if symptoms persist or worsen",
        ],
        "dos": [
            "Share this result with your neurologist for professional interpretation",
            "Continue routine health checkups as recommended by your doctor",
            "Maintain a brain-healthy lifestyle: regular exercise, balanced diet, adequate sleep",
            "Manage stress through relaxation, social activities, and hobbies",
            "Protect your brain: wear helmets, avoid excessive alcohol, don't smoke",
            "Stay mentally active with puzzles, reading, learning new skills",
            "Keep a log of any symptoms for your doctor's reference",
        ],
        "donts": [
            "Do NOT interpret this AI result as a definitive medical diagnosis",
            "Do NOT stop seeking medical attention for persistent symptoms based on this result alone",
            "Do NOT assume 'no tumor detected' means 'no health issue' — many conditions look normal on MRI",
            "Do NOT share this result as a medical clearance — it is an AI-assisted screening tool only",
        ],
        "treatment_overview": (
            "If the AI analysis shows no tumor, no specific tumor treatment is needed. However, "
            "if you are experiencing symptoms, your doctor may recommend additional imaging (contrast-enhanced MRI, "
            "CT scan), blood tests, neurological examination, or referral to a specialist. "
            "Regular health monitoring and a brain-healthy lifestyle are always recommended."
        ),
    },
}


@app.route("/education/<tumor_type>")
def education_detail(tumor_type):

    # Normalize the tumor type
    tumor_key = tumor_type.lower().replace("-", "_").strip()

    data = TUMOR_EDUCATION.get(tumor_key)

    if data is None:
        return render_template(
            "education.html"
        )

    return render_template(
        "education_detail.html",
        tumor=data,
        tumor_type=tumor_key
    )


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