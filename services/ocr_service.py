import os
import re

import cv2
import numpy as np
import pytesseract
import pymupdf


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

# Windows default
TESSERACT_PATH_WIN = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# macOS Homebrew default
TESSERACT_PATH_MAC = "/opt/homebrew/bin/tesseract"

# macOS Intel Homebrew
TESSERACT_PATH_MAC_INTEL = "/usr/local/bin/tesseract"

if os.path.exists(TESSERACT_PATH_WIN):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH_WIN
elif os.path.exists(TESSERACT_PATH_MAC):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH_MAC
elif os.path.exists(TESSERACT_PATH_MAC_INTEL):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH_MAC_INTEL


def _tesseract_available():
    """Check if Tesseract OCR is available on this system."""
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


TESSERACT_INSTALLED = _tesseract_available()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean OCR/extracted text while preserving useful
    medical report information.
    """

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# NORMAL PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(pdf_path):
    """
    Extract selectable text from a normal text-based PDF.
    """

    document = pymupdf.open(pdf_path)

    pages = []

    for page in document:

        text = page.get_text("text")

        if text:
            pages.append(text)

    document.close()

    return clean_text("\n".join(pages))


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    """
    Prepare an image for OCR.

    Steps:
        1. Convert to grayscale
        2. Normalize contrast
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return gray


# ============================================================
# IMAGE OCR
# ============================================================

def extract_text_from_image(image_path):
    """
    Extract text from an image using Tesseract OCR.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    processed = preprocess_image(image)

    text = pytesseract.image_to_string(
        processed,
        config="--psm 6"
    )

    return clean_text(text)


# ============================================================
# SCANNED PDF OCR
# ============================================================

def extract_text_from_scanned_pdf(pdf_path):
    """
    Convert PDF pages to high-resolution images
    and perform OCR on each page.

    Rendering:
        4x resolution

    OCR:
        Tesseract PSM 6
    """

    document = pymupdf.open(pdf_path)

    pages = []

    for page in document:

        # ----------------------------------------------------
        # Render page at 4x resolution
        # ----------------------------------------------------

        matrix = pymupdf.Matrix(4, 4)

        pixmap = page.get_pixmap(
            matrix=matrix
        )

        # ----------------------------------------------------
        # Convert Pixmap to NumPy array
        # ----------------------------------------------------

        image = np.frombuffer(
            pixmap.samples,
            dtype=np.uint8
        )

        channels = pixmap.n

        # ----------------------------------------------------
        # RGBA
        # ----------------------------------------------------

        if channels == 4:

            image = image.reshape(
                pixmap.height,
                pixmap.width,
                4
            )

            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGBA2BGR
            )

        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        else:

            image = image.reshape(
                pixmap.height,
                pixmap.width,
                3
            )

            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR
            )

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        processed = preprocess_image(image)

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        text = pytesseract.image_to_string(
            processed,
            config="--psm 6"
        )

        if text:
            pages.append(text)

    document.close()

    return clean_text(
        "\n".join(pages)
    )


# ============================================================
# REMOVE REPEATED HOSPITAL HEADER
# ============================================================

def remove_repeated_header_from_impression(text):
    """
    Remove repeated hospital/patient header that can appear
    in the middle of an impression because of a PDF page break.
    """

    if not text:
        return ""

    text = text.strip()

    header_start = re.search(
        r"\n\s*Name\b",
        text,
        re.IGNORECASE
    )

    if header_start is None:
        return clean_text(text)

    start_index = header_start.start()

    header_end = re.search(
        r"(?:DEPARTEMENT|DEPARTMENT)\s+OF\s+IMAG(?:ING|IN)",
        text[header_start.start():],
        re.IGNORECASE
    )

    if header_end is None:
        return clean_text(text)

    end_index = (
        header_start.start()
        + header_end.end()
    )

    before_header = text[:start_index].strip()
    after_header = text[end_index:].strip()

    before_header = re.sub(
        r"\n(?:Glad,|s\s*\*?\s*Add:|Ee\s+Wekvwv|"
        r"Medical C\.|ecical Corrpptex|Laneilines:).*",
        "",
        before_header,
        flags=re.IGNORECASE | re.DOTALL
    ).strip()

    combined = ""

    if before_header:
        combined += before_header

    if after_header:

        if combined:
            combined += "\n"

        combined += after_header

    return clean_text(combined)


# ============================================================
# REMOVE REPORT FOOTER
# ============================================================

def remove_report_footer(text):
    """
    Stop the impression at the radiologist/report signature.
    """

    if not text:
        return ""

    footer_match = re.search(
        r"\n\s*(?:"
        r"Dr\.?\s+Baryalai"
        r"|Dr\.?\s+Barylai"
        r"|Clinical\s+Associate"
        r"|Report\s+status"
        r")",
        text,
        re.IGNORECASE
    )

    if footer_match:

        text = text[
            :footer_match.start()
        ]

    return clean_text(text)


# ============================================================
# REPORT SECTION EXTRACTION
# ============================================================

def extract_report_sections(text):
    """
    Extract important sections from a medical report.

    Returns:
        dict:
            {
                "diagnosis": str,
                "history": str,
                "findings": str,
                "impression": str
            }
    """

    sections = {
        "diagnosis": "",
        "history": "",
        "findings": "",
        "impression": ""
    }

    if not text:
        return sections

    text = clean_text(text)

    if not text:
        return sections

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)


    # ========================================================
    # DIAGNOSIS
    # ========================================================

    diagnosis_patterns = [
        r"FINAL\s+DIAGNOSIS\s*:?\s*(.*?)(?=\n\s*(?:HISTORY|CHIEF\s+COMPLAINT|FINDINGS|REPORT\s+DETAILS|IMPRESSION)\s*:?\s*|\Z)",

        r"DIAGNOSIS\s*:?\s*(.*?)(?=\n\s*(?:HISTORY|CHIEF\s+COMPLAINT|FINDINGS|REPORT\s+DETAILS|IMPRESSION)\s*:?\s*|\Z)"
    ]

    for pattern in diagnosis_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE | re.DOTALL
        )

        if match:
            sections["diagnosis"] = clean_text(
                match.group(1)
            )
            break


    # ========================================================
    # HISTORY
    # ========================================================

    history_patterns = [

        r"CHIEF\s+COMPLAINT\s+AND\s+HISTORY\s*:?\s*"
        r"(.*?)"
        r"(?=\n\s*(?:PERTINENT\s+PHYSICAL\s+FINDINGS|"
        r"SEQUENCES|REPORT\s+DETAILS|FINDINGS|IMPRESSION)"
        r"\s*:?\s*|\Z)",

        r"\bHISTORY\s*:?\s*"
        r"(.*?)"
        r"(?=\n\s*(?:SEQUENCES|REPORT\s+DETAILS|FINDINGS|"
        r"IMPRESSION|FINAL\s+DIAGNOSIS)"
        r"\s*:?\s*|\Z)"
    ]

    for pattern in history_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE | re.DOTALL
        )

        if match:

            history = clean_text(
                match.group(1)
            )

            if history:
                sections["history"] = history
                break


    # ========================================================
    # FINDINGS
    # ========================================================

    findings_patterns = [

        r"\bFINDINGS\s*:?\s*"
        r"(.*?)"
        r"(?=\n\s*IMPRESSION\s*:?\s*|\Z)",

        r"\bREPORT\s+DETAILS\s*:?\s*"
        r"(.*?)"
        r"(?=\n\s*IMPRESSION\s*:?\s*|\Z)",

        r"\bSEQUENCES\s*:?\s*"
        r"(.*?)"
        r"(?=\n\s*IMPRESSION\s*:?\s*|\Z)"
    ]

    for pattern in findings_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE | re.DOTALL
        )

        if match:

            findings = clean_text(
                match.group(1)
            )

            if findings:
                sections["findings"] = findings
                break


    # ========================================================
    # FALLBACK FINDINGS
    # ========================================================

    if not sections["findings"]:

        fallback_patterns = [

            r"\bHISTORY\s*:?.*?\n"
            r"(.*?)"
            r"(?=\n\s*IMPRESSION\s*:?\s*|\Z)",

            r"CHIEF\s+COMPLAINT\s+AND\s+HISTORY\s*:?.*?\n"
            r"(.*?)"
            r"(?=\n\s*IMPRESSION\s*:?\s*|\Z)"
        ]

        for pattern in fallback_patterns:

            match = re.search(
                pattern,
                text,
                re.IGNORECASE | re.DOTALL
            )

            if match:

                findings = clean_text(
                    match.group(1)
                )

                if findings:

                    findings = re.sub(
                        r"^\s*SEQUENCES\s*:?.*?\n",
                        "",
                        findings,
                        flags=re.IGNORECASE | re.DOTALL
                    )

                    findings = clean_text(findings)

                    if findings:
                        sections["findings"] = findings
                        break


    # ========================================================
    # IMPRESSION
    # ========================================================

    impression_match = re.search(
        r"\bIMPRESSION\s*:?\s*(.*)",
        text,
        re.IGNORECASE | re.DOTALL
    )

    if impression_match:

        impression = impression_match.group(1)

        impression = remove_repeated_header_from_impression(
            impression
        )

        impression = remove_report_footer(
            impression
        )

        impression = clean_text(
            impression
        )

        sections["impression"] = impression


    # ========================================================
    # CLEAN ALL SECTIONS
    # ========================================================

    for key in sections:

        if sections[key]:

            sections[key] = clean_text(
                sections[key]
            )

    return sections


# ============================================================
# MAIN REPORT TEXT EXTRACTION
# ============================================================

def extract_report_text(file_path):
    """
    Main report text extraction function.

    Supports:
        PDF
        PNG
        JPG
        JPEG
        TIFF
        BMP

    PDF:
        First attempts normal text extraction.
        If insufficient text is found, OCR is performed.
    """

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Report file not found: {file_path}"
        )

    extension = os.path.splitext(
        file_path
    )[1].lower()


    # ========================================================
    # PDF
    # ========================================================

    if extension == ".pdf":

        text = extract_text_from_pdf(
            file_path
        )

        # Use normal PDF extraction if enough text exists
        if len(text.strip()) >= 50:

            return text

        # PDF has little selectable text — needs OCR
        if not TESSERACT_INSTALLED:
            # Still return whatever text we got rather than crash
            if text.strip():
                return text
            raise RuntimeError(
                "This PDF appears to be scanned and requires "
                "Tesseract OCR to extract text. "
                "Install Tesseract: brew install tesseract"
            )

        # Otherwise perform OCR
        return extract_text_from_scanned_pdf(
            file_path
        )


    # ========================================================
    # IMAGE
    # ========================================================

    image_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp"
    }

    if extension in image_extensions:

        if not TESSERACT_INSTALLED:
            raise RuntimeError(
                "Image-based reports require Tesseract OCR "
                "to extract text. "
                "Install Tesseract: brew install tesseract"
            )

        return extract_text_from_image(
            file_path
        )


    # ========================================================
    # UNSUPPORTED FILE
    # ========================================================

    raise ValueError(
        f"Unsupported report format: {extension}"
    )


# ============================================================
# COMPLETE REPORT EXTRACTION
# ============================================================

def extract_report(file_path):
    """
    Complete medical report extraction pipeline.

    Steps:
        1. Extract report text
        2. Clean text
        3. Extract structured sections

    Returns:
        dict containing:
            raw_text
            sections
    """

    text = extract_report_text(file_path)

    sections = extract_report_sections(text)

    return {
        "raw_text": text,
        "sections": sections
    }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print("OCR service loaded successfully.")
