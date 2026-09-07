import re


# ============================================================
# BASIC HELPERS
# ============================================================

def _clean(value):
    if not value:
        return ""

    value = str(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _unique(items):
    cleaned = []

    for item in items:
        item = _clean(item)

        if not item:
            continue

        if any(item.lower() == existing.lower() for existing in cleaned):
            continue

        cleaned.append(item)

    # Remove generic terms when a more specific term exists
    final = []

    for item in cleaned:
        item_lower = item.lower()
        generic = False

        for other in cleaned:
            other_lower = other.lower()

            if item_lower == other_lower:
                continue

            if item_lower in other_lower and len(other_lower) > len(item_lower):
                generic = True
                break

        if not generic:
            final.append(item)

    return final


# ============================================================
# DOCUMENT TYPE DETECTION
# ============================================================

def detect_document_type(text):

    if not text:
        return "unknown"

    text = _clean(text).lower()

    # --------------------------------------------------------
    # STRONG REFERENCE / GUIDELINE INDICATORS
    # --------------------------------------------------------

    guideline_terms = [
        "guideline",
        "guidelines",
        "reporting checklist",
        "reporting checklists",
        "synoptic reporting",
        "synoptic report",
        "recommendations",
        "consensus",
        "imaging protocols",
        "clinical practice guideline",
        "literature review",
        "reference document",
        "technical recommendations",
        "oncologic imaging",
        "reporting protocol",
        "protocols and reporting",
        "reporting in brain tumor",
        "reporting in brain tumour",
    ]

    # --------------------------------------------------------
    # STRONG PATIENT REPORT INDICATORS
    # --------------------------------------------------------

    patient_terms = [
        "patient name",
        "patient id",
        "patient id:",
        "date of birth",
        "dob",
        "clinical history",
        "history:",
        "findings:",
        "impression:",
        "final diagnosis",
        "radiologist",
        "exam date",
        "study date",
        "report date",
        "mri of the brain",
        "mri brain",
        "sequences:",
        "report details:",
        "operation",
        "clinical indication:",
    ]

    guideline_score = sum(
        1
        for term in guideline_terms
        if term in text
    )

    patient_score = sum(
        1
        for term in patient_terms
        if term in text
    )

    if guideline_score >= 2:
        return "guideline_reference"

    if patient_score >= 3:
        return "patient_report"

    return "unknown"


# ============================================================
# MEASUREMENTS
# ============================================================

def extract_measurements(text):

    if not text:
        return []

    pattern = (
        r"\b\d+(?:\.\d+)?"
        r"(?:\s*[x×]\s*\d+(?:\.\d+)?){0,2}"
        r"\s*cm(?:s)?\b"
    )

    matches = re.findall(
        pattern,
        text,
        re.IGNORECASE
    )

    measurements = []

    for measurement in matches:

        measurement = _clean(measurement)

        measurement = re.sub(
            r"\s*[x×]\s*",
            " x ",
            measurement,
            flags=re.IGNORECASE
        )

        measurement = re.sub(
            r"\s*cm(?:s)?\b",
            " cm",
            measurement,
            flags=re.IGNORECASE
        )

        measurements.append(measurement)

    return _unique(measurements)


# ============================================================
# CONTEXTUAL LOCATION EXTRACTION
# ============================================================

# Negation indicators — if any of these appear in the
# same sentence as an anatomical term, the location is
# NOT extracted as a tumor location.

_NEGATION_INDICATORS = [
    "unremarkable",
    "normal",
    "no abnormality",
    "no significant",
    "within normal limits",
    "grossly normal",
    "no evidence of",
    "appears normal",
    "no lesion",
    "no mass",
    "no change",
    "intact",
]


def _get_sentence_context(text, start, end):
    """
    Extract the sentence containing the match
    at positions [start, end].

    A sentence boundary is a period or newline.
    """

    # Find previous sentence boundary
    prev_boundary = max(
        text.rfind(".", 0, start),
        text.rfind("\n", 0, start)
    )
    if prev_boundary == -1:
        prev_boundary = 0
    else:
        prev_boundary += 1

    # Find next sentence boundary
    next_dot = text.find(".", end)
    next_nl = text.find("\n", end)

    if next_dot == -1 and next_nl == -1:
        next_boundary = len(text)
    elif next_dot == -1:
        next_boundary = next_nl
    elif next_nl == -1:
        next_boundary = next_dot
    else:
        next_boundary = min(next_dot, next_nl)

    return text[prev_boundary:next_boundary].lower()


def _is_negated_context(sentence):
    """
    Check if a sentence context contains negation
    indicators that would disqualify a location
    from being a tumor location.
    """
    for indicator in _NEGATION_INDICATORS:
        if indicator in sentence:
            return True
    return False


def extract_location(text):

    if not text:
        return []

    patterns = [

        # Specific tumor/anatomical locations
        r"\bright\s+temporal\s*/\s*occipital\s+region\b",
        r"\bleft\s+temporal\s*/\s*occipital\s+region\b",

        r"\bright\s+temporal\s+region\b",
        r"\bleft\s+temporal\s+region\b",

        r"\bright\s+occipital\s+region\b",
        r"\bleft\s+occipital\s+region\b",

        r"\bright\s+frontal\s+region\b",
        r"\bleft\s+frontal\s+region\b",

        r"\bright\s+frontal\s+lobe\b",
        r"\bleft\s+frontal\s+lobe\b",

        r"\bright\s+parietal\s+region\b",
        r"\bleft\s+parietal\s+region\b",

        r"\bright\s+parietal\s+lobe\b",
        r"\bleft\s+parietal\s+lobe\b",

        r"\bright\s+frontoparietal\s+region\b",
        r"\bleft\s+frontoparietal\s+region\b",

        r"\bright\s+cerebral\s+hemisphere\b",
        r"\bleft\s+cerebral\s+hemisphere\b",

        r"\bright\s+lateral\s+ventricle\b",
        r"\bleft\s+lateral\s+ventricle\b",

        r"\blateral\s+ventricle\b",

        r"\bperiventricular\s+white\s+matter\b",

        r"\bcerebral\s+peduncle\b",

        r"\bprecentral\s+gyrus\b",

        r"\bcentrum\s+semiovale\b",

        r"\bpituitary\s+gland\b",
        r"\bpituitary\s+region\b",
        r"\bsellar\s+region\b",
        r"\bsella\s+turcica\b",

        r"\bcerebellum\b",
        r"\bcerebellar\s+hemisphere\b",

        r"\bcorpus\s+callosum\b",

        r"\bbrainstem\b",
    ]

    locations = []

    text_lower = text.lower()

    for pattern in patterns:

        for match in re.finditer(pattern, text_lower):

            # Get the sentence containing this match
            sentence = _get_sentence_context(
                text_lower,
                match.start(),
                match.end()
            )

            # Skip if the sentence context is negated
            if _is_negated_context(sentence):
                continue

            locations.append(match.group())

    return _unique(locations)


# ============================================================
# SYMPTOM EXTRACTION
# ============================================================

def extract_symptoms(history):

    if not history:
        return []

    history = _clean(history)

    symptoms = []

    symptom_patterns = {

        "headache":
            r"\bheadaches?\b",

        "dizziness":
            r"\bdizziness\b|\bdizzy\b",

        "seizures":
            r"\bseizures?\b",

        "weakness":
            r"\bweakness\b|\bweak\b",

        "numbness":
            r"\bnumbness\b|\bnumb\b",

        "vision problems":
            r"\bvision\s+(?:problem|problems|loss|change|changes)\b",

        "speech problems":
            r"\bspeech\s+(?:problem|problems|difficulty|difficulties|change|changes)\b",

        "memory problems":
            r"\bmemory\s+(?:problem|problems|loss|change|changes)\b",

        "confusion":
            r"\bconfusion\b|\bconfused\b",

        "balance problems":
            r"\bbalance\s+(?:problem|problems|difficulty|difficulties)\b",

        "nausea":
            r"\bnausea\b",

        "vomiting":
            r"\bvomiting\b|\bvomit\b",

        "fatigue":
            r"\bfatigue\b|\bfatigued\b",
    }

    for symptom, pattern in symptom_patterns.items():

        if re.search(
            pattern,
            history,
            re.IGNORECASE
        ):
            symptoms.append(symptom)

    # More specific weakness patterns
    if re.search(
        r"weakness\s+of\s+(?:the\s+)?right\s+upper\s+limb",
        history,
        re.IGNORECASE
    ):
        symptoms.append(
            "right upper limb weakness"
        )

    if re.search(
        r"weakness\s+of\s+(?:the\s+)?left\s+upper\s+limb",
        history,
        re.IGNORECASE
    ):
        symptoms.append(
            "left upper limb weakness"
        )

    return _unique(symptoms)


# ============================================================
# TUMOR TERM EXTRACTION
# ============================================================

def extract_tumor_terms(text):

    if not text:
        return []

    terms = [

        "glioblastoma",

        "high-grade glioma",

        "low-grade glioma",

        "glioma",

        "meningioma",

        "pituitary adenoma",

        "pituitary tumor",

        "brain tumor",

        "brain tumour",

        "neoplasm",

        "mass lesion",

        "space occupying lesion",

        "space-occupying lesion",

        "solid cystic lesion",

        "residual lesion",

        "recurrent lesion",

        "edema",

        "oedema",

        "enhancement",

        "post-treatment changes",
    ]

    found_terms = []

    for term in terms:

        if re.search(
            re.escape(term),
            text,
            re.IGNORECASE
        ):
            found_terms.append(term)

    return _unique(found_terms)


# ============================================================
# TREATMENT EXTRACTION
# ============================================================

def extract_treatment_history(text):

    empty = {
        "surgery": False,
        "chemotherapy": False,
        "radiotherapy": False,
        "temozolomide": False,
    }

    if not text:
        return empty

    return {

        "surgery": bool(
            re.search(
                r"\b(?:"
                r"surgery|"
                r"surgical|"
                r"craniotomy|"
                r"operation|"
                r"decompression|"
                r"resection|"
                r"operated|"
                r"post[- ]operative"
                r")\b",
                text,
                re.IGNORECASE
            )
        ),

        "chemotherapy": bool(
            re.search(
                r"\bchemotherapy\b|\bchemo\b",
                text,
                re.IGNORECASE
            )
        ),

        "radiotherapy": bool(
            re.search(
                r"\bradiotherapy\b|"
                r"\bradiation\s+therapy\b|"
                r"\bradiation\b|"
                r"\bRT\b",
                text,
                re.IGNORECASE
            )
        ),

        "temozolomide": bool(
            re.search(
                r"\btemozolomide\b",
                text,
                re.IGNORECASE
            )
        ),
    }


# ============================================================
# PROGRESSION / CHANGE EXTRACTION
# ============================================================

def extract_progression(impression):

    if not impression:
        return []

    progression_terms = []

    patterns = {

        "increase in size":
            r"increase\s+in\s+(?:size|extent)",

        "decrease in size":
            r"decrease\s+in\s+(?:size|extent)",

        "more prominent":
            r"more\s+prominent",

        "less prominent":
            r"less\s+prominent",

        "increase in edema":
            r"increase\s+in\s+.*(?:edema|oedema)",

        "recurrent/residual lesion":
            r"recurrent[,\s]+residual\s+lesion",

        "post-treatment changes":
            r"post[,\s-]*treatment\s+changes",

        "stable":
            r"\bstable\b",

        "unchanged":
            r"\bunchanged\b",

        "no significant change":
            r"no\s+significant\s+change",

        "follow-up recommended":
            r"follow[-\s]?up\s+recommended|"
            r"further\s+follow[-\s]?up",
    }

    for label, pattern in patterns.items():

        if re.search(
            pattern,
            impression,
            re.IGNORECASE
        ):
            progression_terms.append(label)

    return _unique(progression_terms)


# ============================================================
# DIAGNOSIS EXTRACTION
# ============================================================

def extract_diagnosis(diagnosis):

    if not diagnosis:
        return ""

    diagnosis = _clean(diagnosis)

    # Remove everything after obvious operation/history sections
    diagnosis = re.split(
        r"\b(?:"
        r"OPERATION|"
        r"OPERATION\s+DATE|"
        r"OPERATION\s+TITLE|"
        r"SURGERY|"
        r"PROCEDURE|"
        r"OPERATED|"
        r"CRANIOTOMY"
        r")\b",
        diagnosis,
        maxsplit=1,
        flags=re.IGNORECASE
    )[0]

    # Remove trailing OCR artifacts
    diagnosis = re.split(
        r"\b(?:"
        r"HISTORY|"
        r"FINDINGS|"
        r"REPORT\s+DETAILS|"
        r"SEQUENCES|"
        r"IMPRESSION"
        r")\s*:?",
        diagnosis,
        maxsplit=1,
        flags=re.IGNORECASE
    )[0]

    return _clean(diagnosis)


# ============================================================
# SUMMARY GENERATION
# ============================================================

def generate_report_summary(result):

    if not result:
        return {
            "title": "Report Summary",
            "items": []
        }

    summary_items = []

    diagnosis = result.get(
        "diagnosis",
        ""
    )

    locations = result.get(
        "locations",
        []
    )

    measurements = result.get(
        "measurements",
        []
    )

    symptoms = result.get(
        "symptoms",
        []
    )

    treatment = result.get(
        "treatment",
        {}
    )

    progression = result.get(
        "progression",
        []
    )

    tumor_terms = result.get(
        "tumor_terms",
        []
    )

    # Diagnosis
    if diagnosis:

        summary_items.append({
            "section": "Reported Diagnosis",
            "text": diagnosis,
        })

    # Symptoms
    if symptoms:

        summary_items.append({
            "section": "Reported Symptoms",
            "text": ", ".join(symptoms),
        })

    # Locations
    if locations:

        summary_items.append({
            "section": "Reported Location",
            "text": ", ".join(locations),
        })

    # Measurements
    if measurements:

        summary_items.append({
            "section": "Reported Measurements",
            "text": ", ".join(measurements),
        })

    # Treatment
    treatment_items = []

    if treatment.get(
        "surgery",
        False
    ):
        treatment_items.append(
            "surgery"
        )

    if treatment.get(
        "chemotherapy",
        False
    ):
        treatment_items.append(
            "chemotherapy"
        )

    if treatment.get(
        "radiotherapy",
        False
    ):
        treatment_items.append(
            "radiotherapy"
        )

    if treatment.get(
        "temozolomide",
        False
    ):
        treatment_items.append(
            "temozolomide"
        )

    if treatment_items:

        summary_items.append({
            "section": "Treatment Mentioned",
            "text": ", ".join(
                treatment_items
            ),
        })

    # Progression
    if progression:

        summary_items.append({
            "section": "Reported Changes",
            "text": ", ".join(
                progression
            ),
        })

    # Tumor terms
    if tumor_terms:

        summary_items.append({
            "section": "Important Report Terms",
            "text": ", ".join(
                tumor_terms
            ),
        })

    # Nothing extracted
    if not summary_items:

        summary_items.append({
            "section": "Information",
            "text": (
                "No structured findings were "
                "identified by the current "
                "extraction rules."
            ),
        })

    return {
        "title": "Report Summary",
        "items": summary_items,
    }


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_report(report):

    empty_result = {

        "document_type": "unknown",

        "diagnosis": "",

        "tumor_terms": [],

        "locations": [],

        "measurements": [],

        "symptoms": [],

        "treatment": {
            "surgery": False,
            "chemotherapy": False,
            "radiotherapy": False,
            "temozolomide": False,
        },

        "progression": [],
    }

    if not report:

        empty_result["summary"] = (
            generate_report_summary(
                empty_result
            )
        )

        return empty_result

    raw_text = _clean(
        report.get(
            "raw_text",
            ""
        )
    )

    # --------------------------------------------------------
    # DOCUMENT TYPE
    # --------------------------------------------------------

    document_type = detect_document_type(
        raw_text
    )

    empty_result[
        "document_type"
    ] = document_type

    # --------------------------------------------------------
    # GUIDELINE / REFERENCE DOCUMENT
    # --------------------------------------------------------

    if document_type == "guideline_reference":

        empty_result["summary"] = {

            "title":
                "Document Not Analyzed",

            "items": [

                {
                    "section":
                        "Document Type",

                    "text":
                        (
                            "This document appears "
                            "to be a guideline or "
                            "reference document, "
                            "not an individual "
                            "patient medical report."
                        ),
                }
            ],
        }

        return empty_result

    # --------------------------------------------------------
    # SECTIONS
    # --------------------------------------------------------

    sections = report.get(
        "sections",
        {}
    )

    diagnosis = _clean(
        sections.get(
            "diagnosis",
            ""
        )
    )

    history = _clean(
        sections.get(
            "history",
            ""
        )
    )

    findings = _clean(
        sections.get(
            "findings",
            ""
        )
    )

    impression = _clean(
        sections.get(
            "impression",
            ""
        )
    )

    # --------------------------------------------------------
    # UNKNOWN DOCUMENT
    # --------------------------------------------------------

    if document_type == "unknown":

        section_count = sum(
            bool(section)
            for section in [
                diagnosis,
                history,
                findings,
                impression,
            ]
        )

        if section_count < 2:

            empty_result["summary"] = {

                "title":
                    "Document Not Analyzed",

                "items": [

                    {
                        "section":
                            "Information",

                        "text":
                            (
                                "The document could "
                                "not be confidently "
                                "identified as an "
                                "individual patient "
                                "medical report."
                            ),
                    }
                ],
            }

            return empty_result

    # --------------------------------------------------------
    # PATIENT REPORT ANALYSIS
    # --------------------------------------------------------

    patient_content = " ".join([
        diagnosis,
        history,
        findings,
        impression,
    ])

    result = {

        "document_type":
            document_type,

        "diagnosis":
            extract_diagnosis(
                diagnosis
            ),

        "tumor_terms":
            extract_tumor_terms(
                patient_content
            ),

        "locations":
            extract_location(
                findings
            ),

        "measurements":
            extract_measurements(
                findings
            ),

        "symptoms":
            extract_symptoms(
                history
            ),

        "treatment":
            extract_treatment_history(
                " ".join([
                    diagnosis,
                    history,
                ])
            ),

        "progression":
            extract_progression(
                impression
            ),
    }

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    result["summary"] = (
        generate_report_summary(
            result
        )
    )

    return result


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print(
        "AI/NLP service loaded successfully."
    )
