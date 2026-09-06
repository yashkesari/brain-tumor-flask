import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)


def generate_pdf_report(
    output_path,
    filename,
    prediction,
    classification_confidence,
    probabilities,
    segmentation_detected=False,
    yolo_confidence=0,
    tumor_area=0,
    number_of_masks=0,
    tumor_pixels=0,
    result_image=None,
):
    """
    Generate a PDF report containing the AI-assisted MRI analysis results.
    """

    # --------------------------------------------------
    # Ensure output directory exists
    # --------------------------------------------------

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    # --------------------------------------------------
    # PDF document
    # --------------------------------------------------

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=22,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        textColor=colors.grey,
        spaceAfter=18,
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=12,
        spaceAfter=8,
    )

    normal_style = ParagraphStyle(
        "NormalCustom",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
    )

    # --------------------------------------------------
    # Story
    # --------------------------------------------------

    story = []

    # Title
    story.append(
        Paragraph(
            "NeuroScan AI",
            title_style
        )
    )

    story.append(
        Paragraph(
            "AI-Assisted Brain MRI Analysis Report",
            subtitle_style
        )
    )

    # --------------------------------------------------
    # Patient / scan information
    # --------------------------------------------------

    story.append(
        Paragraph(
            "Scan Information",
            heading_style
        )
    )

    scan_data = [
        ["MRI File", filename],
        ["Analysis Type", "AI-assisted MRI classification and segmentation"],
    ]

    scan_table = Table(
        scan_data,
        colWidths=[45 * mm, 125 * mm]
    )

    scan_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(scan_table)
    story.append(Spacer(1, 10))

    # --------------------------------------------------
    # Classification result
    # --------------------------------------------------

    story.append(
        Paragraph(
            "Classification Result",
            heading_style
        )
    )

    classification_data = [
        ["Predicted Class", str(prediction)],
        [
            "Classification Confidence",
            f"{classification_confidence:.2f}%"
        ],
    ]

    classification_table = Table(
        classification_data,
        colWidths=[65 * mm, 105 * mm]
    )

    classification_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ])
    )

    story.append(classification_table)

    # --------------------------------------------------
    # Class probabilities
    # --------------------------------------------------

    story.append(
        Paragraph(
            "Class Probabilities",
            heading_style
        )
    )

    probability_data = [
        ["Class", "Probability"],
        [
            "Glioma",
            f"{float(probabilities[0]):.2f}%"
        ],
        [
            "Meningioma",
            f"{float(probabilities[1]):.2f}%"
        ],
        [
            "No Tumor",
            f"{float(probabilities[2]):.2f}%"
        ],
        [
            "Pituitary",
            f"{float(probabilities[3]):.2f}%"
        ],
    ]

    probability_table = Table(
        probability_data,
        colWidths=[100 * mm, 70 * mm]
    )

    probability_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (1, 1), (1, -1), "RIGHT"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(probability_table)

    # --------------------------------------------------
    # Segmentation results
    # --------------------------------------------------

    story.append(
        Paragraph(
            "Tumor Segmentation",
            heading_style
        )
    )

    if segmentation_detected:

        segmentation_data = [
            ["Segmentation Status", "Tumor region detected"],
            [
                "YOLO Confidence",
                f"{float(yolo_confidence):.2f}%"
            ],
            [
                "Number of Masks",
                str(number_of_masks)
            ],
            [
                "Tumor Pixels",
                str(tumor_pixels)
            ],
            [
                "Tumor Area",
                f"{float(tumor_area):.2f}%"
            ],
        ]

    else:

        segmentation_data = [
            [
                "Segmentation Status",
                "No tumor region detected"
            ]
        ]

    segmentation_table = Table(
        segmentation_data,
        colWidths=[65 * mm, 105 * mm]
    )

    segmentation_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(segmentation_table)

    # --------------------------------------------------
    # Result image
    # --------------------------------------------------

    if result_image:

        result_path = os.path.join(
            "results",
            result_image
        )

        if os.path.exists(result_path):

            story.append(
                Paragraph(
                    "Segmentation Visualization",
                    heading_style
                )
            )

            image = Image(
                result_path,
                width=120 * mm,
                height=90 * mm,
                kind="proportional"
            )

            story.append(image)

    # --------------------------------------------------
    # Disclaimer
    # --------------------------------------------------

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            "<b>Important Disclaimer</b>",
            heading_style
        )
    )

    story.append(
        Paragraph(
            "This report is generated by a student/research prototype "
            "for AI-assisted analysis of brain MRI images. The results "
            "are not a medical diagnosis and should not be used as a "
            "substitute for evaluation by a qualified healthcare "
            "professional. MRI findings should be interpreted by "
            "appropriate medical specialists.",
            normal_style
        )
    )

    # --------------------------------------------------
    # Build PDF
    # --------------------------------------------------

    doc.build(story)

    return output_path