from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image
)

from reportlab.lib.styles import getSampleStyleSheet

from datetime import datetime


def create_pdf_report(
    filename,
    disease,
    confidence,
    xray_path=None,
    heatmap_path=None
):

    pdf = SimpleDocTemplate(filename)

    styles = getSampleStyleSheet()

    content = []

    title = Paragraph(
        "AI Lung Disease Detection Report",
        styles["Title"]
    )

    content.append(title)

    content.append(Spacer(1, 20))

    current_time = datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )

    content.append(
        Paragraph(
            f"<b>Date:</b> {current_time}",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"<b>Prediction:</b> {disease}",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"<b>Confidence:</b> {confidence:.2f}%",
            styles["Normal"]
        )
    )

    content.append(Spacer(1, 20))

    if xray_path:
        content.append(
            Paragraph(
                "<b>Uploaded X-ray</b>",
                styles["Heading2"]
            )
        )

        content.append(
            Image(
                xray_path,
                width=250,
                height=250
            )
        )

        content.append(
            Spacer(1, 20)
        )

    if heatmap_path:
        content.append(
            Paragraph(
                "<b>Grad-CAM Heatmap</b>",
                styles["Heading2"]
            )
        )

        content.append(
            Image(
                heatmap_path,
                width=250,
                height=250
            )
        )

    content.append(
        Spacer(1, 20)
    )

    content.append(
        Paragraph(
            "Generated using CNN + Attention U-Net + Grad-CAM",
            styles["Italic"]
        )
    )

    pdf.build(content)