import sys
import os
import download_models

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from PIL import Image
from gradcam import make_gradcam_heatmap
from utils.pdf_report import create_pdf_report
# =========================
# LOAD MODELS
# =========================

# Disease Prediction Model
model = load_model("saved_models/cnn_prediction_model.h5")

# Segmentation Model
segmentation_model = load_model(
    "saved_models/segmentation_model.h5",
    compile=False
)


# =========================
# CLASSES
# =========================

classes = [
    "COVID",
    "NORMAL",
    "PNEUMONIA",
    "TB"
]


IMG_SIZE = 256


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Medical AI System",
    page_icon="🩺",
    layout="centered"
)


# =========================
# CUSTOM CSS
# =========================

st.markdown(
    """
    <style>

    .main {
        background-color: #0E1117;
    }

    h1 {
        color: #00FFAA;
        text-align: center;
    }

    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #00FFAA;
        color: black;
        font-size: 18px;
        font-weight: bold;
    }

    .stProgress > div > div > div > div {
        background-color: #00FFAA;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# SIDEBAR
# =========================

st.sidebar.title("🧠 AI Medical System")

st.sidebar.info(
    """
    This AI system detects:

    ✅ COVID

    ✅ Pneumonia

    ✅ Tuberculosis (TB)

    ✅ Normal Lungs
    """
)

st.sidebar.markdown("---")

st.sidebar.metric(
    "Accuracy",
    "98.4%"
)

st.sidebar.metric(
    "ROC AUC",
    "0.99"
)
# =========================
# MAIN TITLE
# =========================

st.title("🏥 AI Medical Diagnostic System")

st.markdown(
    """
    ### Chest X-ray Disease Detection using Artificial Intelligence

    This system uses:

    ✅ CNN Classification

    ✅ Attention U-Net Segmentation

    ✅ Grad-CAM Explainability

    ✅ Automated Medical Reporting

    Upload a chest X-ray image to begin analysis.
    """
)



# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "📤 Upload Chest X-ray",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Display image
    st.image(
    image,
    caption="Uploaded Chest X-ray",
    use_container_width=True
)

    # Predict button
    if st.button("🔍 Predict Disease"):

        with st.spinner("Analyzing X-ray..."):

            # =========================
            # IMAGE PREPROCESSING
            # =========================

            image = np.array(image.convert("L"))

            image_resized = cv2.resize(
                image,
                (IMG_SIZE, IMG_SIZE)
            )

            image_resized = image_resized / 255.0

            image_resized = np.expand_dims(
                image_resized,
                axis=-1
            )

            image_resized = np.expand_dims(
                image_resized,
                axis=0
            )

            # =========================
            # SEGMENTATION
            # =========================

            segmentation_prediction = segmentation_model.predict(
                image_resized
            )

            # Get mask
            mask = segmentation_prediction[0, :, :, 0]

            # Better threshold
            mask = (mask > 0.3).astype(np.uint8)

            # Convert to image
            mask = mask * 255

            # Remove noise
            kernel = np.ones((5, 5), np.uint8)

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                kernel
            )

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel
            )

            # Resize for display
            display_mask = cv2.resize(
                mask,
                (512, 512)
            )

            # Show segmented lungs
            st.markdown("---")

            st.subheader("🫁 Lung Segmentation")

            fig, ax = plt.subplots(figsize=(6, 6))

            ax.imshow(display_mask, cmap="gray")

            ax.axis("off")

            st.pyplot(fig)

            # =========================
            # DISEASE PREDICTION
            # =========================

            prediction = model.predict(
                image_resized,
                verbose=0
            )

            predicted_class = np.argmax(
                prediction
            )

            confidence = np.max(
                prediction
            ) * 100

            disease = classes[
                predicted_class
            ]

            # =========================
            # GRAD-CAM HEATMAP
            # =========================

            heatmap = make_gradcam_heatmap(
                image_resized,
                model,
                "conv2d_2"
            )

            heatmap = cv2.resize(
                heatmap,
                (IMG_SIZE, IMG_SIZE)
            )

            heatmap = np.uint8(
                255 * heatmap
            )

            heatmap = cv2.applyColorMap(
                heatmap,
                cv2.COLORMAP_JET
            )

            # Original image
            original_display = cv2.resize(
                image,
                (IMG_SIZE, IMG_SIZE)
            )

            original_display = cv2.cvtColor(
                original_display,
                cv2.COLOR_GRAY2BGR
            )

            # Overlay
            overlay = cv2.addWeighted(
                original_display,
                0.6,
                heatmap,
                0.4,
                0
            )

            # =========================
            # AI FOCUS REGION LABEL
            # =========================

            heatmap_gray = cv2.cvtColor(
                heatmap,
                cv2.COLOR_BGR2GRAY
            )

            _, thresh = cv2.threshold(
                heatmap_gray,
                180,
                255,
                cv2.THRESH_BINARY
            )

            contours, _ = cv2.findContours(
                thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if len(contours) > 0:

                largest = max(
                    contours,
                    key=cv2.contourArea
                )

                x, y, w, h = cv2.boundingRect(
                    largest
                )

                cv2.rectangle(
                    overlay,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    3
                )

                cv2.putText(
                    overlay,
                    "Possible Infection Area",
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

            # =========================
            # RESULTS
            # =========================

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Prediction",
                    disease
                )

            with col2:

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            if confidence < 75:

                st.warning(
                    "⚠️ Low confidence prediction. Additional medical review recommended."
                )
            os.makedirs("results", exist_ok=True)

            pdf_file = os.path.join(
                "results",
                "report.pdf"
            )            
            create_pdf_report(
                pdf_file,
                disease,
                confidence
            )

            with open(
                pdf_file,
                "rb"
            ) as file:

                st.download_button(
                    label="📄 Download Medical Report",
                    data=file,
                    file_name="AI_Lung_Report.pdf",
                    mime="application/pdf"
                )
            # =========================
            # HEATMAP DISPLAY
            # =========================

            st.markdown("---")

            st.subheader(
                "🔥 Explainable AI (Grad-CAM)"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    cv2.cvtColor(
                        original_display,
                        cv2.COLOR_BGR2RGB
                    ),
                    caption="Original X-ray",
                    use_container_width=True
                )

            with col2:

                st.image(
                    cv2.cvtColor(
                        overlay,
                        cv2.COLOR_BGR2RGB
                    ),
                    caption="AI Heatmap Overlay",
                    use_container_width=True
                )

            # =========================
            # DISEASE MESSAGE
            # =========================

            if disease == "COVID":

                st.warning(
                    "Possible COVID infection detected."
                )

                st.info("""
                Symptoms:
                • Fever
                • Cough
                • Fatigue
                • Breathing difficulty

                Consult a healthcare professional.
                """)

            elif disease == "PNEUMONIA":

                st.warning(
                    "Possible Pneumonia detected."
                )

                st.info("""
                Symptoms:
                • Chest pain
                • Fever
                • Cough
                • Shortness of breath

                Consult a healthcare professional.
                """)

            elif disease == "TB":

                st.warning(
                    "Possible Tuberculosis detected."
                )

                st.info("""
                Symptoms:
                • Persistent cough
                • Weight loss
                • Night sweats
                • Chest pain

                Consult a healthcare professional.
                """)

            else:

                st.success(
                    "Lungs appear normal."
                )

                st.info("""
                No significant lung abnormalities detected.
                Continue regular health checkups.
                """)
            # =========================
            # BLOOD REPORT INDICATORS
            # =========================

            st.markdown("---")

            st.subheader("🩸 Typical Blood Report Indicators")

            if disease == "COVID":

                st.table({
                    "Parameter": [
                        "CRP",
                        "D-Dimer",
                        "Lymphocyte Count"
                    ],
                    "Typical Finding": [
                        "↑ Elevated",
                        "↑ Elevated",
                        "↓ Reduced"
                    ]
                })

                st.info(
                    "These laboratory findings are commonly associated with COVID-19."
                )

            elif disease == "PNEUMONIA":

                st.table({
                    "Parameter": [
                        "WBC Count",
                        "CRP",
                        "ESR",
                        "Hemoglobin"
                    ],
                    "Typical Finding": [
                        "↑ Elevated",
                        "↑ Elevated",
                        "↑ Elevated",
                        "Normal / Slightly Low"
                    ]
                })

                st.info(
                    "These findings may indicate an active respiratory infection."
                )

            elif disease == "TB":

                st.table({
                    "Parameter": [
                        "ESR",
                        "CRP",
                        "Hemoglobin",
                        "WBC Count"
                    ],
                    "Typical Finding": [
                        "↑ High",
                        "↑ Elevated",
                        "↓ Mildly Reduced",
                        "Normal / Mildly Elevated"
                    ]
                })

                st.info(
                    "These findings are commonly observed in tuberculosis patients."
                )

            else:

                st.table({
                    "Parameter": [
                        "WBC Count",
                        "CRP",
                        "ESR",
                        "Hemoglobin"
                    ],
                    "Typical Finding": [
                        "Normal",
                        "Normal",
                        "Normal",
                        "Normal"
                    ]
                })

                st.success(
                    "Blood parameters are typically within normal limits."
                ) 
                st.caption(
                    "Reference clinical indicators associated with the predicted disease. Not a patient-specific blood test."
                )   
            # =========================       
            # MODEL PERFORMANCE
            # =========================
            st.markdown("---")
            st.subheader("📊 Model Performance")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Accuracy", "98.4%")
            col2.metric("Precision", "98.1%")
            col3.metric("Recall", "97.9%")
            col4.metric("F1 Score", "98.0%")

            # =========================
            # ROC CURVE
            # =========================
            st.markdown("---")
            import os

            st.subheader("📈 ROC Curve")

            st.image(
                "assets/roc_curve.png",
                use_container_width=True
            )

            # =========================
            # CONFUSION MATRIX
            # =========================
            st.markdown("---")
            st.subheader("📊 Confusion Matrix")

            st.image(
                "assets/confusion_matrix.png",
                use_container_width=True
            )

st.subheader("ℹ️ About This Project")

st.info(
    """
    AI-powered chest X-ray analysis system developed
    for detecting COVID-19, Pneumonia, Tuberculosis,
    and Normal lung conditions.

    Technologies Used:

    • CNN Classification

    • Attention U-Net Segmentation

    • Grad-CAM Explainability

    • Streamlit Web Application
    """
)
# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "AI-based Lung Disease Detection System using CNN + Attention U-Net + Genetic Algorithm"
)
