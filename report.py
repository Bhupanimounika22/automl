import streamlit as st
import os, joblib
import pandas as pd

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

DOWNLOAD_DIR = "downloads"


def save_download_page():
    st.title("📄 Save Model & Generate Report")

    if "model" not in st.session_state:
        st.error("❌ Train a model first.")
        return

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    model_path = os.path.join(DOWNLOAD_DIR, "model.pkl")
    prep_path = os.path.join(DOWNLOAD_DIR, "preprocessor.pkl")
    pdf_path = os.path.join(DOWNLOAD_DIR, "report.pdf")
    cm_path = os.path.join(DOWNLOAD_DIR, "confusion_matrix.png")

    # ================= SAVE ARTIFACTS =================
    joblib.dump(st.session_state.model, model_path)
    joblib.dump(st.session_state.preprocessor, prep_path)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    elements = []

    # ================= TITLE =================
    elements.append(Paragraph("No-Code AutoML – Final Report", styles["Title"]))
    elements.append(Spacer(1, 20))

    # ================= BASIC INFO =================
    task = st.session_state.task
    score_label = "Accuracy" if task == "classification" else "R² Score"
    score_value = (
        f"{st.session_state.final_accuracy * 100:.2f}%"
        if task == "classification"
        else f"{st.session_state.final_accuracy:.3f}"
    )

    elements.append(Paragraph(
        f"""
        <b>Target Column:</b> {st.session_state.target}<br/>
        <b>Task Type:</b> {task.upper()}<br/>
        <b>Best Model:</b> {st.session_state.best_model_name}<br/>
        <b>{score_label}:</b> {score_value}
        """,
        styles["Normal"]
    ))
    elements.append(Spacer(1, 20))

    # ================= DATASET OVERVIEW =================
    df = st.session_state.df

    elements.append(Paragraph("Dataset Overview", styles["Heading2"]))
    elements.append(Paragraph(
        f"""
        <b>Total Rows:</b> {df.shape[0]}<br/>
        <b>Total Columns:</b> {df.shape[1]}<br/>
        <b>Numerical Features:</b> {len(df.select_dtypes(include="number").columns)}<br/>
        <b>Categorical Features:</b> {len(df.select_dtypes(include="object").columns)}
        """,
        styles["Normal"]
    ))
    elements.append(Spacer(1, 20))

    # ================= PREPROCESSED DATA PREVIEW =================
    elements.append(Paragraph("Preprocessed Dataset (Preview)", styles["Heading2"]))

    preview_df = df.head(20).copy()
    preview_df = preview_df.astype(str)  # PDF-safe

    elements.append(
        Table(
            [preview_df.columns.tolist()] + preview_df.values.tolist(),
            repeatRows=1
        )
    )
    elements.append(Spacer(1, 20))

    # ================= PREPROCESSING SUMMARY =================
    if "preprocessing_summary" in st.session_state:
        elements.append(Paragraph("Preprocessing Summary", styles["Heading2"]))
        prep = st.session_state.preprocessing_summary.reset_index()
        prep = prep.astype(str)
        elements.append(Table([prep.columns.tolist()] + prep.values.tolist()))
        elements.append(Spacer(1, 20))

    # ================= MODEL COMPARISON =================
    if "model_comparison" in st.session_state:
        elements.append(Paragraph("Model Comparison", styles["Heading2"]))
        comp = st.session_state.model_comparison.round(4).astype(str)
        elements.append(Table([comp.columns.tolist()] + comp.values.tolist()))
        elements.append(Spacer(1, 20))

    # ================= CONFUSION MATRIX =================
    if task == "classification" and os.path.exists(cm_path):
        elements.append(Paragraph("Confusion Matrix", styles["Heading2"]))
        elements.append(Image(cm_path, width=350, height=300))
        elements.append(Spacer(1, 20))

    # ================= BUILD PDF =================
    doc.build(elements)

    st.success("✅ PDF report generated successfully")

    # ================= DOWNLOAD BUTTONS =================
    with open(pdf_path, "rb") as f:
        st.download_button("⬇️ Download Report (PDF)", f, file_name="AutoML_Report.pdf")

    with open(model_path, "rb") as f:
        st.download_button("⬇️ Download Model (.pkl)", f, file_name="model.pkl")

    with open(prep_path, "rb") as f:
        st.download_button("⬇️ Download Preprocessor (.pkl)", f, file_name="preprocessor.pkl")
