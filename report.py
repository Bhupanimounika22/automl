import streamlit as st
import os, joblib
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

DOWNLOAD_DIR = "downloads"


def save_download_page():
    st.title("📄 Save Model & Generate Report")

    if st.session_state.model is None:
        st.error("❌ Train a model first.")
        return

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    model_path = os.path.join(DOWNLOAD_DIR, "model.pkl")
    prep_path = os.path.join(DOWNLOAD_DIR, "preprocessor.pkl")
    pdf_path = os.path.join(DOWNLOAD_DIR, "report.pdf")
    cm_path = os.path.join(DOWNLOAD_DIR, "confusion_matrix.png")

    # ---------- SAVE ARTIFACTS ----------
    joblib.dump(st.session_state.model, model_path)
    joblib.dump(st.session_state.preprocessor, prep_path)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    elements = []

    # ================= TITLE =================
    elements.append(Paragraph("No-Code AutoML Final Report", styles["Title"]))
    elements.append(Spacer(1, 20))

    # ================= TRAINING INFO =================
    task = st.session_state.task
    score_label = "Accuracy" if task == "classification" else "R² Score"
    score_value = (
        f"{st.session_state.final_accuracy*100:.2f}%"
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

    # ================= PREPROCESSING SUMMARY =================
    if st.session_state.preprocessing_summary is not None:
        elements.append(Paragraph("Preprocessing Summary", styles["Heading2"]))
        prep = st.session_state.preprocessing_summary.reset_index()
        elements.append(Table([prep.columns.tolist()] + prep.values.tolist()))
        elements.append(Spacer(1, 20))

    # ================= MODEL COMPARISON =================
    if st.session_state.model_comparison is not None:
        elements.append(Paragraph("Model Comparison", styles["Heading2"]))
        comp = st.session_state.model_comparison
        elements.append(Table([comp.columns.tolist()] + comp.values.tolist()))
        elements.append(Spacer(1, 20))

    # ================= REPORT TABLE =================
    if st.session_state.report_table is not None:
        title = "Classification Report" if task == "classification" else "Regression Metrics"
        elements.append(Paragraph(title, styles["Heading2"]))
        rpt = st.session_state.report_table.reset_index()
        elements.append(Table([rpt.columns.tolist()] + rpt.values.tolist()))
        elements.append(Spacer(1, 20))

    # ================= CONFUSION MATRIX (CLASSIFICATION ONLY) =================
    if task == "classification" and os.path.exists(cm_path):
        elements.append(Paragraph("Confusion Matrix", styles["Heading2"]))
        elements.append(Image(cm_path, width=350, height=300))
        elements.append(Spacer(1, 20))

    # ================= BUILD PDF =================
    doc.build(elements)

    st.success("✅ PDF report generated successfully")

    # ================= DOWNLOADS =================
    with open(pdf_path, "rb") as f:
        st.download_button("⬇️ Download Report (PDF)", f, file_name="report.pdf")

    with open(model_path, "rb") as f:
        st.download_button("⬇️ Download Model (.pkl)", f, file_name="model.pkl")
