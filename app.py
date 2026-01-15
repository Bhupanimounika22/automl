import streamlit as st
from utils.session import init_session
from eda import upload_page
from preprocessing import preprocessing_page
from training import training_page
from prediction import prediction_page
from report import save_download_page

st.set_page_config("No-Code AutoML Platform", layout="wide")

init_session()

st.sidebar.title("🚀 No-Code AutoML")
page = st.sidebar.radio(
    "Navigation",
    [
        "1️⃣ Upload Data",
        "2️⃣ Preprocessing",
        "3️⃣ Train Model",
        "4️⃣ Predict",
        "5️⃣ Save & Report"
    ]
)

if page == "1️⃣ Upload Data":
    upload_page()
elif page == "2️⃣ Preprocessing":
    preprocessing_page()
elif page == "3️⃣ Train Model":
    training_page()
elif page == "4️⃣ Predict":
    prediction_page()
elif page == "5️⃣ Save & Report":
    save_download_page()
