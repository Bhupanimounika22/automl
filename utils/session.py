import streamlit as st

def init_session():
    defaults = {
        "df": None,
        "target": None,
        "task": None,
        "preprocessor": None,
        "label_encoder": None,
        "model": None,
        "best_model_name": None,
        "final_accuracy": None,
        "report_table": None,
        "model_comparison": None,
        "preprocessing_summary": None,
        "processed_preview": None
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
