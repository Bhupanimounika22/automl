import streamlit as st
import pandas as pd

def upload_page():
    st.title("📂 Upload Dataset (Large Files Supported)")

    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file is None:
        st.info("👆 Upload a CSV file to continue")
        return

    try:
        # Read full file (Streamlit can handle large files in memory)
        df = pd.read_csv(file, low_memory=False)
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        return

    st.session_state.df = df

    st.success("✅ Dataset loaded successfully")

    st.subheader("📄 Dataset Preview (First 50 Rows)")
    st.dataframe(df.head(50))

    st.info(
        f"""
        **Rows:** {df.shape[0]}  
        **Columns:** {df.shape[1]}  
        **Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
        """
    )
