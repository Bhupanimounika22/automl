import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocessing_page():
    st.title("🧹 Data Preprocessing (Transparent Mode)")

    df = st.session_state.df
    if df is None:
        st.error("❌ Upload data first.")
        return

    st.subheader("📄 Original Dataset (Preview)")
    st.dataframe(df.head(30))

    # ---------- TARGET ----------
    target = st.selectbox("🎯 Select Target Column", df.columns)
    st.session_state.target = target

    # ---------- TASK ----------
    if df[target].dtype == "object" or df[target].nunique() <= 20:
        task = "classification"
    else:
        task = "regression"
    st.session_state.task = task

    st.success(f"Task detected: **{task.upper()}**")

    # ---------- MISSING VALUES ----------
    st.subheader("❗ Missing Values Analysis")
    missing_before = df.isnull().sum()
    missing_df = missing_before[missing_before > 0].to_frame("Missing Count")
    st.dataframe(missing_df)

    # ---------- USER OPTIONS ----------
    st.subheader("⚙️ Missing Value Strategy")

    missing_strategy = st.radio(
        "Choose how to handle missing values:",
        ["Fill Missing Values", "Drop Rows with Missing", "Drop Columns with Missing"]
    )

    df_processed = df.copy()

    if missing_strategy == "Drop Rows with Missing":
        df_processed = df_processed.dropna()
    elif missing_strategy == "Drop Columns with Missing":
        df_processed = df_processed.dropna(axis=1)
    else:
        num_cols = df_processed.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = df_processed.select_dtypes(include=["object", "category"]).columns

        df_processed[num_cols] = df_processed[num_cols].fillna(
            df_processed[num_cols].median()
        )
        for col in cat_cols:
            df_processed[col] = df_processed[col].fillna(
                df_processed[col].mode()[0]
            )

    # ---------- AFTER ----------
    st.subheader("✅ After Missing Value Handling")
    missing_after = df_processed.isnull().sum()
    summary = pd.DataFrame({
        "Before": missing_before,
        "After": missing_after,
        "Handled": missing_before - missing_after
    })

    st.dataframe(summary)

    # ---------- FEATURE SPLIT ----------
    X = df_processed.drop(columns=[target])
    y = df_processed[target]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # ---------- PIPELINES ----------
    num_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    preprocessor.fit(X)

    label_encoder = None
    if task == "classification":
        label_encoder = LabelEncoder()
        label_encoder.fit(y)

    # ---------- SAVE ----------
    st.session_state.df = df_processed
    st.session_state.preprocessor = preprocessor
    st.session_state.label_encoder = label_encoder
    st.session_state.preprocessing_summary = summary

    st.success("✅ Preprocessing completed successfully")
