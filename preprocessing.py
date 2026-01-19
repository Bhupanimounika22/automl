import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def preprocessing_page():
    st.title("🧹 Data Preprocessing (Transparent Mode)")

    if "df" not in st.session_state or st.session_state.df is None:
        st.error("❌ Upload data first.")
        return

    df = st.session_state.df.copy()

    # ================= ORIGINAL DATA =================
    st.subheader("📄 Original Dataset Preview")
    st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(30), use_container_width=True)

    # ================= TARGET =================
    st.divider()
    target = st.selectbox("🎯 Select Target Column", df.columns)
    st.session_state.target = target

    # ================= TASK =================
    if df[target].dtype == "object" or df[target].nunique() <= 20:
        task = "classification"
    else:
        task = "regression"

    st.session_state.task = task
    st.success(f"🧠 Task Detected: **{task.upper()}**")

    # ================= MISSING VALUES =================
    st.divider()
    st.subheader("❗ Missing Values Analysis")

    missing_before = df.isnull().sum()
    missing_df = missing_before[missing_before > 0].to_frame("Missing Count")

    if missing_df.empty:
        st.info("✅ No missing values found")
    else:
        st.dataframe(missing_df, use_container_width=True)

    # ================= STRATEGY =================
    st.divider()
    st.subheader("⚙️ Missing Value Strategy")

    missing_strategy = st.radio(
        "Choose how to handle missing values:",
        [
            "Fill Missing Values (Recommended)",
            "Drop Rows with Missing",
            "Drop Columns with Missing"
        ]
    )

    df_processed = df.copy()

    if missing_strategy == "Drop Rows with Missing":
        df_processed.dropna(inplace=True)

    elif missing_strategy == "Drop Columns with Missing":
        df_processed.dropna(axis=1, inplace=True)

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

    # ================= SUMMARY =================
    st.divider()
    st.subheader("✅ Missing Value Handling Summary")

    missing_after = df_processed.isnull().sum()
    summary = pd.DataFrame({
        "Before": missing_before,
        "After": missing_after,
        "Handled": missing_before - missing_after
    })

    st.dataframe(summary, use_container_width=True)

    # ================= BEFORE VS AFTER =================
    st.divider()
    st.subheader("🔍 Dataset Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📄 Before")
        st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(15), use_container_width=True)

    with col2:
        st.markdown("### ✨ After Preprocessing")
        st.caption(f"{df_processed.shape[0]} rows × {df_processed.shape[1]} columns")
        st.dataframe(df_processed.head(15), use_container_width=True)

    # ================= FEATURE ENGINEERING =================
    st.divider()
    st.subheader("🧩 Feature Engineering")

    X = df_processed.drop(columns=[target])
    y = df_processed[target]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    st.markdown(
        f"""
        **Numerical Features:** {len(num_cols)}  
        **Categorical Features:** {len(cat_cols)}
        """
    )

    # ================= PIPELINES =================
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

    # ================= LABEL ENCODER =================
    label_encoder = None
    if task == "classification":
        label_encoder = LabelEncoder()
        label_encoder.fit(y)

    # ================= SAVE =================
    st.session_state.df = df_processed
    st.session_state.preprocessor = preprocessor
    st.session_state.label_encoder = label_encoder
    st.session_state.preprocessing_summary = summary

    st.success("🎉 Preprocessing completed successfully!")
