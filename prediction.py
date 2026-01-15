import streamlit as st
import pandas as pd

def prediction_page():
    st.title("🔮 Prediction")

    if st.session_state.model is None:
        st.error("❌ Train a model first.")
        return

    df = st.session_state.df
    target = st.session_state.target
    task = st.session_state.task

    inputs = {}
    for col in df.drop(columns=[target]).columns:
        if df[col].dtype == "object":
            inputs[col] = st.selectbox(col, sorted(df[col].dropna().unique()))
        else:
            inputs[col] = st.number_input(col, value=float(df[col].dropna().mean()))

    if st.button("Predict"):
        X = pd.DataFrame([inputs])
        X = st.session_state.preprocessor.transform(X)
        pred = st.session_state.model.predict(X)[0]

        if task == "classification":
            pred = st.session_state.label_encoder.inverse_transform([int(pred)])[0]
            st.success(f"🎯 Prediction: **{pred}**")
        else:
            st.success(f"📈 Prediction: **{pred:.4f}**")
