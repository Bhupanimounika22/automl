import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    cross_val_score
)
from sklearn.metrics import f1_score, r2_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# ================= CLASSIFICATION MODELS =================
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# ================= REGRESSION MODELS =================
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# ================= DATA BALANCING =================
from imblearn.over_sampling import SMOTE

TARGET_SCORE = 0.90


# ================= DATASET INTELLIGENCE =================
def extract_metadata(df, target):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "numerical_features": len(df.select_dtypes(include="number").columns),
        "categorical_features": len(df.select_dtypes(include="object").columns),
        "missing_percent": round(df.isnull().mean().mean() * 100, 2),
        "target_type": "categorical" if df[target].dtype == "object" else "numeric",
        "num_classes": df[target].nunique()
    }


# ================= TRAINING PAGE =================
def training_page():
    st.title("🤖 AutoML – Train ALL Models")

    if "df" not in st.session_state:
        st.error("❌ Upload & preprocess data first")
        return

    df = st.session_state.df
    target = st.session_state.target
    task = st.session_state.task
    preprocessor = st.session_state.preprocessor

    meta = extract_metadata(df, target)
    with st.expander("🧠 Dataset Intelligence"):
        st.json(meta)

    # ================= DATA SIZE =================
    rows = df.shape[0]
    dataset_size = "small" if rows < 500 else "medium" if rows < 5000 else "large"
    st.info(f"📊 Dataset Size: **{dataset_size.upper()}**")

    # ================= SPLIT =================
    X = df.drop(columns=[target])

    if task == "classification":
        y = st.session_state.label_encoder.transform(df[target])
    else:
        y = df[target].values

    stratify = y if task == "classification" and min(Counter(y).values()) >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )

    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)

    # ================= SMALL DATA FIX =================
    if dataset_size == "small":
        st.warning("⚠️ Small dataset → Enhancing")

        if task == "classification":
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        else:
            noise = np.random.normal(0, 0.01, X_train.shape)
            X_train = np.vstack([X_train, X_train + noise])
            y_train = np.concatenate([y_train, y_train])

    # ================= MODELS =================
    if task == "classification":
        models = {
            "LogisticRegression": LogisticRegression(max_iter=3000, class_weight="balanced"),
            "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced"),
            "ExtraTrees": ExtraTreesClassifier(n_estimators=400, class_weight="balanced"),
            "GradientBoosting": GradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "SVC": SVC(C=10, probability=True, class_weight="balanced"),
            "KNN": KNeighborsClassifier(n_neighbors=7),
            "DecisionTree": DecisionTreeClassifier(class_weight="balanced"),
            "NaiveBayes": GaussianNB()
        }
    else:
        models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.01),
            "RandomForest": RandomForestRegressor(n_estimators=300),
            "ExtraTrees": ExtraTreesRegressor(n_estimators=400),
            "GradientBoosting": GradientBoostingRegressor(),
            "SVR": SVR(C=10),
            "KNN": KNeighborsRegressor(n_neighbors=7),
            "DecisionTree": DecisionTreeRegressor()
        }

    # ================= TRAIN =================
    if st.button("🚀 Train ALL Models"):
        results = []
        best_score = -np.inf
        best_model = None
        best_name = None

        progress = st.progress(0)
        total = len(models)

        for i, (name, model) in enumerate(models.items(), start=1):
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                score = (
                    f1_score(y_test, preds, average="macro")
                    if task == "classification"
                    else r2_score(y_test, preds)
                )

                results.append({"Model": name, "Score": score})

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_name = name

                    # SAVE FOR CONFUSION MATRIX
                    st.session_state.y_test = y_test
                    st.session_state.y_pred = preds

            except Exception:
                results.append({"Model": name, "Score": None})

            progress.progress(i / total)

        # ================= SAVE =================
        st.session_state.model = best_model
        st.session_state.best_model_name = best_name
        st.session_state.final_accuracy = best_score

        df_results = pd.DataFrame(results).sort_values(by="Score", ascending=False)
        st.session_state.model_comparison = df_results

        # ================= DISPLAY =================
        st.subheader("📊 Model Comparison")
        st.dataframe(df_results.style.format({"Score": "{:.4f}"}))

        st.success(f"🏆 Best Model: **{best_name}**")
        st.metric("Best Score", f"{best_score:.4f}")

        # ================= CONFUSION MATRIX =================
        if task == "classification":
            st.subheader("🧩 Confusion Matrix")

            cm = confusion_matrix(
                st.session_state.y_test,
                st.session_state.y_pred
            )

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix – {best_name}")

            st.pyplot(fig)


# ================= RUN =================
training_page()
