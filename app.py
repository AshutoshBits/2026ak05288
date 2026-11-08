from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

model_files = {
    "Logistic Regression": "logistic.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.title("Handwritten Digit Classification")
selected_model = st.selectbox("Choose a classifier", list(model_files))
uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    features = [f"pixel_{row}_{col}" for row in range(8) for col in range(8)]
    missing = [name for name in features + ["digit"] if name not in data.columns]

    if missing or data.empty:
        st.error("Upload test_data.csv with all 64 pixel columns, the digit column, and test rows.")
        st.stop()

     app_folder = Path(__file__).resolve().parent
    filename = model_files[selected_model]

    model_path = app_folder / "model" / filename
    if not model_path.is_file():
        model_path = app_folder / filename

    if not model_path.is_file():
        st.error(f"Missing model file: {filename}. Upload it to the model folder on GitHub.")
        st.stop()

    model = joblib.load(model_path)

    X = data.loc[:, features]
    y = data["digit"]
    predictions = model.predict(X)
    labels = model.classes_

    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predictions, labels=labels, average="macro", zero_division=0
    )
    auc = None
    if set(y) == set(labels):
        auc = roc_auc_score(
            y, model.predict_proba(X), labels=labels,
            multi_class="ovr", average="macro"
        )

    scores = {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": matthews_corrcoef(y, predictions),
    }
    matrix = pd.DataFrame(
        confusion_matrix(y, predictions, labels=labels),
        index=pd.Index(labels, name="Actual"),
        columns=pd.Index(labels, name="Predicted"),
    )

    st.write(f"Evaluated {len(data)} test rows")
    st.subheader("Evaluation metrics")
    st.dataframe(pd.DataFrame([scores], index=[selected_model]).round(4))
    st.subheader("Confusion matrix")
    st.dataframe(matrix)
