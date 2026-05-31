import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

from scipy.sparse import hstack

# =====================================================
# CONFIG
# =====================================================

CONFIDENCE_THRESHOLD = 60

st.set_page_config(
    page_title="Banking Intent Classification",
    layout="wide"
)

st.title("🏦 Banking FAQ Intent Classification")

st.markdown(
    """
Predict banking query intents using a trained NLP model.
"""
)

# =====================================================
# LOAD MODELS
# =====================================================

@st.cache_resource
def load_models():

    with open(
        "models/intent_classification/best_intent_classifier.pkl",
        "rb"
    ) as f:
        best_model = pickle.load(f)

    with open(
        "models/intent_classification/word_vectorizer.pkl",
        "rb"
    ) as f:
        word_vectorizer = pickle.load(f)

    with open(
        "models/intent_classification/char_vectorizer.pkl",
        "rb"
    ) as f:
        char_vectorizer = pickle.load(f)

    with open(
        "models/intent_classification/label_encoder.pkl",
        "rb"
    ) as f:
        label_encoder = pickle.load(f)

    return (
        best_model,
        word_vectorizer,
        char_vectorizer,
        label_encoder
    )


(
    best_model,
    word_vectorizer,
    char_vectorizer,
    label_encoder
) = load_models()

# =====================================================
# PREPROCESSING
# =====================================================

def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =====================================================
# CONFIDENCE BAND
# =====================================================

def get_confidence_band(confidence):

    if confidence >= 80:
        return "High"

    elif confidence >= 60:
        return "Medium"

    elif confidence >= 40:
        return "Low"

    return "Very Low"


# =====================================================
# FEATURE CREATION
# =====================================================

def create_features(text):

    word_features = word_vectorizer.transform(
        [text]
    )

    char_features = char_vectorizer.transform(
        [text]
    )

    combined_features = hstack([
        word_features,
        char_features
    ])

    return combined_features


# =====================================================
# SINGLE PREDICTION
# =====================================================

def predict_intent(question):

    clean_question = preprocess_text(
        question
    )

    features = create_features(
        clean_question
    )

    prediction = (
        best_model.predict(features)
    )[0]

    predicted_intent = (
        label_encoder
        .inverse_transform([prediction])[0]
    )

    # Handle models without predict_proba()
    if hasattr(best_model, "predict_proba"):

        confidence = (
            best_model
            .predict_proba(features)
            .max()
            * 100
        )

    else:

        confidence = np.nan

    confidence_band = (
        get_confidence_band(confidence)
        if not np.isnan(confidence)
        else "Not Available"
    )

    review_required = (
        "Yes"
        if (
            not np.isnan(confidence)
            and confidence < CONFIDENCE_THRESHOLD
        )
        else "No"
    )

    return {
        "Question": question,
        "clean_question": clean_question,
        "Predicted_Intent": predicted_intent,
        "Confidence": (
            round(confidence, 2)
            if not np.isnan(confidence)
            else "N/A"
        ),
        "Confidence_Band": confidence_band,
        "Review_Required": review_required
    }


# =====================================================
# BATCH PREDICTION
# =====================================================

def predict_dataframe(df):

    df = df.copy()

    if "Question" not in df.columns:

        raise ValueError(
            "Question column not found."
        )

    results = (
        df["Question"]
        .astype(str)
        .apply(predict_intent)
    )

    output_df = pd.DataFrame(
        results.tolist()
    )

    return output_df


# =====================================================
# SIDEBAR
# =====================================================

option = st.sidebar.radio(
    "Choose Prediction Mode",
    [
        "Single Question Prediction",
        "File Prediction"
    ]
)

# =====================================================
# SINGLE QUESTION
# =====================================================

if option == "Single Question Prediction":

    st.header(
        "📝 Single Question Prediction"
    )

    user_query = st.text_area(
        "Enter Banking Question"
    )

    if st.button(
        "Predict Intent"
    ):

        if not user_query.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            result = predict_intent(
                user_query
            )

            st.success(
                f"Predicted Intent: "
                f"{result['Predicted_Intent']}"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Confidence",
                    result["Confidence"]
                )

            with col2:

                st.metric(
                    "Confidence Band",
                    result["Confidence_Band"]
                )

            with col3:

                st.metric(
                    "Review Required",
                    result["Review_Required"]
                )

# =====================================================
# FILE PREDICTION
# =====================================================

elif option == "File Prediction":

    st.header(
        "📂 File Prediction"
    )

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel File",
        type=[
            "csv",
            "xlsx",
            "xls"
        ]
    )

    st.info(
        "File must contain a column named 'Question'"
    )

    if uploaded_file is not None:

        try:

            extension = (
                uploaded_file.name
                .split(".")[-1]
                .lower()
            )

            if extension == "csv":

                try:

                    df = pd.read_csv(
                        uploaded_file,
                        encoding="utf-8"
                    )

                except:

                    df = pd.read_csv(
                        uploaded_file,
                        encoding="latin1"
                    )

            else:

                df = pd.read_excel(
                    uploaded_file
                )

            st.subheader(
                "Uploaded Data"
            )

            st.dataframe(
                df.head()
            )

            if "Question" not in df.columns:

                st.error(
                    "Question column not found."
                )

            else:

                if st.button(
                    "Predict Intents"
                ):

                    prediction_df = (
                        predict_dataframe(df)
                    )

                    total_records = len(
                        prediction_df
                    )

                    avg_confidence = pd.to_numeric(
                        prediction_df["Confidence"],
                        errors="coerce"
                    ).mean()

                    review_required = (
                        prediction_df[
                            "Review_Required"
                        ]
                        .eq("Yes")
                        .sum()
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Total Records",
                            total_records
                        )

                    with col2:

                        st.metric(
                            "Average Confidence",
                            (
                                f"{avg_confidence:.2f}%"
                                if not np.isnan(avg_confidence)
                                else "N/A"
                            )
                        )

                    with col3:

                        st.metric(
                            "Review Required",
                            review_required
                        )

                    st.subheader(
                        "Prediction Results"
                    )

                    st.dataframe(
                        prediction_df
                    )

                    csv = (
                        prediction_df
                        .to_csv(index=False)
                    )

                    st.download_button(
                        label="⬇ Download Predictions",
                        data=csv,
                        file_name="predicted_output.csv",
                        mime="text/csv"
                    )

        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )