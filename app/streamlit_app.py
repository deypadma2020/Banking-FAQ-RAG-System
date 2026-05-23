import streamlit as st
import pandas as pd
import pickle
import re

from scipy.sparse import hstack

# Page Config
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

# Load Models
@st.cache_resource
def load_models():

    with open("models/best_intent_classifier.pkl", "rb") as f:
        best_model = pickle.load(f)

    with open("models/word_vectorizer.pkl", "rb") as f:
        word_vectorizer = pickle.load(f)

    with open("models/char_vectorizer.pkl", "rb") as f:
        char_vectorizer = pickle.load(f)

    with open("models/label_encoder.pkl", "rb") as f:
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


# Preprocessing Function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Prediction Function
def predict_intent(query):
    cleaned_query = preprocess_text(query)

    # Word Features
    word_features = word_vectorizer.transform(
        [cleaned_query]
    )

    # Character Features
    char_features = char_vectorizer.transform(
        [cleaned_query]
    )

    # Combine Features
    combined_features = hstack([
        word_features,
        char_features
    ])

    # Predict
    prediction = best_model.predict(
        combined_features
    )

    # Decode Label
    predicted_label = label_encoder.inverse_transform(
        prediction
    )[0]

    return predicted_label


# Sidebar
option = st.sidebar.radio(
    "Choose Prediction Mode",
    [
        "Single Question Prediction",
        "CSV File Prediction"
    ]
)

# SINGLE QUESTION PREDICTION
if option == "Single Question Prediction":
    st.header("📝 Single Question Prediction")
    user_query = st.text_area("Enter Banking Question")

    if st.button("Predict Intent"):
        if user_query.strip() == "":
            st.warning("Please enter a question.")

        else:
            prediction = predict_intent(user_query)
            st.success(f"Predicted Intent: {prediction}")


# CSV FILE PREDICTION
elif option == "CSV File Prediction":
    st.header("📂 CSV File Prediction")
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    st.info("CSV must contain a column named 'Question'")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data")
        st.dataframe(df.head())

        if "Question" not in df.columns:
            st.error( "CSV file must contain 'Question' column.")

        else:
            if st.button("Predict CSV Intents"):

                # Predict
                df["Predicted_Intent"] = df["Question"].apply(predict_intent)

                # Total Predictions
                total_predictions = len(df)

                st.success(f"Total Predictions: {total_predictions}")
                st.subheader("Prediction Results")
                st.dataframe(df.head())

                # Download Button
                csv = df.to_csv(index=False)

                st.download_button(
                    label="⬇ Download Predicted CSV",
                    data=csv,
                    file_name="predicted_output.csv",
                    mime="text/csv"
                )