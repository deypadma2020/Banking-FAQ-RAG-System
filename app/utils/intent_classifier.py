import re
import pickle
from pathlib import Path

import pandas as pd
from scipy.sparse import hstack


class IntentClassifier:

    CONFIDENCE_THRESHOLD = 60

    def __init__(self):

        project_root = Path(__file__).resolve().parents[2]

        model_path = (
            project_root
            / "models"
            / "intent_classification"
            / "best_intent_classifier.pkl"
        )

        word_vectorizer_path = (
            project_root
            / "models"
            / "intent_classification"
            / "word_vectorizer.pkl"
        )

        char_vectorizer_path = (
            project_root
            / "models"
            / "intent_classification"
            / "char_vectorizer.pkl"
        )

        label_encoder_path = (
            project_root
            / "models"
            / "intent_classification"
            / "label_encoder.pkl"
        )

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(word_vectorizer_path, "rb") as f:
            self.word_vectorizer = pickle.load(f)

        with open(char_vectorizer_path, "rb") as f:
            self.char_vectorizer = pickle.load(f)

        with open(label_encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

        print("✅ Intent Classification Artifacts Loaded")

    # =====================================================
    # PREPROCESSING
    # =====================================================

    @staticmethod
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

    @staticmethod
    def get_confidence_band(confidence):

        if confidence >= 80:
            return "High"

        elif confidence >= 60:
            return "Medium"

        elif confidence >= 40:
            return "Low"

        else:
            return "Very Low"

    # =====================================================
    # FEATURE CREATION
    # =====================================================

    def create_features(self, text_series):

        word_features = (
            self.word_vectorizer.transform(
                text_series
            )
        )

        char_features = (
            self.char_vectorizer.transform(
                text_series
            )
        )

        combined_features = hstack([
            word_features,
            char_features
        ])

        return combined_features

    # =====================================================
    # SINGLE PREDICTION
    # =====================================================

    def predict(self, question):

        clean_question = self.preprocess_text(
            question
        )

        features = self.create_features(
            pd.Series([clean_question])
        )

        prediction = (
            self.model.predict(features)
        )[0]

        predicted_label = (
            self.label_encoder
            .inverse_transform([prediction])[0]
        )

        confidence = 0.0

        if hasattr(
            self.model,
            "predict_proba"
        ):
            confidence = (
                self.model
                .predict_proba(features)
                .max()
                * 100
            )

        confidence = round(
            float(confidence),
            2
        )

        return {

            # Notebook 14 compatibility
            "intent":
            predicted_label,

            "confidence":
            confidence,

            # Streamlit compatibility
            "Question":
            question,

            "clean_question":
            clean_question,

            "Predicted_Intent":
            predicted_label,

            "Confidence":
            confidence,

            "Confidence_Band":
            self.get_confidence_band(
                confidence
            ),

            "Review_Required":
            (
                "Yes"
                if confidence <
                self.CONFIDENCE_THRESHOLD
                else "No"
            )

        }

    # =====================================================
    # BATCH PREDICTION
    # =====================================================

    def predict_dataframe(
        self,
        dataframe,
        question_column="Question"
    ):

        if question_column not in dataframe.columns:

            raise ValueError(
                f"{question_column} column not found."
            )

        results = (
            dataframe[question_column]
            .astype(str)
            .apply(self.predict)
        )

        return pd.DataFrame(
            results.tolist()
        )


# =========================================================
# GLOBAL INSTANCE
# =========================================================

intent_classifier = IntentClassifier()


# =========================================================
# MODULE LEVEL FUNCTIONS
# =========================================================

def predict(question):
    return intent_classifier.predict(
        question
    )


def predict_dataframe(
    dataframe,
    question_column="Question"
):
    return intent_classifier.predict_dataframe(
        dataframe,
        question_column
    )


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    result = predict(
        "How can I reset my internet banking password?"
    )

    print(result)