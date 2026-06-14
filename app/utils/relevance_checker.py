import os
import re
import pickle
from pathlib import Path

import numpy as np

from dotenv import load_dotenv

from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ==========================================================
# LOAD ARTIFACTS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


with open(
    PROJECT_ROOT /
    "models" /
    "intent_classification" /
    "best_intent_classifier.pkl",
    "rb"
) as f:

    best_model = pickle.load(f)


with open(
    PROJECT_ROOT /
    "models" /
    "intent_classification" /
    "word_vectorizer.pkl",
    "rb"
) as f:

    word_vectorizer = pickle.load(f)


with open(
    PROJECT_ROOT /
    "models" /
    "intent_classification" /
    "char_vectorizer.pkl",
    "rb"
) as f:

    char_vectorizer = pickle.load(f)


with open(
    PROJECT_ROOT /
    "models" /
    "intent_classification" /
    "label_encoder.pkl",
    "rb"
) as f:

    label_encoder = pickle.load(f)


with open(
    PROJECT_ROOT /
    "models" /
    "relevance" /
    "banking_reference_embeddings.pkl",
    "rb"
) as f:

    banking_embeddings = pickle.load(f)


# ==========================================================
# EMBEDDINGS
# ==========================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("✅ Relevance Checker Loaded")


# ==========================================================
# LLM
# ==========================================================

judge_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0
)
# judge_llm = None


# ==========================================================
# PREPROCESS
# ==========================================================

def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ==========================================================
# INTENT CONFIDENCE
# ==========================================================

def get_intent_confidence(question):

    clean_question = preprocess_text(
        question
    )

    word_features = (
        word_vectorizer.transform(
            [clean_question]
        )
    )

    char_features = (
        char_vectorizer.transform(
            [clean_question]
        )
    )

    features = hstack([
        word_features,
        char_features
    ])

    prediction = (
        best_model.predict(features)
    )

    predicted_label = (
        label_encoder.inverse_transform(
            prediction
        )[0]
    )

    confidence = (
        best_model.predict_proba(features)
        .max()
        * 100
    )

    return (
        predicted_label,
        round(float(confidence), 2)
    )


# ==========================================================
# SIMILARITY SCORE
# ==========================================================

def get_similarity_score(question):

    query_embedding = (
        embeddings.embed_query(
            question
        )
    )

    similarities = cosine_similarity(
        [query_embedding],
        banking_embeddings
    )[0]

    return float(
        similarities.max()
    )


# ==========================================================
# LLM RELEVANCE
# ==========================================================

RELEVANCE_PROMPT = """
You are a Banking Query Validator.

Determine whether the user's query belongs
to the banking domain.

Return ONLY:

Relevant

or

Not Relevant

Query:

{query}
"""


def llm_relevance_check(query):
    # if judge_llm is None:
    #     return "Relevant"

    prompt = (
        RELEVANCE_PROMPT
        .format(query=query)
    )

    response = (
        judge_llm.invoke(prompt)
    )

    return response.content.strip()


# ==========================================================
# MAIN FUNCTION
# ==========================================================

def knowledge_base_relevance_check(question):

    predicted_category, confidence = (
        get_intent_confidence(question)
    )

    similarity_score = (
        get_similarity_score(question)
    )

    try:

        llm_decision = (
            llm_relevance_check(question)
        )

    except:

        llm_decision = "Relevant"

    result = {

        "question":
        question,

        "predicted_category":
        predicted_category,

        "intent_confidence":
        confidence,

        "semantic_similarity":
        round(similarity_score, 4),

        "llm_decision":
        llm_decision
    }

    if (
        confidence >= 50
        and similarity_score >= 0.55
        and llm_decision == "Relevant"
    ):

        result["final_decision"] = (
            "Relevant"
        )

    elif similarity_score >= 0.40:

        result["final_decision"] = (
            "Partially Relevant"
        )

    else:

        result["final_decision"] = (
            "Not Relevant"
        )

    return result