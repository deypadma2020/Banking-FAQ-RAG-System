# app/utils/response_validator.py

import os
import json
import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============================================================
# VALIDATION LLM
# ============================================================

validator_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# ============================================================
# SAFE JSON PARSER
# ============================================================

def extract_json(response_text):

    try:

        match = re.search(
            r"\{.*\}",
            response_text,
            re.DOTALL
        )

        if match:
            return json.loads(match.group())

    except:
        pass

    return None


# ============================================================
# GROUNDEDNESS VALIDATION
# ============================================================

GROUNDING_PROMPT = """
You are a Banking AI Validator.

User Question:
{query}

Retrieved Context:
{context}

Generated Answer:
{answer}

Evaluate:

1. Is answer fully supported by context?
2. Any unsupported claims?
3. Any missing evidence?

Return JSON:

{{
    "grounded": true,
    "confidence": 0,
    "reason": ""
}}
"""


def validate_groundedness(
    query,
    context,
    answer
):

    prompt = GROUNDING_PROMPT.format(
        query=query,
        context=context,
        answer=answer
    )

    response = validator_llm.invoke(prompt)

    parsed = extract_json(
        response.content
    )

    if parsed:
        return parsed

    return {
        "grounded": False,
        "confidence": 0,
        "reason": "Unable to validate"
    }


# ============================================================
# HALLUCINATION DETECTION
# ============================================================

HALLUCINATION_PROMPT = """
You are a Banking Hallucination Detector.

Question:
{query}

Retrieved Context:
{context}

Answer:
{answer}

Identify:

1. Unsupported statements
2. Fabricated facts
3. Invented banking policies
4. Incorrect regulations

Return JSON:

{{
    "hallucination": false,
    "confidence": 0,
    "reason": ""
}}
"""


def detect_hallucination(
    query,
    context,
    answer
):

    prompt = HALLUCINATION_PROMPT.format(
        query=query,
        context=context,
        answer=answer
    )

    response = validator_llm.invoke(prompt)

    parsed = extract_json(
        response.content
    )

    if parsed:
        return parsed

    return {
        "hallucination": True,
        "confidence": 0,
        "reason": "Unable to validate"
    }


# ============================================================
# EVIDENCE COVERAGE
# ============================================================

COVERAGE_PROMPT = """
You are a Banking Evidence Validator.

Question:
{query}

Context:
{context}

Answer:
{answer}

Rate evidence coverage.

Return JSON:

{{
    "coverage_score": 0,
    "reason": ""
}}
"""


def check_evidence_coverage(
    query,
    context,
    answer
):

    prompt = COVERAGE_PROMPT.format(
        query=query,
        context=context,
        answer=answer
    )

    response = validator_llm.invoke(prompt)

    parsed = extract_json(
        response.content
    )

    if parsed:
        return parsed

    return {
        "coverage_score": 0,
        "reason": "Unable to validate"
    }


# ============================================================
# RELEVANCE VALIDATION
# ============================================================

RELEVANCE_PROMPT = """
You are a Banking Relevance Validator.

Question:
{query}

Answer:
{answer}

Evaluate:

1. Relevance
2. Completeness
3. Directness

Return JSON:

{{
    "relevant": true,
    "score": 0,
    "reason": ""
}}
"""


def validate_relevance(
    query,
    answer
):

    prompt = RELEVANCE_PROMPT.format(
        query=query,
        answer=answer
    )

    response = validator_llm.invoke(prompt)

    parsed = extract_json(
        response.content
    )

    if parsed:
        return parsed

    return {
        "relevant": False,
        "score": 0,
        "reason": "Unable to validate"
    }


# ============================================================
# COMPLETE VALIDATION PIPELINE
# ============================================================

def validate_response(
    query,
    context,
    answer
):

    grounding = validate_groundedness(
        query,
        context,
        answer
    )

    hallucination = detect_hallucination(
        query,
        context,
        answer
    )

    coverage = check_evidence_coverage(
        query,
        context,
        answer
    )

    relevance = validate_relevance(
        query,
        answer
    )

    return {

        "grounding": grounding,

        "hallucination": hallucination,

        "coverage": coverage,

        "relevance": relevance

    }


# ============================================================
# NUMERIC SCORE EXTRACTION
# ============================================================

def get_validation_scores(
    validation_report
):

    grounded = validation_report[
        "grounding"
    ].get(
        "grounded",
        False
    )

    groundedness_score = validation_report[
        "grounding"
    ].get(
        "confidence",
        0
    )

    hallucination_score = (
        100 -
        validation_report[
            "hallucination"
        ].get(
            "confidence",
            0
        )
    )

    coverage_score = validation_report[
        "coverage"
    ].get(
        "coverage_score",
        0
    )

    relevance_score = validation_report[
        "relevance"
    ].get(
        "score",
        0
    )

    return {

        "grounded":
        grounded,

        "groundedness_score":
        groundedness_score,

        "hallucination_score":
        hallucination_score,

        "coverage_score":
        coverage_score,

        "relevance_score":
        relevance_score

    }


# ============================================================
# VALIDATION GATE
# ============================================================

def validation_gate(
    grounded,
    hallucination_detected,
    coverage_score,
    relevance_score
):

    if not grounded:
        return "FAIL"

    if hallucination_detected:
        return "FAIL"

    if coverage_score < 70:
        return "FAIL"

    if relevance_score < 70:
        return "FAIL"

    return "PASS"


# ============================================================
# HIGH LEVEL VALIDATOR
# ============================================================

def run_validation_pipeline(
    query,
    context,
    answer
):

    report = validate_response(
        query,
        context,
        answer
    )

    scores = get_validation_scores(
        report
    )

    hallucination_detected = report[
        "hallucination"
    ].get(
        "hallucination",
        False
    )

    gate_status = validation_gate(

        grounded=scores["grounded"],

        hallucination_detected=
        hallucination_detected,

        coverage_score=
        scores["coverage_score"],

        relevance_score=
        scores["relevance_score"]

    )

    return {

        "validation_report":
        report,

        "scores":
        scores,

        "status":
        gate_status

    }


# ============================================================
# HEALTH CHECK
# ============================================================

def health_check():

    try:

        response = validator_llm.invoke(
            "Reply with OK"
        )

        return {
            "status": "healthy",
            "response":
            response.content
        }

    except Exception as e:

        return {
            "status": "failed",
            "error": str(e)
        }