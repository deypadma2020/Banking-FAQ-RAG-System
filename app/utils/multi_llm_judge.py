# ============================================================
# File: app/utils/multi_llm_judge.py
# ============================================================

import os
import re
import json

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============================================================
# LOAD JUDGE MODELS
# ============================================================

judge_70b = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

judge_8b = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# ============================================================
# JUDGE PROMPT
# ============================================================

judge_prompt = PromptTemplate(
    input_variables=[
        "question",
        "context",
        "answer"
    ],
    template="""
You are a Senior Banking AI Auditor.

Evaluate the answer strictly.

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Evaluate:

1. Groundedness
2. Correctness
3. Hallucination Risk
4. Banking Safety

Scoring Rules:

0 = Completely Wrong

100 = Perfect

Return ONLY JSON.

{{
    "groundedness": number,
    "correctness": number,
    "hallucination": number,
    "safety": number,
    "overall_score": number,
    "verdict":"PASS"
}}

No explanation.
"""
)

# ============================================================
# CLEAN JSON RESPONSE
# ============================================================

def clean_json_response(content):

    content = content.strip()

    content = re.sub(
        r"^```json",
        "",
        content
    )

    content = re.sub(
        r"```$",
        "",
        content
    )

    return content.strip()

# ============================================================
# RUN SINGLE JUDGE
# ============================================================

def run_judge(
    llm,
    question,
    context,
    answer
):

    prompt = judge_prompt.format(
        question=question,
        context=context,
        answer=answer
    )

    try:

        response = llm.invoke(prompt)

        cleaned_content = clean_json_response(
            response.content
        )

        result = json.loads(
            cleaned_content
        )

        return result

    except Exception as e:

        print("Judge Error:", e)

        return {
            "groundedness": 0,
            "correctness": 0,
            "hallucination": 100,
            "safety": 0,
            "overall_score": 0,
            "verdict": "FAIL"
        }

# ============================================================
# TRUST CALCULATION
# ============================================================

def calculate_trust_score(

    relevance_score,
    intent_confidence,

    groundedness_score,
    correctness_score,

    hallucination_score,

    consensus_score,
    agreement_score

):

    score = (

        0.15 * intent_confidence
        +
        0.20 * relevance_score
        +
        0.20 * groundedness_score
        +
        0.15 * correctness_score
        +
        0.10 * (100 - hallucination_score)
        +
        0.10 * consensus_score
        +
        0.10 * agreement_score

    )

    return round(score, 2)

# ============================================================
# TRUST LEVEL
# ============================================================

def get_trust_level(score):

    if score >= 90:
        return "HIGH"

    elif score >= 75:
        return "MEDIUM"

    else:
        return "LOW"

# ============================================================
# FINAL DECISION
# ============================================================

def final_decision(score):

    if score >= 90:

        return {
            "status": "APPROVED",
            "human_review": False
        }

    elif score >= 75:

        return {
            "status": "CAUTION",
            "human_review": False
        }

    else:

        return {
            "status": "REJECTED",
            "human_review": True
        }

# ============================================================
# MAIN MULTI LLM JUDGE
# ============================================================

def evaluate_answer(

    question,
    context,
    answer,

    intent_confidence,
    relevance_score

):

    judge_result_1 = run_judge(
        judge_70b,
        question,
        context,
        answer
    )

    judge_result_2 = run_judge(
        judge_8b,
        question,
        context,
        answer
    )

    # --------------------------------------------------------
    # Consensus Metrics
    # --------------------------------------------------------

    consensus_score = (

        judge_result_1["overall_score"]
        +
        judge_result_2["overall_score"]

    ) / 2

    avg_groundedness = (

        judge_result_1["groundedness"]
        +
        judge_result_2["groundedness"]

    ) / 2

    avg_correctness = (

        judge_result_1["correctness"]
        +
        judge_result_2["correctness"]

    ) / 2

    avg_hallucination = (

        judge_result_1["hallucination"]
        +
        judge_result_2["hallucination"]

    ) / 2

    avg_safety = (

        judge_result_1["safety"]
        +
        judge_result_2["safety"]

    ) / 2

    # --------------------------------------------------------
    # Agreement Score
    # --------------------------------------------------------

    scores = [

        judge_result_1["overall_score"],
        judge_result_2["overall_score"]

    ]

    agreement_score = 100 - (
        max(scores) - min(scores)
    )

    # --------------------------------------------------------
    # Trust Score
    # --------------------------------------------------------

    trust_score = calculate_trust_score(

        relevance_score=relevance_score,

        intent_confidence=intent_confidence,

        groundedness_score=avg_groundedness,

        correctness_score=avg_correctness,

        hallucination_score=avg_hallucination,

        consensus_score=consensus_score,

        agreement_score=agreement_score

    )

    trust_level = get_trust_level(
        trust_score
    )

    decision = final_decision(
        trust_score
    )

    return {

        "question": question,

        "intent_confidence": intent_confidence,

        "relevance_score": relevance_score,

        "groundedness_score": round(
            avg_groundedness,
            2
        ),

        "correctness_score": round(
            avg_correctness,
            2
        ),

        "hallucination_score": round(
            avg_hallucination,
            2
        ),

        "safety_score": round(
            avg_safety,
            2
        ),

        "consensus_score": round(
            consensus_score,
            2
        ),

        "agreement_score": round(
            agreement_score,
            2
        ),

        "trust_score": trust_score,

        "trust_level": trust_level,

        "decision": decision,

        "judge_70b": judge_result_1,

        "judge_8b": judge_result_2

    }

# ============================================================
# STREAMLIT HELPER
# ============================================================

def predict(
    question,
    context,
    answer,
    intent_confidence,
    relevance_score
):

    return evaluate_answer(
        question=question,
        context=context,
        answer=answer,
        intent_confidence=intent_confidence,
        relevance_score=relevance_score
    )


# ============================================================
# STREAMLIT COMPATIBILITY FUNCTION
# ============================================================

def run_multi_llm_judge(
    question,
    context,
    answer,
    intent_confidence,
    relevance_score
):

    return evaluate_answer(
        question=question,
        context=context,
        answer=answer,
        intent_confidence=intent_confidence,
        relevance_score=relevance_score
    )