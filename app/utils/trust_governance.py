# ============================================================
# File: app/utils/trust_governance.py
# ============================================================

import os
import json
import uuid

from datetime import datetime

# ============================================================
# DIRECTORY SETUP
# ============================================================

ESCALATION_DIR = "data/escalations"
LOG_DIR = "logs"

os.makedirs(
    ESCALATION_DIR,
    exist_ok=True
)

os.makedirs(
    LOG_DIR,
    exist_ok=True
)

# ============================================================
# RESPONSE DECISION
# ============================================================

def determine_response_action(
    trust_score
):

    if trust_score >= 90:

        return {
            "status": "APPROVED",
            "show_response": True,
            "human_review": False
        }

    elif trust_score >= 75:

        return {
            "status": "CAUTION",
            "show_response": True,
            "human_review": False
        }

    else:

        return {
            "status": "ESCALATE",
            "show_response": False,
            "human_review": True
        }

# ============================================================
# SESSION ID
# ============================================================

def generate_session_id():

    timestamp = datetime.now().strftime(
        "%Y%m%d%H%M%S"
    )

    random_part = str(uuid.uuid4())[:6]

    return f"{timestamp}{random_part}"

# ============================================================
# ESCALATION TICKET
# ============================================================

def create_escalation_ticket(

    question,
    trust_score,
    trust_level

):

    ticket = {

        "session_id":
        generate_session_id(),

        "timestamp":
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "question":
        question,

        "trust_score":
        trust_score,

        "trust_level":
        trust_level,

        "status":
        "OPEN"

    }

    return ticket

# ============================================================
# SAVE TICKET
# ============================================================

def save_ticket(ticket):

    file_name = (

        f"{ESCALATION_DIR}/"
        f"{ticket['session_id']}.json"

    )

    with open(
        file_name,
        "w"
    ) as f:

        json.dump(
            ticket,
            f,
            indent=4
        )

    return file_name

# ============================================================
# AUDIT LOG
# ============================================================

def save_audit_log(

    question,

    intent_confidence,

    relevance_score,

    groundedness_score,

    hallucination_score,

    consensus_score,

    trust_score,

    trust_level,

    final_decision

):

    audit_log = {

        "session_id":
        generate_session_id(),

        "timestamp":
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "question":
        question,

        "intent_confidence":
        intent_confidence,

        "relevance_score":
        relevance_score,

        "groundedness_score":
        groundedness_score,

        "hallucination_score":
        hallucination_score,

        "consensus_score":
        consensus_score,

        "trust_score":
        trust_score,

        "trust_level":
        trust_level,

        "decision":
        final_decision["status"]

    }

    file_path = (
        f"{LOG_DIR}/audit_log.json"
    )

    with open(
        file_path,
        "w"
    ) as f:

        json.dump(
            audit_log,
            f,
            indent=4
        )

    return file_path

# ============================================================
# GOVERNANCE REPORT
# ============================================================

def save_governance_report(

    trust_score,
    trust_level,
    final_decision

):

    governance_report = {

        "validation_layers": [

            "Intent Classification",

            "Knowledge Base Relevance",

            "Groundedness Validation",

            "Hallucination Detection",

            "Multi LLM Judge"

        ],

        "trust_score":
        trust_score,

        "trust_level":
        trust_level,

        "final_status":
        final_decision["status"]

    }

    file_path = (
        f"{LOG_DIR}/governance_report.json"
    )

    with open(
        file_path,
        "w"
    ) as f:

        json.dump(
            governance_report,
            f,
            indent=4
        )

    return file_path

# ============================================================
# MAIN GOVERNANCE PIPELINE
# ============================================================

def run_governance_pipeline(

    question,

    intent_confidence,

    relevance_score,

    groundedness_score,

    hallucination_score,

    consensus_score,

    trust_score,

    trust_level

):

    final_decision = determine_response_action(
        trust_score
    )

    ticket_path = None

    if final_decision["human_review"]:

        ticket = create_escalation_ticket(

            question,
            trust_score,
            trust_level

        )

        ticket_path = save_ticket(
            ticket
        )

    audit_log_path = save_audit_log(

        question,

        intent_confidence,

        relevance_score,

        groundedness_score,

        hallucination_score,

        consensus_score,

        trust_score,

        trust_level,

        final_decision

    )

    governance_report_path = (
        save_governance_report(

            trust_score,
            trust_level,
            final_decision

        )
    )

    return {

        "question":
        question,

        "trust_score":
        trust_score,

        "trust_level":
        trust_level,

        "decision":
        final_decision,

        "ticket_path":
        ticket_path,

        "audit_log":
        audit_log_path,

        "governance_report":
        governance_report_path

    }

# ============================================================
# STREAMLIT HELPER
# ============================================================

def predict(

    question,

    intent_confidence,

    relevance_score,

    groundedness_score,

    hallucination_score,

    consensus_score,

    trust_score,

    trust_level

):

    return run_governance_pipeline(

        question=question,

        intent_confidence=intent_confidence,

        relevance_score=relevance_score,

        groundedness_score=groundedness_score,

        hallucination_score=hallucination_score,

        consensus_score=consensus_score,

        trust_score=trust_score,

        trust_level=trust_level

    )