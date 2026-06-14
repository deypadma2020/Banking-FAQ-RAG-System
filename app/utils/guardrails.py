# app/utils/guardrails.py

import os
import re
import json

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# =====================================================
# LLM CONFIG
# =====================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

guardrail_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# =====================================================
# BANKING POLICIES
# =====================================================

BANKING_POLICIES = {

    "allowed_topics": [

        "accounts",
        "cards",
        "payments",
        "loans",
        "insurance",
        "banking_security",
        "customer_support",
        "banking_procedures",
        "government_schemes"

    ],

    "restricted_topics": [

        "loan_approval",
        "investment_advice",
        "stock_recommendation",
        "credit_decision",
        "internal_policy",
        "employee_information",
        "database_access",
        "system_prompt"

    ]
}

# =====================================================
# PROMPT INJECTION
# =====================================================

INJECTION_PATTERNS = [

    r"ignore previous instructions",
    r"forget previous instructions",
    r"reveal system prompt",
    r"show system prompt",
    r"developer message",
    r"act as",
    r"pretend to be",
    r"you are now",
    r"bypass security",
    r"override rules"

]

def detect_prompt_injection(query):

    query = str(query).lower()

    for pattern in INJECTION_PATTERNS:

        if re.search(pattern, query):

            return True

    return False

# =====================================================
# JAILBREAK DETECTION
# =====================================================

JAILBREAK_PATTERNS = [

    r"dan mode",
    r"developer mode",
    r"jailbreak",
    r"root access",
    r"admin mode",
    r"unrestricted mode",
    r"do anything now",
    r"bypass restrictions"

]

def detect_jailbreak(query):

    query = str(query).lower()

    for pattern in JAILBREAK_PATTERNS:

        if re.search(pattern, query):

            return True

    return False

# =====================================================
# SENSITIVE REQUEST DETECTION
# =====================================================

SENSITIVE_PATTERNS = [

    r"customer password",
    r"password database",
    r"password list",
    r"employee password",
    r"customer database",
    r"account numbers",
    r"employee records",
    r"credit card details",
    r"cvv",
    r"pin number",
    r"internal database",
    r"internal system"

]

def detect_sensitive_request(query):

    query = str(query).lower()

    for pattern in SENSITIVE_PATTERNS:

        if re.search(pattern, query):

            return True

    return False

# =====================================================
# PROMPT LEAKAGE DETECTION
# =====================================================

PROMPT_LEAK_PATTERNS = [

    r"system prompt",
    r"hidden prompt",
    r"internal instructions",
    r"show configuration",
    r"print instructions",
    r"reveal prompt"

]

def detect_prompt_leakage(query):

    query = str(query).lower()

    for pattern in PROMPT_LEAK_PATTERNS:

        if re.search(pattern, query):

            return True

    return False

# =====================================================
# COMPLIANCE CHECK PROMPT
# =====================================================

COMPLIANCE_PROMPT = """
You are a Banking Compliance Officer.

Question:
{query}

Determine:

1. Is this query allowed?
2. Does it violate banking policies?
3. Is it asking for financial advice?
4. Is it requesting internal information?

Return ONLY JSON:

{{
    "allowed": true,
    "risk_level": "LOW",
    "reason": "short explanation"
}}
"""

def compliance_check(query):

    try:

        prompt = COMPLIANCE_PROMPT.format(
            query=query
        )

        response = guardrail_llm.invoke(
            prompt
        )

        content = response.content.strip()

        try:
            return json.loads(content)

        except Exception:

            return {
                "allowed": True,
                "risk_level": "LOW",
                "reason": content
            }

    except Exception as e:

        return {
            "allowed": True,
            "risk_level": "LOW",
            "reason": str(e)
        }

# =====================================================
# GUARDRAIL REPORT
# =====================================================

def run_guardrails(query):

    report = {

        "prompt_injection":
        detect_prompt_injection(query),

        "jailbreak":
        detect_jailbreak(query),

        "sensitive_request":
        detect_sensitive_request(query),

        "prompt_leakage":
        detect_prompt_leakage(query)

    }

    return report

# =====================================================
# SECURITY GATEWAY
# =====================================================

def security_gateway(query):

    report = run_guardrails(query)

    if any(report.values()):

        return {

            "allowed": False,
            "report": report

        }

    return {

        "allowed": True,
        "report": report

    }

# =====================================================
# REJECTION MESSAGES
# =====================================================

REJECTION_MESSAGES = {

    "prompt_injection":
    "Request rejected due to prompt manipulation attempt.",

    "jailbreak":
    "Request rejected due to security policy violation.",

    "sensitive_request":
    "Access to sensitive banking information is not permitted.",

    "prompt_leakage":
    "System instructions cannot be disclosed."

}

def generate_rejection(report):

    for key, value in report.items():

        if value:

            return REJECTION_MESSAGES.get(
                key,
                "Request blocked by security policy."
            )

    return None

# =====================================================
# COMPLETE VALIDATION
# =====================================================

def validate_query(query):

    security_result = security_gateway(
        query
    )

    if not security_result["allowed"]:

        return {

            "allowed": False,

            "reason":
            generate_rejection(
                security_result["report"]
            ),

            "report":
            security_result["report"]

        }

    compliance_result = compliance_check(
        query
    )

    return {

        "allowed": True,

        "reason": "Query Passed",

        "report":
        security_result["report"],

        "compliance":
        compliance_result

    }

# =====================================================
# SIMPLE ENTRY FUNCTION
# =====================================================

def check_guardrails(query):

    return validate_query(query)