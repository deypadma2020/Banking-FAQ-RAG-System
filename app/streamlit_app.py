import streamlit as st
import pandas as pd
import json

from utils.intent_classifier import predict

from utils.relevance_checker import (
    knowledge_base_relevance_check
)

from utils.guardrails import (
    security_gateway,
    generate_rejection
)

from utils.vector_store import (
    retrieve_documents,
    generate_rag_answer,
    format_sources
)

from utils.response_validator import (
    validate_response
)

from utils.multi_llm_judge import (
    run_multi_llm_judge
)

from utils.trust_governance import (
    run_governance_pipeline
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Enterprise Banking AI Assistant",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Enterprise Banking AI Assistant")

st.markdown(
    """
Enterprise RAG Banking Assistant

Features:

- Intent Classification
- Knowledge Relevance Validation
- Guardrails Protection
- FAISS Vector Search
- Groq LLM Generation
- Groundedness Validation
- Multi LLM Judging
- Trust Scoring
- Human Escalation
"""
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("System Information")

st.sidebar.success("Intent Classification")
st.sidebar.success("Knowledge Relevance")
st.sidebar.success("Guardrails")
st.sidebar.success("FAISS Retrieval")
st.sidebar.success("Groq LLM")
st.sidebar.success("Response Validation")
st.sidebar.success("Multi LLM Judge")
st.sidebar.success("Trust Governance")

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

query = st.text_area(
    "Enter Banking Question",
    height=120
)

# --------------------------------------------------
# PROCESS BUTTON
# --------------------------------------------------

if st.button("Ask Banking Assistant"):

    if not query.strip():

        st.warning(
            "Please enter a banking question."
        )

    else:

        with st.spinner(
            "Running Enterprise Banking Pipeline..."
        ):

            # ==================================
            # GUARDRAILS
            # ==================================

            guardrail_result = security_gateway(
                query
            )

            if not guardrail_result["allowed"]:

                rejection_message = (
                    generate_rejection(
                        guardrail_result["report"]
                    )
                )

                st.error(rejection_message)

                st.stop()

            # ==================================
            # INTENT
            # ==================================

            intent_result = predict(query)

            # ==================================
            # RELEVANCE
            # ==================================

            relevance_result = (
                knowledge_base_relevance_check(
                    query
                )
            )

            if (
                relevance_result["final_decision"]
                == "Not Relevant"
            ):

                st.warning(
                    "Query outside banking domain."
                )

                st.stop()

            # ==================================
            # RETRIEVE DOCUMENTS
            # ==================================

            docs = retrieve_documents(
                query,
                k=5
            )

            context = "\n\n".join(
                [
                    doc.page_content
                    for doc in docs
                ]
            )

            # ==================================
            # GENERATE ANSWER
            # ==================================

            answer = generate_rag_answer(
                query,
                context
            )

            # ==================================
            # RESPONSE VALIDATION
            # ==================================

            validation_results = (
                validate_response(
                    query,
                    context,
                    answer
                )
            )

            # ==================================
            # MULTI LLM JUDGE
            # ==================================

            judge_results = (
                run_multi_llm_judge(
                    question=query,
                    context=context,
                    answer=answer,
                    intent_confidence=
                    intent_result["Confidence"],
                    relevance_score=
                    relevance_result[
                        "semantic_similarity"
                    ] * 100
                )
            )

            # ==================================
            # TRUST GOVERNANCE
            # ==================================

            governance = run_governance_pipeline(

                question=query,

                intent_confidence=intent_result["Confidence"],

                relevance_score=relevance_result["semantic_similarity"] * 100,

                groundedness_score=validation_results["grounding"]["confidence"],

                hallucination_score=validation_results["hallucination"]["confidence"],

                consensus_score=judge_results["consensus_score"],

                trust_score=judge_results["trust_score"],

                trust_level=judge_results["trust_level"]
            )

            # ==================================
            # ANSWER
            # ==================================

            st.subheader("AI Answer")

            st.success(answer)

            # ==================================
            # TRUST SCORE
            # ==================================

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Trust Score",
                    governance["trust_score"]
                )

            with col2:

                st.metric(
                    "Trust Level",
                    governance["trust_level"]
                )

            with col3:

                st.metric(
                    "Decision",
                    governance["decision"]["status"]
                )

            # ==================================
            # INTENT DETAILS
            # ==================================

            with st.expander(
                "Intent Classification"
            ):

                st.json(intent_result)

            # ==================================
            # RELEVANCE
            # ==================================

            with st.expander(
                "Knowledge Relevance"
            ):

                st.json(relevance_result)

            # ==================================
            # RESPONSE VALIDATION
            # ==================================

            with st.expander(
                "Response Validation"
            ):

                st.json(validation_results)

            # ==================================
            # MULTI LLM JUDGE
            # ==================================

            with st.expander(
                "Multi LLM Judge"
            ):

                st.json(judge_results)

            # ==================================
            # GOVERNANCE
            # ==================================

            with st.expander(
                "Governance Report"
            ):

                st.json(governance)

            # ==================================
            # SOURCES
            # ==================================

            with st.expander(
                "Retrieved Sources"
            ):

                sources = format_sources(
                    docs
                )

                st.dataframe(
                    pd.DataFrame(sources)
                )

            # ==================================
            # HUMAN ESCALATION
            # ==================================

            if governance["decision"][
                "human_review"
            ]:

                st.error(
                    "Human Review Required."
                )

                st.info(
                    "Escalation Ticket Created."
                )

            else:

                st.success(
                    "No Human Review Required."
                )