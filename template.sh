#!/bin/bash

echo "Creating Enterprise Banking FAQ RAG System Folder Structure..."

# =========================================================
# ROOT FOLDERS
# =========================================================

mkdir -p app

mkdir -p data/raw_data
mkdir -p data/processed_data
mkdir -p data/verification_data

mkdir -p models/intent_classification
mkdir -p models/embeddings
mkdir -p models/rag
mkdir -p models/validation

mkdir -p vectorstore/faiss_index

mkdir -p notebooks

mkdir -p scripts

mkdir -p config

mkdir -p prompts

mkdir -p evaluation

mkdir -p logs

mkdir -p test

# =========================================================
# APP FILES
# =========================================================

touch app/streamlit_app.py
touch app/rag_chatbot_app.py
touch app/enterprise_banking_chatbot.py

# =========================================================
# DATA FILES
# =========================================================

touch data/processed_data/banking_cleaned.csv
touch data/processed_data/banking_processed.csv

# =========================================================
# NOTEBOOK FILES
# =========================================================

touch notebooks/01_eda.ipynb
touch notebooks/01_edawithllmclassification.ipynb
touch notebooks/02_preprocessing.ipynb
touch notebooks/03_intent_classification.ipynb
touch notebooks/04_model_acc_verification.ipynb

touch notebooks/05_semantic_search.ipynb
touch notebooks/06_faiss_vector_db.ipynb
touch notebooks/07_rag_pipeline.ipynb
touch notebooks/08_generative_chatbot.ipynb

# Enterprise GenAI Notebooks
touch notebooks/09_response_validation.ipynb
touch notebooks/10_guardrails.ipynb
touch notebooks/11_enterprise_rag_pipeline.ipynb

# =========================================================
# SCRIPT FILES
# =========================================================

touch scripts/download_nlp_resources.py
touch scripts/setup_nltk.py
touch scripts/setup_spacy.py

touch scripts/train_intent_model.py
touch scripts/build_faiss_index.py
touch scripts/run_rag_pipeline.py

touch scripts/response_validator.py
touch scripts/hallucination_detector.py
touch scripts/guardrails.py

# =========================================================
# CONFIG FILES
# =========================================================

touch config/model_config.py
touch config/prompt_config.py
touch config/settings.py

# =========================================================
# PROMPT FILES
# =========================================================

touch prompts/banking_prompt.txt
touch prompts/judge_prompt.txt
touch prompts/guardrail_prompt.txt

# =========================================================
# EVALUATION FILES
# =========================================================

touch evaluation/rag_evaluation.ipynb
touch evaluation/hallucination_analysis.ipynb

# =========================================================
# TEST FILES
# =========================================================

touch test/test.ipynb
touch test/test_rag.ipynb
touch test/test_guardrails.ipynb

# =========================================================
# ROOT FILES
# =========================================================

touch .gitignore

touch .env
touch .env.example

touch LICENSE
touch README.md

touch requirements.txt
touch runtime.txt
touch render.yaml
touch setup.py

# =========================================================
# ENVIRONMENT VARIABLES TEMPLATE
# =========================================================

cat <<EOL > .env.example

# =========================================================
# API KEYS
# =========================================================

GROQ_API_KEY=your_groq_api_key_here

GEMINI_API_KEY=your_gemini_api_key_here

OPENAI_API_KEY=your_openai_api_key_here

EOL

# =========================================================
# GITIGNORE CONTENT
# =========================================================

cat <<EOL > .gitignore

# =========================================================
# Python
# =========================================================

__pycache__/
*.pyc
*.pyo
*.pyd

# =========================================================
# Virtual Environment
# =========================================================

banking_ai_assistant_venv/
venv/
env/

# =========================================================
# Jupyter
# =========================================================

.ipynb_checkpoints/

# =========================================================
# Environment Variables
# =========================================================

.env

# =========================================================
# Data Files
# =========================================================

data/raw_data/
data/processed_data/
data/verification_data/

# =========================================================
# Vector Database
# =========================================================

vectorstore/faiss_index/

# =========================================================
# Models
# =========================================================

models/

# =========================================================
# Logs
# =========================================================

logs/

# =========================================================
# OS Files
# =========================================================

.DS_Store
Thumbs.db

EOL

echo "Enterprise Banking FAQ RAG System Structure Created Successfully!"