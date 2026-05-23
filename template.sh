#!/bin/bash

echo "Creating Banking FAQ RAG System Folder Structure..."

# Root folders
mkdir -p app
mkdir -p data/raw_data
mkdir -p data/verification_data
mkdir -p models
mkdir -p notebooks
mkdir -p scripts
mkdir -p test

# App files
touch app/streamlit_app.py

# Data files
touch data/banking_cleaned.csv
touch data/banking_processed.csv

# Notebook files
touch notebooks/01_eda.ipynb
touch notebooks/02_preprocessing.ipynb
touch notebooks/03_intent_classification.ipynb
touch notebooks/04_model_acc_verification.ipynb

# Script files
touch scripts/download_nlp_resources.py
touch scripts/setup_nltk.py
touch scripts/setup_spacy.py

# Test files
touch test/test.ipynb

# Root files
touch .gitignore
touch LICENSE
touch README.md
touch render.yaml
touch requirements.txt
touch runtime.txt
touch setup.py

echo "Folder structure created successfully!"