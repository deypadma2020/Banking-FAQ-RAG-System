# 🏦 Banking FAQ Intent Classification System

An end-to-end NLP-based Banking FAQ Intent Classification System built using:

- TF-IDF Feature Engineering
- Machine Learning Models
- Ensemble Learning
- Streamlit Deployment
- Render Cloud Hosting

The project classifies banking-related customer queries into predefined intent categories such as:
- UPI Services
- Loans
- Debit Card
- Mutual Funds
- Account Services
- Financial Markets
- and more.

---

# 🚀 Project Overview

This project demonstrates a complete NLP pipeline for banking query intent classification.

The system:
- preprocesses banking text data
- generates TF-IDF features
- trains multiple ML models
- evaluates model performance
- predicts intents for new queries
- supports bulk CSV prediction
- deploys as a Streamlit web application

---

# 📌 Features

## ✅ Single Question Prediction
Users can enter a banking question and instantly receive the predicted intent.

Example:
```text
How can I block my debit card?
```

Output:
```text
Debit Card Services
```

---

## ✅ Bulk CSV Prediction
Users can upload a CSV file containing multiple banking questions.

The application:
- predicts intents for all questions
- displays total predictions
- allows downloading the predicted CSV file

---

## ✅ NLP Pipeline
Implemented:
- text preprocessing
- regex cleaning
- TF-IDF vectorization
- word-level features
- character-level features

---

## ✅ Machine Learning Models
Models evaluated:
- Logistic Regression
- Naive Bayes
- Random Forest
- LinearSVC
- Ensemble Voting Classifier

---

## ✅ Best Performing Model
`LinearSVC` achieved the best overall performance for sparse TF-IDF text classification.

---

# 🧠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Streamlit
- Pickle
- Render

---

# 📂 Project Structure

```text
BANKING-FAQ-RAG-SYSTEM/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw_data/
│   ├── verification_data/
│   ├── banking_cleaned.csv
│   └── banking_processed.csv
│
├── models/
│   ├── best_model.pkl
│   ├── word_vectorizer.pkl
│   ├── char_vectorizer.pkl
│   └── label_encoder.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_intent_classification.ipynb
│   └── 04_model_acc_verification.ipynb
│
├── scripts/
│   ├── download_nlp_resources.py
│   ├── setup_nltk.py
│   └── setup_spacy.py
│
├── test/
│   └── test.ipynb
│
├── .gitignore
├── LICENSE
├── README.md
├── render.yaml
├── requirements.txt
├── runtime.txt
├── setup.py
└── template.sh
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/Banking-FAQ-RAG-System.git
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

### Windows
```bash
venv\Scripts\activate
```

### Linux / Mac
```bash
source venv/bin/activate
```

---

## 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Streamlit Application

```bash
streamlit run app/streamlit_app.py
```

---

# 🌐 Render Deployment

This project is deployment-ready for Render.

## Deployment Files Included

- `render.yaml`
- `runtime.txt`
- `requirements.txt`

---

# 📊 Machine Learning Workflow

## 1. Data Preprocessing
- lowercasing
- punctuation removal
- whitespace normalization

---

## 2. Feature Engineering

### Word-Level TF-IDF
```python
ngram_range=(1,3)
```

### Character-Level TF-IDF
```python
analyzer='char_wb'
ngram_range=(3,5)
```

---

## 3. Sparse Feature Combination

```python
hstack([word_features, char_features])
```

---

## 4. Model Training

Trained Models:
- Logistic Regression
- Naive Bayes
- LinearSVC
- Ensemble Voting Classifier

---

# 📈 Model Evaluation

Evaluation Metrics:
- Training Accuracy
- Testing Accuracy
- Cross Validation Accuracy
- Classification Report
- Overfitting Gap

---

# 🧪 Verification System

A separate verification dataset was used to:
- validate predictions
- calculate real-world accuracy
- generate classification reports

---

# 📦 Model Artifacts

Saved using `pickle`:
- trained model
- vectorizers
- label encoder

These files are used directly during Streamlit inference.

---

# 🖥️ Streamlit Application Functionalities

## Single Query Prediction
Input:
```text
How to activate UPI?
```

Output:
```text
UPI Services
```

---

## Bulk CSV Prediction

Input CSV:
```csv
Question
How to activate UPI?
What is SWIFT transfer?
```

Output CSV:
```csv
Question,Predicted_Intent
How to activate UPI?,UPI Services
What is SWIFT transfer?,International Banking
```

---

# 📌 Future Improvements

Possible future enhancements:
- BERT / Transformer Models
- Semantic Search
- FAISS Vector Database
- Retrieval-Augmented Generation (RAG)
- Banking Chatbot Integration
- FastAPI Backend
- Docker Deployment

---

# 👨‍💻 Author

**Tuktuki Halder**

---

# 📜 License

This project is licensed under the MIT License.

---

# ⭐ If You Like This Project

Please consider giving this repository a ⭐ on GitHub.
