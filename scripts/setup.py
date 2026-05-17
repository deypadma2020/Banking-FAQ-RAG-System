import nltk
import subprocess
import sys

print("Downloading NLTK resources...")

nltk_resources = [
    "punkt",
    "stopwords",
    "wordnet",
    "omw-1.4"
]

for resource in nltk_resources:
    nltk.download(resource)

print("NLTK setup completed!")

print("Downloading spaCy English model...")

subprocess.check_call([
    sys.executable,
    "-m",
    "spacy",
    "download",
    "en_core_web_sm"
])

print("spaCy setup completed!")
print("All NLP resources installed successfully!")