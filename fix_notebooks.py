"""
fix_notebooks.py

Purpose:
1. Validate all notebooks
2. Rewrite them using nbformat v4
3. Remove corrupted metadata (optional)
4. Create backups before modifying files
"""

from pathlib import Path
import shutil
import nbformat

NOTEBOOKS = [
    "notebooks/01_eda.ipynb",
    "notebooks/01_edawithllmclassification.ipynb",
    "notebooks/02_preprocessing.ipynb",
    "notebooks/03_intent_classification.ipynb",
    "notebooks/04_model_acc_verification.ipynb",
    "notebooks/05_semantic_search.ipynb",
    "notebooks/06_faiss_vector_db.ipynb",
    "notebooks/07_rag_pipeline.ipynb",
    "notebooks/08_generative_chatbot.ipynb",
    "notebooks/09_response_validation.ipynb",
    "notebooks/10_guardrails.ipynb",
    "notebooks/11_enterprise_rag_pipeline.ipynb",
]


def clean_notebook(notebook_path: str):
    notebook_file = Path(notebook_path)

    if not notebook_file.exists():
        print(f"[SKIPPED] {notebook_path} does not exist")
        return

    try:
        # Backup
        backup_file = notebook_file.with_suffix(".backup.ipynb")
        shutil.copy2(notebook_file, backup_file)

        # Read notebook
        with open(notebook_file, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # Remove problematic metadata
        nb.metadata.pop("widgets", None)

        # Optional: Clean cell metadata
        for cell in nb.cells:
            if "metadata" in cell:
                cell.metadata.pop("execution", None)

        # Rewrite notebook
        with open(notebook_file, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        print(f"[SUCCESS] Fixed: {notebook_path}")

    except Exception as e:
        print(f"[FAILED] {notebook_path}")
        print(f"         Error: {e}")


def main():
    print("=" * 80)
    print("Notebook Validation & Repair Started")
    print("=" * 80)

    for notebook in NOTEBOOKS:
        clean_notebook(notebook)

    print("\n" + "=" * 80)
    print("Notebook Validation & Repair Completed")
    print("=" * 80)


if __name__ == "__main__":
    main()