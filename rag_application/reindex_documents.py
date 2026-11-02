
import os
from rag_pipeline import process_document

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")

pdf_files = [
    "Azure_data_engineering.pdf",
    "mastering-microsoft-fabric-saasification-of-analytics.pdf"
]

for filename in pdf_files:
    pdf_path = os.path.join(DOCUMENTS_DIR, filename)

    if not os.path.exists(pdf_path):
        print(f"File not found: {filename}")
        continue

    print(f"\nReindexing: {filename}")

    try:
        total_chunks = process_document(pdf_path, filename)
        print(f"Successfully reindexed {filename}: {total_chunks} pages")
    except Exception as e:
        print(f"Failed to reindex {filename}: {e}")
        raise

print("\nReindexing completed.")
