import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

from pathlib import Path

RAG_DIR = Path(__file__).resolve().parent
VECTOR_DB_PATH = RAG_DIR / "vector_db"

client = chromadb.PersistentClient(
    path=str(VECTOR_DB_PATH)
)

def get_collection():
    return client.get_or_create_collection(name="documents")


def search_documents(query, top_k=3):
    collection = get_collection()

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    return results

if __name__ == "__main__":

    question = "What are the features of Azure Data Factory?"

    results = search_documents(question, top_k=3)

    print("\nQuestion:")
    print(question)

    print("\nRetrieved Results:")

    for i, document in enumerate(
        results["documents"][0],
        start=1
    ):
        print(f"\n--- Result {i} ---")

        print(
            "Source:",
            results["metadatas"][0][i - 1]["source"]
        )

        print(
            "Page:",
            results["metadatas"][0][i - 1]["page_number"]
        )

        print(
            "Distance:",
            results["distances"][0][i - 1]
        )

        print(document[:500])