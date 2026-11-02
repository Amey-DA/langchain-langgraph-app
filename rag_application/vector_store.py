import chromadb
from sentence_transformers import SentenceTransformer

from chunker import create_chunks
from document_loader import full_text


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Create Chroma client
client = chromadb.PersistentClient(path="vector_db")


# Create or get collection
collection = client.get_or_create_collection(
    name="documents"
)


# Create chunks
chunks = create_chunks(full_text)


# Generate embeddings
embeddings = model.encode(chunks).tolist()


# Create IDs
ids = [f"chunk_{i}" for i in range(len(chunks))]


# Create metadata
metadatas = [
    {
        "source": "data_engineering.pdf",
        "chunk_number": i + 1
    }
    for i in range(len(chunks))
]


# Store everything in Chroma
collection.upsert(
    ids=ids,
    documents=chunks,
    embeddings=embeddings,
    metadatas=metadatas
)


print("Documents stored successfully!")
print("Total chunks in Chroma:", collection.count())