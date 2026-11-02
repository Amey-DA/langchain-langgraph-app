from sentence_transformers import SentenceTransformer
from chunker import create_chunks
from document_loader import full_text


model = SentenceTransformer("all-MiniLM-L6-v2")

chunks = create_chunks(full_text)

embeddings = model.encode(chunks)

print("Total chunks:", len(chunks))
print("Total embeddings:", len(embeddings))
print("Embedding dimensions:", len(embeddings[0]))

print("\nFirst chunk:")
print(chunks[0][:300])

print("\nFirst embedding:")
print(embeddings[0][:10])