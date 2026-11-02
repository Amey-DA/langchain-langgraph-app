from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "Azure Data Factory is a cloud-based data integration service."

embedding = model.encode(text)

print("Embedding generated successfully!")
print("Embedding dimensions:", len(embedding))
print("First 10 values:", embedding[:10])