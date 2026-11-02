import chromadb

client = chromadb.PersistentClient(
    path="vector_db"
)

try:
    client.delete_collection(
        name="documents"
    )

    print("Old ChromaDB collection deleted.")

except Exception:
    print("Collection did not exist.")

collection = client.get_or_create_collection(
    name="documents"
)

print("New ChromaDB collection created.")
print("Total chunks:", collection.count())