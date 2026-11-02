from retriever import search_documents
from llm import ask_watsonx


question = "What is Azure Data Factory?"


# Retrieve relevant chunks
results = search_documents(
    question,
    top_k=3
)


# Get retrieved documents
documents = results["documents"][0]


# Combine retrieved chunks into context
context = "\n\n".join(documents)


print("\nRetrieved Context:")
print(context)


# Generate answer using Watsonx
answer = ask_watsonx(
    question,
    context
)


print("\n\nFinal Answer:")
print(answer)