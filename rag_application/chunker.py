def create_chunks(text, chunk_size=1000, overlap=200):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start = end - overlap

    return chunks


# Test
if __name__ == "__main__":

    from document_loader import full_text

    chunks = create_chunks(full_text)

    print("Total chunks:", len(chunks))

    for i, chunk in enumerate(chunks[:3], start=1):
        print(f"\n--- Chunk {i} ---")
        print(chunk)