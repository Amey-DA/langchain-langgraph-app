import os
import pymupdf
import chromadb
import pymupdf
from llm import extract_page_content_with_vision
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Embedding Model
# --------------------------------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# ChromaDB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="vector_db"
)

collection = client.get_or_create_collection(
    name="documents"
)


# --------------------------------------------------
# Page-Based Chunking
# --------------------------------------------------


def create_page_chunks(pdf_path):

    document = pymupdf.open(pdf_path)
    pages = []

    try:
        for page_index, page in enumerate(document):

            # Step 1: Extract selectable PDF text
            text = page.get_text().strip()

            # Step 2: Check for embedded images
            images = page.get_images(full=True)

            if images:
                print(
                    f"Page {page_index + 1}: "
                    "Embedded images detected. "
                    "Sending page to vision LLM..."
                )

                # Step 3: Render the page as an image
                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2),
                    alpha=False
                )

                image_bytes = pixmap.tobytes("png")

                # Step 4: Extract image and diagram information
                image_text = extract_page_content_with_vision(
                    image_bytes,
                    page_index + 1
                )

                # Step 5: Combine both text outputs
                if image_text:
                    text = (
                        f"{text}\n\n"
                        f"[Visual Content]\n"
                        f"{image_text}"
                    ).strip()

            pages.append(text)

            print(
                f"Page {page_index + 1}: "
                f"{len(text)} characters extracted"
            )

    finally:
        document.close()

    # Preserve the existing previous/current/next page chunking
    chunks = []
    total_pages = len(pages)

    for page_index in range(total_pages):
        current_page = pages[page_index]

        previous_page = (
            pages[page_index - 1]
            if page_index > 0
            else ""
        )

        next_page = (
            pages[page_index + 1]
            if page_index < total_pages - 1
            else ""
        )

        chunk_parts = []

        if previous_page:
            chunk_parts.append(
                f"[Previous Page - Page {page_index}]\n{previous_page}"
            )

        chunk_parts.append(
            f"[Current Page - Page {page_index + 1}]\n{current_page}"
        )

        if next_page:
            chunk_parts.append(
                f"[Next Page - Page {page_index + 2}]\n{next_page}"
            )

        chunks.append({
            "text": "\n\n".join(chunk_parts),
            "page_number": page_index + 1
        })

    return chunks


# --------------------------------------------------
# Process Document
# --------------------------------------------------

def process_document(pdf_path, filename):

    print("\n====================================")
    print("Processing:", filename)
    print("====================================")

    chunks = create_page_chunks(pdf_path)

    print("Total pages/chunks:", len(chunks))

    # Generate embeddings
    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True
    ).tolist()

    # Create unique IDs
    ids = []

    metadatas = []

    for index, chunk in enumerate(chunks):

        chunk_id = (
            f"{filename}_page_{chunk['page_number']}"
        )

        ids.append(chunk_id)

        metadatas.append({
            "source": filename,
            "page_number": chunk["page_number"]
        })
    
    # Remove old chunks for this PDF before reindexing
    collection.delete(
        where={"source": filename}
    )

    print(f"Removed old chunks for: {filename}")

    # Store in ChromaDB

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print("\nDocument indexed successfully.")

    print(
        "Total chunks in ChromaDB:",
        collection.count()
    )

    return len(chunks)