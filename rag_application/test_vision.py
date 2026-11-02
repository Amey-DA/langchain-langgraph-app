
import os
import pymupdf

from llm import extract_page_content_with_vision


PDF_PATH = os.path.join(
    "documents",
    "Medallion_Architecture_OCR_Test.pdf"
)

PAGE_NUMBER = 2  # Test the second page


document = pymupdf.open(PDF_PATH)

try:
    page = document[PAGE_NUMBER - 1]

    # Render the PDF page as an image
    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(2, 2),
        alpha=False
    )

    # Convert the rendered image into PNG bytes
    image_bytes = pixmap.tobytes("png")

    print(f"Testing vision extraction on page {PAGE_NUMBER}...")

    extracted_text = extract_page_content_with_vision(
        image_bytes,
        PAGE_NUMBER
    )

    print("\n========== EXTRACTED CONTENT ==========\n")
    print(extracted_text)

finally:
    document.close()
