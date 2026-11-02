
import os
from paddleocr import PaddleOCR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "images.png")

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# Initialize PaddleOCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False
)

# Extract text from the image
results = ocr.predict(IMAGE_PATH)

print("\n========== OCR EXTRACTED TEXT ==========\n")

for result in results:
    result.print()

    # Display recognized text separately
    data = result.json
    if callable(data):
        data = data()

    if isinstance(data, dict):
        data = data.get("res", data)
        texts = data.get("rec_texts", [])

        for text in texts:
            print(text)

print("\nOCR test completed.")
