import pymupdf as pf

pdf_path = "documents/Azure_data_engineering.pdf"

document = pf.open(pdf_path)

full_text = ""

for page_number, page in enumerate(document, start=1):
    text = page.get_text()

    print(f"Page {page_number}: {len(text)} characters")

    full_text += text + "\n"

document.close()

print("\nTotal characters:", len(full_text))
print("\nFirst 1000 characters:\n")
print(full_text[:1000])