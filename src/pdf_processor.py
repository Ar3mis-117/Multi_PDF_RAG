import pymupdf as fitz


def extract_text_from_pdf(pdf_file):
    """
    Extract text from a PDF while preserving page information.

    Returns:
        list[dict]: Each item contains the page text and metadata.
    """

    document = fitz.open(stream=pdf_file.read(), filetype="pdf")

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text").strip()

        if text:
            pages.append(
                {
                    "text": text,
                    "page": page_number,
                }
            )

    document.close()

    return pages