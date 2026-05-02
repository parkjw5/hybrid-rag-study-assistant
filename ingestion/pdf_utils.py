from PyPDF2 import PdfReader


def extract_text_from_pdf(pdf_path: str):
    """
    Returns list of tuples:
    [(page_number, text), ...]
    """

    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        pages.append((page_number, text))

    return pages