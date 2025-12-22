import fitz  # PyMuPDF
from typing import Optional

def parse_pdf_with_pypdf(pdf_path: str, last_page_only: bool = False, start_page: Optional[int] = None, end_page: Optional[int] = None) -> str:
    """
    Parses a PDF file using PyMuPDF (fitz) and extracts its text content.
    """
    document = fitz.open(pdf_path)
    text_content = ""
    if last_page_only:
        page = document.load_page(document.page_count - 1)
        text_content += page.get_text()
    elif start_page is not None:
        # Parse from start_page to end_page (or to end of document if end_page is None)
        end_page_num = end_page if end_page is not None else document.page_count
        for page_num in range(start_page - 1, end_page_num):
            page = document.load_page(page_num)
            text_content += page.get_text()
    else:
        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            text_content += page.get_text()
    return text_content