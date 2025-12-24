import fitz  # PyMuPDF

def is_text_based(pdf_path: str, text_threshold: int = 500) -> bool:
    """
    Analyzes a PDF to determine if it is text-based or image-based.

    Args:
        pdf_path: The path to the PDF file.
        text_threshold: The minimum number of characters to be considered text-based.

    Returns:
        True if the PDF is text-based, False otherwise.
    """
    document = fitz.open(pdf_path)
    first_page = document.load_page(0)
    text = first_page.get_text()
    return len(text) > text_threshold