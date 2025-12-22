from .pdf_analyzer import is_text_based
from .pypdf_parser import parse_pdf_with_pypdf
from .unstructured_parser import parse_document_with_unstructured
from ....core.logging import get_logger

logger = get_logger(__name__)

def parse_pdf_novlm(file_path: str) -> str:
    """
    Intelligently parses a document using either pypdf or unstructured, based on the document's content.
    """
    if file_path.lower().endswith(".pdf") and is_text_based(file_path):
        logger.info("--- Using pypdf parser for text-based PDF ---")
        return parse_pdf_with_pypdf(file_path)
    else:
        logger.info("--- Using unstructured parser for non-PDF or complex PDF ---")
        return parse_document_with_unstructured(file_path)