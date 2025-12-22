import logging
import sys

# Patch for pi_heif compatibility with newer pillow-heif versions
try:
    import pillow_heif
    sys.modules["pi_heif"] = pillow_heif
except ImportError:
    pass

from unstructured.partition.auto import partition
from markdownify import markdownify as md

# Suppress the specific pdfminer warning
logging.getLogger("pdfminer").setLevel(logging.ERROR)

def parse_document_with_unstructured(file_path: str) -> str:
    """
    Parses a document using the unstructured library and returns the content as a markdown string.
    """
    strategy = "auto"
    kwargs = {}
    if file_path.lower().endswith(".pdf"):
        strategy = "hi_res"
    elif file_path.lower().endswith(".xml"):
        # For data-heavy XML, keep tags to ensure we get something
        kwargs["xml_keep_tags"] = True

    elements = partition(
        filename=file_path,
        strategy=strategy,
        infer_table_structure=True,
        **kwargs
    )
    
    output_elements = []
    for el in elements:
        if hasattr(el.metadata, 'text_as_html') and el.metadata.text_as_html:
            output_elements.append(md(el.metadata.text_as_html))
        else:
            output_elements.append(str(el))
            
    return "\n\n".join(output_elements)