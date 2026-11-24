"""
Image processing utilities for document ingestion.
Provides optimized PDF-to-image conversion for both Ollama and HuggingFace parsers.
"""

from PIL import Image
import fitz  # PyMuPDF
from typing import List, Optional
import os
import io
from ....core.logging import get_logger

logger = get_logger(__name__)

def convert_pdf_to_optimized_images(
    pdf_path: str,
    dpi: int = 200,
    max_dimension: Optional[int] = 1024,
    quality: int = 95
) -> List[Image.Image]:
    """
    Convert PDF to optimized images for VLM processing using PyMuPDF.

    This function provides optimized PDF-to-image conversion that balances
    processing speed with OCR quality for both Ollama and HuggingFace VLMs.
    Uses PyMuPDF (fitz) for pure Python PDF rendering without system dependencies.

    Args:
        pdf_path: Path to the PDF file
        dpi: Dots per inch for rendering (higher = better quality, lower = faster)
        max_dimension: Maximum width/height in pixels (None to disable resizing)
        quality: JPEG compression quality (1-100)

    Returns:
        List of PIL Image objects
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    max_dim_str = f"{max_dimension}px" if max_dimension is not None else "None"
    logger.info(f"🔄 Converting PDF to optimized images (DPI={dpi}, max_dim={max_dim_str})...")

    # Convert PDF with PyMuPDF (pure Python, no system dependencies)
    try:
        doc = fitz.open(pdf_path)
        images = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # Calculate zoom factor for desired DPI
            # PyMuPDF uses 72 DPI as base, so zoom = dpi / 72
            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)

            # Render page to pixmap
            pix = page.get_pixmap(matrix=matrix)

            # Convert to PIL Image
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            images.append(image)

        doc.close()

    except Exception as e:
        raise ValueError(f"Failed to convert PDF to images with PyMuPDF: {e}")

    if not images:
        raise ValueError(f"No images extracted from PDF: {pdf_path}")

    logger.info(f"✅ Successfully converted {len(images)} pages to optimized images")
    return images
