"""
Image processing utilities for VLM parsers.
Provides optimized PDF-to-image conversion for both Ollama and HuggingFace parsers.
"""

from PIL import Image
from pdf2image import convert_from_path
from typing import List, Optional
import os
from ....core.logging import get_logger

logger = get_logger(__name__)


def convert_pdf_to_optimized_images(
    pdf_path: str,
    dpi: int = 300,  # Higher DPI for better OCR quality across all VLM models
    max_dimension: Optional[int] = 1024,  # Re-enable resizing to 1024px max
    quality: int = 95
) -> List[Image.Image]:
    """
    Convert PDF to optimized images for VLM processing.

    This function provides optimized PDF-to-image conversion that balances
    processing speed with OCR quality for both Ollama and HuggingFace VLMs.

    Args:
        pdf_path: Path to the PDF file
        dpi: DPI for PDF conversion (higher = better quality, default: 300)
        max_dimension: Maximum dimension for resizing in pixels (default: 1024)
        quality: JPEG quality if saving intermediate files (default: 95)

    Returns:
        List of optimized PIL Images ready for VLM processing

    Optimization strategy:
    - Higher DPI improves OCR accuracy for text recognition
    - Smart resizing maintains aspect ratio while limiting max dimension
    - High-quality LANCZOS resampling preserves text clarity
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    max_dim_str = f"{max_dimension}px" if max_dimension is not None else "None"
    logger.info(f"🔄 Converting PDF to optimized images (DPI={dpi}, max_dim={max_dim_str})...")

    # Convert PDF with optimized DPI
    images = convert_from_path(pdf_path, dpi=dpi)

    if not images:
        raise ValueError(f"No images extracted from PDF: {pdf_path}")

    # Resize images for optimal VLM performance
    optimized_images = []
    total_original_size = 0
    total_optimized_size = 0

    for i, image in enumerate(images):
        width, height = image.size
        total_original_size += width * height

        # Calculate new size maintaining aspect ratio
        if max_dimension is not None:
            if width > height:
                if width > max_dimension:
                    new_width = max_dimension
                    new_height = int(max_dimension * height / width)
                    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                if height > max_dimension:
                    new_height = max_dimension
                    new_width = int(max_dimension * width / height)
                    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        optimized_images.append(image)
        total_optimized_size += image.size[0] * image.size[1]

        logger.debug(f"  📄 Page {i+1}: {width}x{height} → {image.size[0]}x{image.size[1]}")

    # Calculate compression ratio
    if total_original_size > 0:
        compression_ratio = total_optimized_size / total_original_size
        size_reduction = (1 - compression_ratio) * 100
        logger.debug(f"  📊 Size reduction: {size_reduction:.1f}% ({compression_ratio:.2f}x smaller)")
        logger.debug(f"  📊 Total pixels: {total_original_size:,} → {total_optimized_size:,}")

    logger.info(f"✅ PDF converted to {len(optimized_images)} optimized images")
    return optimized_images


def optimize_image_for_vlm(
    image: Image.Image,
    max_dimension: int = 1024,
    maintain_aspect_ratio: bool = True
) -> Image.Image:
    """
    Optimize a single image for VLM processing.

    Args:
        image: PIL Image to optimize
        max_dimension: Maximum dimension in pixels
        maintain_aspect_ratio: Whether to maintain aspect ratio

    Returns:
        Optimized PIL Image
    """
    width, height = image.size

    if maintain_aspect_ratio:
        if width > height:
            if width > max_dimension:
                new_width = max_dimension
                new_height = int(max_dimension * height / width)
                return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            if height > max_dimension:
                new_height = max_dimension
                new_width = int(max_dimension * width / height)
                return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else:
        # Square crop/resize
        size = min(max_dimension, width, height)
        return image.resize((size, size), Image.Resampling.LANCZOS)

    return image  # Return unchanged if already optimized