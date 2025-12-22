"""
PDF splitting functionality for document processing.
Mock implementation for job processing.
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def pre_split_pdf(
    file_path: str,
    pages_per_split: int = 200,
    overlap_pages: int = 1
) -> Dict[str, Any]:
    """
    Split a PDF into smaller chunks if it exceeds page limit and create a manifest.

    Args:
        file_path: Path to the PDF file
        pages_per_split: Number of pages per split
        overlap_pages: Number of overlapping pages

    Returns:
        Dict containing:
        - split_files: List[str] - List of paths to split files (original if no split needed)
        - manifest_path: str - Path to the manifest file (None if no split needed)
    """
    # Increase recursion limit for complex PDFs
    old_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(5000)
    
    try:
        logger.info(f"DEBUG_SPLIT: Checking PDF for splitting: {file_path}, pages_per_split={pages_per_split}, overlap={overlap_pages}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        # Get page count
        try:
            from pypdf import PdfReader, PdfWriter
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            logger.info(f"PDF has {total_pages} pages")
        except ImportError:
            logger.warning("pypdf not available, cannot check page count. Treating as single file.")
            return {
                "split_files": [file_path],
                "manifest_path": None
            }

        # Check if splitting is needed
        if total_pages <= pages_per_split:
            logger.info(f"PDF has {total_pages} pages, no splitting needed (limit: {pages_per_split})")
            # Create manifest for single file
            manifest_data = {
                "original_file": file_path,
                "total_pages": total_pages,
                "pages_per_split": pages_per_split,
                "overlap_pages": overlap_pages,
                "splitting_occurred": False,
                "splits": [
                    {
                        "chunk_number": 1,
                        "file_path": file_path,
                        "page_range": {
                            "start": 1,
                            "end": total_pages,
                            "actual_pages": total_pages
                        },
                        "overlap": {
                            "with_previous": None,
                            "with_next": None
                        },
                        "relationships": {
                            "previous_chunk": None,
                            "next_chunk": None
                        }
                    }
                ]
            }

            # Save manifest for single file
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            manifest_filename = f"{base_name}_manifest.json"
            manifest_path = os.path.join(os.path.dirname(file_path), manifest_filename)

            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Created manifest for single file: {manifest_path}")
            return {
                "split_files": [file_path],
                "manifest_path": manifest_path
            }

        # Split the PDF
        logger.info(f"Splitting PDF into chunks of {pages_per_split} pages each")

        split_files = []
        splits_info = []
        start_page = 0
        chunk_num = 0

        # Get base name for file naming
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        while start_page < total_pages:
            chunk_num += 1
            end_page = min(start_page + pages_per_split, total_pages)

            # Create output filename
            output_filename = f"{base_name}_chunk_{chunk_num:03d}.pdf"
            output_path = os.path.join(os.path.dirname(file_path), output_filename)

            # Extract pages for this chunk
            writer = PdfWriter()
            for page_num in range(start_page, end_page):
                writer.add_page(reader.pages[page_num])

            # Save chunk
            with open(output_path, 'wb') as output_file:
                writer.write(output_file)

            split_files.append(output_path)

            # Calculate overlap information
            overlap_with_previous = None
            overlap_with_next = None
            previous_chunk = chunk_num - 1 if chunk_num > 1 else None
            next_chunk = chunk_num + 1 if end_page < total_pages else None

            if previous_chunk:
                # Calculate overlap with previous chunk
                prev_overlap_start = max(0, start_page - overlap_pages)
                prev_overlap_end = start_page
                if prev_overlap_end > prev_overlap_start:
                    overlap_with_previous = {
                        "pages": prev_overlap_end - prev_overlap_start,
                        "range": f"{prev_overlap_start + 1}-{prev_overlap_end}"
                    }

            if next_chunk and end_page < total_pages:
                # Calculate overlap with next chunk
                next_overlap_start = max(start_page, end_page - overlap_pages)
                next_overlap_end = min(total_pages, end_page)
                if next_overlap_end > next_overlap_start:
                    overlap_with_next = {
                        "pages": next_overlap_end - next_overlap_start,
                        "range": f"{next_overlap_start + 1}-{next_overlap_end}"
                    }

            # Create split info entry
            split_info = {
                "chunk_number": chunk_num,
                "file_path": output_path,
                "page_range": {
                    "start": start_page + 1,  # 1-based page numbers
                    "end": end_page,
                    "actual_pages": end_page - start_page
                },
                "overlap": {
                    "with_previous": overlap_with_previous,
                    "with_next": overlap_with_next
                },
                "relationships": {
                    "previous_chunk": previous_chunk,
                    "next_chunk": next_chunk
                }
            }
            splits_info.append(split_info)

            logger.info(f"Created chunk {chunk_num}: pages {start_page+1}-{end_page} -> {output_filename}")

            # Move to next chunk with overlap, but don't overlap if we're at the end
            if end_page >= total_pages:
                break
            start_page = end_page - overlap_pages
            if start_page >= total_pages:
                break

        # Create manifest file
        manifest_data = {
            "original_file": file_path,
            "total_pages": total_pages,
            "pages_per_split": pages_per_split,
            "overlap_pages": overlap_pages,
            "splits": splits_info
        }

        # Save manifest alongside the original file
        manifest_filename = f"{base_name}_manifest.json"
        manifest_path = os.path.join(os.path.dirname(file_path), manifest_filename)

        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)

        logger.info(f"PDF splitting completed: {len(split_files)} chunks created from {total_pages} pages")
        logger.info(f"Manifest created: {manifest_path}")

        return {
            "split_files": split_files,
            "manifest_path": manifest_path
        }

    except Exception as e:
        logger.error(f"Error splitting PDF: {e}")
        # Return original file if splitting fails
        return {
            "split_files": [file_path],
            "manifest_path": None
        }
    finally:
        # Restore original recursion limit
        sys.setrecursionlimit(old_recursion_limit)


def split_pdf_files(
    file_path: str,
    pages_per_split: int = 200,
    overlap_pages: int = 1
) -> List[str]:
    """
    Backward-compatible function that splits a PDF and returns only the file list.

    This function maintains the old API for existing code that expects a list of strings.

    Args:
        file_path: Path to the PDF file
        pages_per_split: Number of pages per split
        overlap_pages: Number of overlapping pages

    Returns:
        List[str]: List of paths to split files (original if no split needed)
    """
    result = pre_split_pdf(file_path, pages_per_split, overlap_pages)
    return result["split_files"]