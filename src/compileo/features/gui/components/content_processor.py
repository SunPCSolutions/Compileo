"""
Document Content Processor
Handles document content loading, sampling, and page extraction for AI analysis.
"""

import streamlit as st
from typing import List, Optional
from ....core.logging import get_logger

logger = get_logger(__name__)

from ..services.document_api_service import get_project_documents


def smart_sample_content(content: str, user_examples: Optional[List[str]] = None) -> str:
    """Smart sampling of document content for AI analysis. Handles large documents by extracting representative sections."""
    if not content:
        return content

    content_length = len(content)

    # For small documents, return as-is
    if content_length <= 10000:  # ~2-3k tokens
        return content

    # For large documents, extract representative sample
    st.info(f"📊 Large document detected ({content_length:,} chars). Analyzing representative sample...")

    # If user provided specific examples, prioritize content around those patterns
    if user_examples:
        st.info("🎯 Focusing analysis on user-provided examples and similar patterns...")

        # Look for patterns similar to user examples in the document
        example_patterns = []
        for example in user_examples:
            # Extract key patterns from examples (e.g., all caps words, author patterns)
            lines = example.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.isupper() or '@' in line or ',' in line):  # Likely headers or author lists
                    example_patterns.append(line)

        # Find similar patterns in the document
        relevant_sections = []
        lines = content.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            for pattern in example_patterns:
                if pattern.lower() in line.lower() or line.isupper():
                    # Include context around this line
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    section = '\n'.join(lines[start:end])
                    if section not in relevant_sections:
                        relevant_sections.append(section)

        # If we found relevant sections, use those instead of generic sampling
        if relevant_sections:
            sampled_content = f"""=== FOCUSED ANALYSIS (Based on User Examples) ===

=== RELEVANT SECTIONS MATCHING USER EXAMPLES ===
{chr(10).join(relevant_sections[:5])}

=== DOCUMENT CONTEXT ===
Total length: {content_length:,} characters
Analysis focused on patterns similar to user examples."""

            return sampled_content

    # Fallback to original sampling strategy if no examples or no matches found
    lines = content.split('\n')
    total_lines = len(lines)

    # Extract headers and titles (lines that look like headers)
    headers = []
    for line in lines[:100]:  # Check first 100 lines for headers
        line = line.strip()
        if line and (line.startswith('#') or len(line) < 100 and line[0].isupper()):
            headers.append(line)

    # Sample different sections
    sample_size = min(5000, content_length // 3)  # Up to ~1k tokens per section

    beginning = content[:sample_size]
    middle_start = max(0, content_length // 2 - sample_size // 2)
    middle = content[middle_start:middle_start + sample_size]
    end_start = max(0, content_length - sample_size)
    end = content[end_start:]

    # Combine with clear section markers
    sampled_content = f"""=== DOCUMENT SAMPLE (Large Document - {content_length:,} chars) ===

=== KEY HEADERS/TITLES ===
{chr(10).join(headers[:10])}

=== BEGINNING SECTION ===
{beginning}

=== MIDDLE SECTION ===
{middle}

=== END SECTION ===
{end}

=== SAMPLING NOTE ===
This is a representative sample from a large document. The AI analysis is based on structural patterns found in these sections."""

    return sampled_content


def extract_document_pages(content: str, chunk_size: int = 3000) -> List[str]:
    """
    Extracts pages from document content using a fixed character chunk size
    to align with the preview pagination.
    """
    if not content:
        return []

    logger.debug(f"Extracting pages with fixed chunk size: {chunk_size}")

    pages = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
    
    # Filter out any empty pages that might result from the final chunk
    filtered_pages = [page for page in pages if page.strip()]
    
    logger.debug(f"Final page count: {len(filtered_pages)}")
    return filtered_pages

