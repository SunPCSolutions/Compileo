"""
Page selection parsing utility for document ingestion.

Supports expressions like:
- Single pages: "1"
- Page ranges: "1-20"
- Batches of ranges: "1, 1-20, 30-120"
"""

import re
from typing import List, Optional


def parse_page_selection(selection_str: Optional[str], max_pages: int) -> List[int]:
    """
    Parse a page selection string into a sorted, deduplicated list of 1-based page numbers.

    Args:
        selection_str: User-provided page selection expression.
            Examples: "1", "1-20", "1, 1-20, 30-120"
            Empty string or None means "all pages" (returns empty list).
        max_pages: Total number of pages in the document. Page numbers are clamped
            to this maximum and validated against it.

    Returns:
        A sorted list of unique 1-based page numbers. Returns an empty list if
        selection_str is None/empty, meaning "process all pages".

    Raises:
        ValueError: If the selection string contains invalid syntax or non-numeric values.
    """
    if not selection_str or not selection_str.strip():
        return []

    if max_pages < 1:
        return []

    pages: set[int] = set()
    parts = [p.strip() for p in selection_str.split(",") if p.strip()]

    # Regex for validating a single page number
    page_pattern = re.compile(r"^\d+$")
    # Regex for validating a range like "1-20" (allows optional spaces around dash)
    range_pattern = re.compile(r"^(\d+)\s*-\s*(\d+)$")

    for part in parts:
        if not part:
            continue

        # Detect clearly negative numbers or leading minus before falling
        # through to the generic syntax error so we can give a helpful message.
        if part.startswith("-") or part.startswith("- "):
            raise ValueError(
                f"Invalid page number '{part.strip()}': page numbers must be >= 1."
            )

        # Check if it's a range
        range_match = range_pattern.match(part)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))

            if start < 1 or end < 1:
                raise ValueError(
                    f"Invalid page range '{part.strip()}': page numbers must be >= 1. "
                    f"Use 1-based page numbering."
                )

            if start > end:
                raise ValueError(
                    f"Invalid page range '{part.strip()}': start page ({start}) cannot be "
                    f"greater than end page ({end})."
                )

            # Clamp to valid document pages (1-based)
            clamped_start = max(1, start)
            clamped_end = min(end, max_pages)

            if clamped_start <= clamped_end:
                pages.update(range(clamped_start, clamped_end + 1))
            continue

        # Check if it's a single page number
        if page_pattern.match(part):
            page = int(part)
            if page < 1:
                raise ValueError(
                    f"Invalid page number '{part}': page numbers must be >= 1."
                )
            if page <= max_pages:
                pages.add(page)
            continue

        # If we get here, the format is invalid
        raise ValueError(
            f"Invalid page selection syntax in '{part}'. "
            f"Expected a single page (e.g., '1') or a range (e.g., '1-20'). "
            f"Multiple selections can be separated by commas (e.g., '1, 1-20, 30-120')."
        )

    return sorted(pages)


def format_page_selection(pages: List[int]) -> str:
    """
    Format a list of page numbers into a compact human-readable string.

    Args:
        pages: A sorted list of 1-based page numbers.

    Returns:
        A formatted string like "1-5, 7, 9-12".
    """
    if not pages:
        return ""

    ranges = []
    start = pages[0]
    prev = pages[0]

    for page in pages[1:]:
        if page == prev + 1:
            prev = page
        else:
            if start == prev:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{prev}")
            start = page
            prev = page

    # Add the final range
    if start == prev:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{prev}")

    return ", ".join(ranges)
