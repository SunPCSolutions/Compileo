"""
Pagination Settings Component
Provides configuration for document pagination during processing.
"""

import streamlit as st
from typing import Tuple


def render_pagination_settings() -> Tuple[int, str]:
    """
    Render pagination settings component with validation.

    Returns:
        Tuple of (pages_per_split, page_selection)
        - pages_per_split: Number of pages per split chunk
        - page_selection: User-provided page selection expression or empty string
    """
    st.markdown("### 📄 Pagination Settings")

    # Initialize default values
    default_pages_per_split = 50

    # Single column for pages per split (overlap removed)
    pages_per_split = st.number_input(
        "Pages per split",
        min_value=5,
        max_value=500,
        value=default_pages_per_split,
        step=25,
        help="Number of pages to include in each document split (5-500 pages)",
        key="pages_per_split"
    )

    # Page selection input
    page_selection = st.text_input(
        "Page Selection (optional)",
        placeholder="e.g., 1-20, 30-120",
        help=(
            "Leave empty to process all pages. "
            "Use commas for batches and hyphens for ranges. "
            "Examples: '1', '1-20', '1, 1-20, 30-120'. "
            "Only applies to PDF documents."
        ),
        key="page_selection"
    )

    # Validation and helpful information
    if pages_per_split < 5:
        st.warning("⚠️ Pages per split should be at least 5 for optimal processing.")
    elif pages_per_split > 500:
        st.warning("⚠️ Large page splits may impact processing performance.")

    return pages_per_split, page_selection