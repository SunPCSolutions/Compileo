"""
Pagination Settings Component
Provides configuration for document pagination during processing.
"""

import streamlit as st
from typing import Tuple


def render_pagination_settings() -> int:
    """
    Render pagination settings component with validation.

    Returns:
        pages_per_split (overlap_pages always defaults to 0)
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

    # Validation and helpful information
    if pages_per_split < 5:
        st.warning("⚠️ Pages per split should be at least 5 for optimal processing.")
    elif pages_per_split > 500:
        st.warning("⚠️ Large page splits may impact processing performance.")

    return pages_per_split