"""Taxonomy browser page for the Compileo GUI."""

import logging
import streamlit as st

# Import modular components
from src.compileo.features.gui.taxonomy_display import (
    render_edit_taxonomy,
    render_delete_taxonomy_confirmation,
    render_taxonomy_browser_management
)
from src.compileo.features.gui.taxonomy_editor import (
    render_unified_taxonomy_builder,
)

logger = logging.getLogger(__name__)


def render_taxonomy():
    """Main render function with state management."""
    logger.debug(
        "DEBUG: taxonomy_page_render_start editing=%s deleting=%s selected_view_mode=%s",
        "editing_taxonomy" in st.session_state,
        "deleting_taxonomy" in st.session_state,
        st.session_state.get("unified_view_mode"),
    )
    # Check for different view states
    if "editing_taxonomy" in st.session_state:
        render_edit_taxonomy()
    elif "deleting_taxonomy" in st.session_state:
        render_delete_taxonomy_confirmation()
    else:
        # Main taxonomy view with unified interface
        st.markdown('<h1 class="section-header">🏷️ Taxonomy Builder</h1>', unsafe_allow_html=True)

        st.markdown("""
        Create, manage, and generate hierarchical taxonomies for content categorization.
        Combine manual editing with AI assistance in a unified interface.
        """)

        # Use conditional navigation instead of ``st.tabs``. Streamlit executes
        # all tab bodies on every rerun, which can eagerly render very large
        # taxonomy trees even when the Browse tab is not selected.
        section = st.radio(
            "Taxonomy section",
            options=["🏗️ Build Taxonomy", "📋 Browse & Manage Taxonomies"],
            horizontal=True,
            key="taxonomy_section_selector",
            label_visibility="collapsed",
        )

        if section == "🏗️ Build Taxonomy":
            render_unified_taxonomy_builder()
        else:
            render_taxonomy_browser_management()
    logger.debug("DEBUG: taxonomy_page_render_done")
