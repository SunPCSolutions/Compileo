"""Taxonomy browser page for the Compileo GUI."""

import streamlit as st

# Import modular components
from src.compileo.features.gui.taxonomy_display import (
    render_taxonomy_browser,
    render_taxonomy_details,
    render_edit_taxonomy,
    render_delete_taxonomy_confirmation,
    render_taxonomy_browser_management
)
from src.compileo.features.gui.taxonomy_editor import (
    render_manual_taxonomy_builder,
    render_unified_taxonomy_builder,
    render_ai_taxonomy_generator
)


def render_taxonomy():
    """Main render function with state management."""
    # Check for different view states
    if "selected_taxonomy" in st.session_state:
        render_taxonomy_details()
    elif "editing_taxonomy" in st.session_state:
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

        # Create tabs for different views
        tab1, tab2 = st.tabs([
            "🏗️ Build Taxonomy",
            "📋 Browse & Manage Taxonomies"
        ])

        with tab1:
            render_unified_taxonomy_builder()

        with tab2:
            render_taxonomy_browser_management()
