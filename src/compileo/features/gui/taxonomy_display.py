"""
Taxonomy Display Module.
Handles all taxonomy display and viewing functionality.
"""

import streamlit as st
from typing import Dict, Any, List
from src.compileo.features.gui.services.api_client import api_client, APIError
from src.compileo.features.gui.components.taxonomy_components import render_taxonomy_tree


def render_taxonomy_browser():
    """Render the main taxonomy browser interface."""
    from src.compileo.features.gui.utils.taxonomy_utils import get_available_taxonomies

    taxonomies = get_available_taxonomies()
    if not taxonomies:
        st.info("No taxonomies available. Create your first taxonomy using the other tabs.")
        return

    taxonomy_options = {f"{t['name']} ({t['categories_count']} categories)": t for t in taxonomies}
    selected_taxonomy_key = st.selectbox(
        "Select Taxonomy",
        options=list(taxonomy_options.keys()),
        help="Choose a taxonomy to explore"
    )

    if selected_taxonomy_key:
        selected_taxonomy = taxonomy_options[selected_taxonomy_key]
        render_taxonomy_viewer(selected_taxonomy)


def render_taxonomy_viewer(taxonomy: Dict[str, Any]):
    """Render the taxonomy viewer with hierarchical display."""
    st.subheader(f"📋 {taxonomy['name']}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Categories", taxonomy['categories_count'])
    with col2:
        st.metric("Confidence Score", f"{taxonomy['confidence_score']:.2f}")
    with col3:
        st.metric("Project ID", taxonomy['project_id'])
    with col4:
        st.metric("Created", taxonomy['created_at'][:10] if taxonomy.get('created_at') else 'Unknown')

    try:
        taxonomy_detail = api_client.get(f"/api/v1/taxonomy/{taxonomy['id']}")
        taxonomy_data = taxonomy_detail.get('taxonomy', {})
        search_term = st.text_input("🔍 Search categories", placeholder="Enter category name...", key=f"search_{taxonomy['id']}")
        render_taxonomy_tree(taxonomy_data, taxonomy_id=taxonomy['id'], search_term=search_term.lower() if search_term else None)
    except APIError as e:
        st.error(f"Failed to load taxonomy details: {e}")


def render_taxonomy_details():
    """Render detailed view of a selected taxonomy."""
    if "selected_taxonomy" not in st.session_state:
        st.info("Select a taxonomy to view details.")
        return

    taxonomy = st.session_state.selected_taxonomy

    st.subheader(f"🏷️ {taxonomy.get('name', 'Unnamed Taxonomy')}")

    # Taxonomy info
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**ID:** {taxonomy.get('id')}")
        st.write(f"**Project ID:** {taxonomy.get('project_id')}")
        st.write(f"**Created:** {taxonomy.get('created_at', 'Unknown')[:10] if taxonomy.get('created_at') else 'Unknown'}")

    with col2:
        st.metric("Categories", taxonomy.get('categories_count', 0))
        st.metric("Confidence Score", f"{taxonomy.get('confidence_score', 0.0):.2f}")

    # Show taxonomy tree if available
    try:
        response = api_client.get(f"/api/v1/taxonomy/{taxonomy['id']}")
        taxonomy_data = response.get('taxonomy', {})
        if taxonomy_data:
            st.subheader("Taxonomy Structure")
            search_term = st.text_input("🔍 Search categories", placeholder="Enter category name...", key=f"detail_search_{taxonomy['id']}")
            render_taxonomy_tree(taxonomy_data, taxonomy_id=taxonomy['id'], search_term=search_term.lower() if search_term else None)
    except APIError as e:
        st.error(f"Failed to load taxonomy details: {e}")

    # Action buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✏️ Edit Taxonomy", width='stretch'):
            st.session_state.editing_taxonomy = taxonomy
            del st.session_state.selected_taxonomy
            st.rerun()

    with col2:
        if st.button("📤 Export JSON", width='stretch'):
            try:
                response = api_client.get(f"/api/v1/taxonomy/{taxonomy['id']}")
                import json
                json_str = json.dumps(response, indent=2)
                st.download_button(
                    label="📥 Download",
                    data=json_str,
                    file_name=f"taxonomy_{taxonomy['id']}_{__import__('time').time():.0f}.json",
                    mime="application/json",
                    width='stretch'
                )
            except APIError as e:
                st.error(f"Failed to export taxonomy: {e}")

    with col3:
        if st.button("⬅️ Back to List", width='stretch'):
            del st.session_state.selected_taxonomy
            st.rerun()


def render_edit_taxonomy():
    """Render the taxonomy editing form."""
    if "editing_taxonomy" not in st.session_state:
        st.info("Select a taxonomy to edit.")
        return

    taxonomy = st.session_state.editing_taxonomy

    st.subheader(f"✏️ Edit Taxonomy: {taxonomy.get('name')}")

    with st.form("edit_taxonomy_form"):
        name = st.text_input("Taxonomy Name", value=taxonomy.get('name', ''))

        col1, col2 = st.columns(2)

        with col1:
            submitted = st.form_submit_button("💾 Save Changes", width='stretch')

        with col2:
            if st.form_submit_button("❌ Cancel", width='stretch'):
                del st.session_state.editing_taxonomy
                st.rerun()
                return

        if submitted:
            if not name.strip():
                st.error("Taxonomy name is required")
                return

            try:
                with st.spinner("Updating taxonomy..."):
                    response = api_client.put(f"/api/v1/taxonomy/{taxonomy['id']}", data={
                        "name": name.strip()
                    })

                st.success("Taxonomy updated successfully!")
                del st.session_state.editing_taxonomy
                st.cache_data.clear()
                st.rerun()

            except APIError as e:
                st.error(f"Failed to update taxonomy: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")


def render_delete_taxonomy_confirmation():
    """Render taxonomy deletion confirmation."""
    if "deleting_taxonomy" not in st.session_state:
        return

    taxonomy = st.session_state.deleting_taxonomy

    st.subheader("🗑️ Delete Taxonomy")
    st.warning(f"Are you sure you want to delete the taxonomy **{taxonomy.get('name')}**?")
    st.error("This action cannot be undone. The taxonomy file will be permanently removed.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("❌ Yes, Delete", width='stretch', type="primary"):
            try:
                with st.spinner("Deleting taxonomy..."):
                    api_client.delete(f"/api/v1/taxonomy/{taxonomy['id']}")

                st.success("Taxonomy deleted successfully!")
                del st.session_state.deleting_taxonomy
                st.cache_data.clear()
                st.rerun()

            except APIError as e:
                st.error(f"Failed to delete taxonomy: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

    with col2:
        if st.button("⬅️ Cancel", width='stretch'):
            del st.session_state.deleting_taxonomy
            st.rerun()


def render_taxonomy_browser_management():
    """Render unified taxonomy browser and management interface."""
    from src.compileo.features.gui.utils.taxonomy_utils import get_available_taxonomies
    from src.compileo.features.gui.components.taxonomy_components import render_taxonomy_card
    from src.compileo.features.gui.utils.taxonomy_export import export_taxonomies_json, export_taxonomies_csv
    import time

    st.subheader("📋 Browse & Manage Taxonomies")

    # Get all taxonomies
    taxonomies = get_available_taxonomies()

    if not taxonomies:
        st.info("No taxonomies available. Create your first taxonomy using the Build Taxonomy tab.")
        return

    # Search and filter controls
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        search_query = st.text_input("🔍 Search taxonomies", placeholder="Enter taxonomy name...", key="unified_taxonomy_search")

    with col2:
        sort_by = st.selectbox("Sort by", ["Newest", "Oldest", "Name A-Z", "Name Z-A", "Confidence"], key="unified_taxonomy_sort")

    with col3:
        view_mode = st.selectbox("View mode", ["Tree", "List"], key="unified_view_mode")

    # Filter taxonomies based on search
    filtered_taxonomies = taxonomies
    if search_query:
        filtered_taxonomies = [t for t in taxonomies if search_query.lower() in t.get("name", "").lower()]

    # Sort taxonomies
    if sort_by == "Newest":
        filtered_taxonomies.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    elif sort_by == "Oldest":
        filtered_taxonomies.sort(key=lambda x: x.get("created_at", ""))
    elif sort_by == "Name A-Z":
        filtered_taxonomies.sort(key=lambda x: x.get("name", "").lower())
    elif sort_by == "Name Z-A":
        filtered_taxonomies.sort(key=lambda x: x.get("name", "").lower(), reverse=True)
    elif sort_by == "Confidence":
        filtered_taxonomies.sort(key=lambda x: x.get("confidence_score", 0.0), reverse=True)

    # Bulk selection controls
    from src.compileo.features.gui.components.taxonomy_components import render_bulk_selection_controls_taxonomies
    render_bulk_selection_controls_taxonomies(filtered_taxonomies)

    # Display taxonomies based on view mode
    if view_mode == "List":
        # List view with management actions
        for taxonomy in filtered_taxonomies:
            render_taxonomy_card(taxonomy)
    else:
        # Tree view - show taxonomy selection and tree
        if filtered_taxonomies:
            taxonomy_options = {f"{t['name']} ({t['categories_count']} categories)": t for t in filtered_taxonomies}
            selected_taxonomy_key = st.selectbox(
                "Select Taxonomy to Browse",
                options=list(taxonomy_options.keys()),
                help="Choose a taxonomy to explore its hierarchical structure"
            )

            if selected_taxonomy_key:
                selected_taxonomy = taxonomy_options[selected_taxonomy_key]
                render_taxonomy_viewer(selected_taxonomy)

                # Management actions for the selected taxonomy
                st.divider()
                st.subheader("⚙️ Manage Selected Taxonomy")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if st.button("👁️ View Details", key=f"view_details_{selected_taxonomy['id']}", width='stretch'):
                        st.session_state.selected_taxonomy = selected_taxonomy
                        st.rerun()
                with col2:
                    if st.button("✏️ Edit", key=f"edit_selected_{selected_taxonomy['id']}", width='stretch'):
                        st.session_state.editing_taxonomy = selected_taxonomy
                        st.rerun()
                with col3:
                    if st.button("📤 Export JSON", key=f"export_selected_{selected_taxonomy['id']}", width='stretch'):
                        try:
                            response = api_client.get(f"/api/v1/taxonomy/{selected_taxonomy['id']}")
                            json_str = json.dumps(response, indent=2)
                            st.download_button(
                                label="📥 Download",
                                data=json_str,
                                file_name=f"taxonomy_{selected_taxonomy['id']}_{int(time.time())}.json",
                                mime="application/json",
                                width='stretch'
                            )
                        except APIError as e:
                            st.error(f"Failed to export taxonomy: {e}")
                with col4:
                    if st.button("🗑️ Delete", key=f"delete_selected_{selected_taxonomy['id']}", width='stretch', type="secondary"):
                        st.session_state.deleting_taxonomy = selected_taxonomy
                        st.rerun()

    # Export functionality
    st.divider()
    st.subheader("📤 Export Taxonomies")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📄 Export All as JSON", width='stretch'):
            export_taxonomies_json(filtered_taxonomies)

    with col2:
        if st.button("📊 Export Summary as CSV", width='stretch'):
            export_taxonomies_csv(filtered_taxonomies)