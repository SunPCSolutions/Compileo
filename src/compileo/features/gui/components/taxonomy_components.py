"""Reusable UI components for taxonomy operations."""

import streamlit as st
import json
from typing import Dict, Any, List, Optional
import time

from src.compileo.features.gui.services.api_client import api_client, APIError


def render_taxonomy_tree(node: Dict[str, Any], depth: int = 0, taxonomy_id: int = 0, search_term: Optional[str] = None, path: str = ""):
    """Recursively render taxonomy tree using collapsible expanders."""
    if not node:
        return
    node_name = node.get('name', 'Unknown')
    node_desc = node.get('description', '')
    confidence = node.get('confidence_threshold', 0.0)
    children = node.get('children', [])

    # Check search filter
    if search_term and search_term not in node_name.lower() and search_term not in node_desc.lower():
        if not any(search_term in child.get('name', '').lower() or search_term in child.get('description', '').lower() for child in children):
            return

    # Determine confidence color
    color = "🟢" if confidence >= 0.8 else "🟡" if confidence >= 0.6 else "🔴"

    # Create expander for this node
    with st.expander(f"{color} {node_name}", expanded=(depth == 0)):
        # Display confidence score
        st.metric("Confidence Score", f"{confidence:.2f}")

        # Display description if available
        if node_desc:
            st.write(f"**Description:** {node_desc}")

        # Render children recursively
        if children:
            st.markdown("**Subcategories:**")
            for i, child in enumerate(children):
                child_path = f"{path}_{i}" if path else str(i)
                render_taxonomy_tree(child, depth + 1, taxonomy_id, search_term, child_path)


def render_collapsible_category_builder(node: Dict[str, Any], level: int, max_depth: int, path: str, taxonomy: Dict[str, Any], default_expanded: Optional[bool] = None, key_prefix: str = ""):
    """Render a collapsible category builder with per-category depth controls."""
    node_name = node.get("name", "")
    node_desc = node.get("description", "")
    confidence = node.get("confidence_threshold", 0.8)
    children = node.get("children", [])
    depth_limit = node.get("depth_limit", max_depth - level - 1)
    
    # Use key_prefix for unique widget keys
    key_path = f"{key_prefix}_{path}" if key_prefix else path

    # Color coding based on confidence
    color = "🟢" if confidence >= 0.8 else "🟡" if confidence >= 0.6 else "🔴"

    # Determine expansion state: use explicit override or default logic (level 0 expanded)
    is_expanded = default_expanded if default_expanded is not None else (level == 0)

    # Collapsible expander for this category
    with st.expander(f"{color} {node_name or f'Category {level + 1}'} (Level {level + 1})", expanded=is_expanded):
        # Category inputs
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        with col1:
            node["name"] = st.text_input(
                f"Category Name (Level {level + 1})",
                value=node_name,
                key=f"node_name_{key_path}",
                placeholder=f"Enter category name for level {level + 1}"
            )
        with col2:
            node["description"] = st.text_input(
                f"Description (Level {level + 1})",
                value=node_desc,
                key=f"node_desc_{key_path}",
                placeholder="Optional description"
            )
        with col3:
            node["confidence_threshold"] = st.slider(
                f"Confidence (Level {level + 1})",
                min_value=0.0,
                max_value=1.0,
                value=confidence,
                step=0.1,
                key=f"node_conf_{key_path}"
            )
        with col4:
            # Per-category depth control
            max_possible_depth = max_depth - level - 1
            if max_possible_depth > 0:
                node["depth_limit"] = st.slider(
                    f"Depth Limit",
                    min_value=0,
                    max_value=max_possible_depth,
                    value=min(depth_limit, max_possible_depth),
                    key=f"depth_limit_{key_path}",
                    help=f"Maximum additional levels for this category (0-{max_possible_depth})"
                )
            else:
                node["depth_limit"] = 0

        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if level < max_depth - 1 and (node.get("depth_limit", 0) > 0 or level < max_depth - 1):
                if st.button(f"➕ Add Subcategory", key=f"add_child_{key_path}", width='stretch'):
                    if "children" not in node:
                        node["children"] = []
                    node["children"].append({
                        "name": "",
                        "description": "",
                        "confidence_threshold": 0.8,
                        "children": [],
                        "depth_limit": node.get("depth_limit", max_depth - level - 1) - 1
                    })
                    st.rerun()

        with col2:
            # AI extend button for individual categories
            if node.get("name", "").strip() and taxonomy.get("creation_mode") in ["manual", "hybrid"]:
                if st.button(f"🚀 AI Extend", key=f"extend_{key_path}", help=f"Use AI to add subcategories to '{node['name']}'", width='stretch'):
                    from src.compileo.features.gui.utils.taxonomy_utils import extend_category_with_ai
                    extend_category_with_ai(path, taxonomy, node.get("depth_limit", 2), "gemini", "general")
                    # Auto-save after AI extension
                    from src.compileo.features.gui.utils.taxonomy_utils import save_unified_taxonomy
                    save_unified_taxonomy(taxonomy)

        with col3:
            if st.button("🗑️ Remove", key=f"remove_{key_path}", help=f"Remove this category", width='stretch'):
                from src.compileo.features.gui.utils.taxonomy_utils import remove_category_from_unified_tree, save_unified_taxonomy
                # 1. Modify the local state immediately
                remove_category_from_unified_tree(path)
                # 2. Sync to backend silently
                save_unified_taxonomy(taxonomy, silent=True)
                # 3. Mark for wizard-wide refresh
                from src.compileo.features.gui.state.wizard_state import wizard_state
                wizard_state.needs_refresh = True
                # 4. Trigger a full script rerun to ensure all widgets are rebuilt correctly
                st.rerun()

        # Render children recursively
        if children:
            st.markdown("**Subcategories:**")
            for i, child in enumerate(children):
                child_path = f"{path}_{i}"
                remaining_depth = min(node.get("depth_limit", max_depth - level - 1), max_depth - level - 1)
                if remaining_depth > 0:
                    render_collapsible_category_builder(child, level + 1, max_depth, child_path, taxonomy, default_expanded=default_expanded, key_prefix=key_prefix)

        st.divider()


def render_taxonomy_category_builder(node: Dict[str, Any], level: int, max_depth: int, path: str, taxonomy: Optional[Dict[str, Any]] = None, additional_depth: int = 2, generator_type: str = "gemini", domain: str = "general"):
    """Recursively render the taxonomy category builder."""
    indent = "  " * level

    with st.container():
        # Level header
        level_indicator = "🏠" if level == 0 else "📁"
        st.markdown(f"**{level_indicator} Level {level + 1}**" + (f" - {indent}" if level > 0 else ""))

        # Node inputs
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        with col1:
            node["name"] = st.text_input(
                f"Category Name (Level {level + 1})",
                value=node.get("name", ""),
                key=f"node_name_{path}",
                placeholder=f"Enter category name for level {level + 1}"
            )
        with col2:
            node["description"] = st.text_input(
                f"Description (Level {level + 1})",
                value=node.get("description", ""),
                key=f"node_desc_{path}",
                placeholder="Optional description"
            )
        with col3:
            node["confidence_threshold"] = st.slider(
                f"Confidence (Level {level + 1})",
                min_value=0.0,
                max_value=1.0,
                value=node.get("confidence_threshold", 0.8),
                step=0.1,
                key=f"node_conf_{path}"
            )
        with col4:
            if st.button("🗑️ Remove", key=f"remove_{path}", help=f"Remove this category"):
                from src.compileo.features.gui.utils.taxonomy_utils import remove_category_from_tree
                remove_category_from_tree(path)
                st.rerun()
                return

        # Add child button and extend button
        col1, col2 = st.columns(2)
        with col1:
            if level < max_depth - 1:
                if st.button(f"➕ Add Subcategory (Level {level + 2})", key=f"add_child_{path}", width='stretch'):
                    if "children" not in node:
                        node["children"] = []
                    node["children"].append({
                        "name": "",
                        "description": "",
                        "confidence_threshold": 0.8,
                        "children": []
                    })
                    st.rerun()

        with col2:
            # Only show extend button if category has a name and taxonomy is available
            if node.get("name", "").strip() and taxonomy is not None:
                if st.button(f"🚀 Extend Category", key=f"extend_{path}", help=f"Use AI to add subcategories to '{node['name']}'", width='stretch'):
                    from src.compileo.features.gui.utils.taxonomy_utils import extend_category_with_ai
                    extend_category_with_ai(path, taxonomy, additional_depth, generator_type, domain)

        # Render children
        if "children" in node and node["children"]:
            st.markdown("**Subcategories:**")
            for i, child in enumerate(node["children"]):
                child_path = f"{path}_{i}"
                render_taxonomy_category_builder(child, level + 1, max_depth, child_path, taxonomy, additional_depth, generator_type, domain)

        st.divider()  # Visual separator between categories


def render_taxonomy_card(taxonomy: Dict[str, Any]):
    """Render a single taxonomy card with actions."""
    with st.container():
        # Checkbox column
        col0, col1, col2, col3, col4 = st.columns([0.5, 2, 1, 1, 1])

        with col0:
            # Checkbox for selection
            taxonomy_id = taxonomy['id']
            is_selected = taxonomy_id in st.session_state.get('selected_taxonomies', set())
            if st.checkbox(
                f"Select {taxonomy.get('name', 'Unnamed Taxonomy')}",
                value=is_selected,
                key=f"select_tax_{taxonomy_id}",
                label_visibility="collapsed"
            ):
                if taxonomy_id not in st.session_state.selected_taxonomies:
                    st.session_state.selected_taxonomies.add(taxonomy_id)
                    st.rerun()
            else:
                if taxonomy_id in st.session_state.selected_taxonomies:
                    st.session_state.selected_taxonomies.remove(taxonomy_id)
                    st.rerun()

        with col1:
            st.subheader(f"🏷️ {taxonomy.get('name', 'Unnamed Taxonomy')}")
            st.write(f"**ID:** {taxonomy.get('id')}")
            st.write(f"**Project:** {taxonomy.get('project_id')}")
            st.write(f"**Categories:** {taxonomy.get('categories_count', 0)}")
            st.write(f"**Confidence:** {taxonomy.get('confidence_score', 0.0):.2f}")
            st.write(f"**Created:** {taxonomy.get('created_at', 'Unknown')[:10] if taxonomy.get('created_at') else 'Unknown'}")

        with col2:
            if st.button("👁️ View", key=f"view_tax_{taxonomy['id']}", width='stretch'):
                st.session_state.selected_taxonomy = taxonomy
                st.rerun()

        with col3:
            if st.button("✏️ Edit", key=f"edit_tax_{taxonomy['id']}", width='stretch'):
                st.session_state.editing_taxonomy = taxonomy
                st.rerun()

        with col4:
            if st.button("🗑️ Delete", key=f"delete_tax_{taxonomy['id']}", width='stretch'):
                st.session_state.deleting_taxonomy = taxonomy
                st.rerun()

        st.divider()


def render_bulk_selection_controls_taxonomies(taxonomies: List[Dict[str, Any]]):
    """Render bulk selection controls for taxonomies."""
    # Initialize selected taxonomies in session state if not exists
    if "selected_taxonomies" not in st.session_state:
        st.session_state.selected_taxonomies = set()

    # Select All checkbox
    all_selected = len(st.session_state.selected_taxonomies) == len(taxonomies) and len(taxonomies) > 0
    select_all = st.checkbox(
        "Select All",
        value=all_selected,
        key="select_all_taxonomies",
        help="Select/deselect all taxonomies"
    )

    # Update selection when select all changes
    if select_all and not all_selected:
        st.session_state.selected_taxonomies = {t["id"] for t in taxonomies}
        st.rerun()
    elif not select_all and all_selected:
        st.session_state.selected_taxonomies.clear()
        st.rerun()

    # Delete Selected button
    selected_count = len(st.session_state.selected_taxonomies)
    if selected_count > 0:
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button(
                f"🗑️ Delete Selected ({selected_count})",
                width='stretch',
                type="primary",
                help=f"Delete {selected_count} selected taxonomy(es)"
            ):
                st.session_state.bulk_deleting_taxonomies = True
                st.rerun()
        with col2:
            if st.button("Clear Selection", width='stretch'):
                st.session_state.selected_taxonomies.clear()
                st.rerun()

    # Bulk delete confirmation modal
    if st.session_state.get("bulk_deleting_taxonomies", False):
        render_bulk_delete_taxonomies_confirmation(taxonomies)


def render_bulk_delete_taxonomies_confirmation(taxonomies: List[Dict[str, Any]]):
    """Render bulk delete confirmation modal for taxonomies."""
    selected_ids = st.session_state.selected_taxonomies
    selected_taxonomies = [t for t in taxonomies if t["id"] in selected_ids]

    st.subheader("🗑️ Delete Multiple Taxonomies")
    st.warning(f"Are you sure you want to delete **{len(selected_taxonomies)}** taxonomy(es)?")
    st.error("This action cannot be undone. All taxonomy files will be permanently removed.")

    # Show selected taxonomies
    with st.expander("Selected Taxonomies"):
        for taxonomy in selected_taxonomies:
            st.write(f"• **{taxonomy.get('name', 'Unknown')}** (ID: {taxonomy.get('id')})")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("❌ Yes, Delete All", width='stretch', type="primary"):
            try:
                with st.spinner(f"Deleting {len(selected_taxonomies)} taxonomies..."):
                    response = api_client.delete("/api/v1/taxonomy/", data={
                        "taxonomy_ids": list(selected_ids)
                    })

                deleted_count = len(response.get("deleted", []))
                failed_count = len(response.get("failed", []))

                if failed_count == 0:
                    st.success(f"Successfully deleted {deleted_count} taxonomy(es)!")
                else:
                    st.warning(f"Deleted {deleted_count} taxonomy(es), but {failed_count} failed. Check the errors below.")
                    for failure in response.get("failed", []):
                        st.error(f"Taxonomy {failure['id']}: {failure['error']}")

                # Clear selection and modal state
                st.session_state.selected_taxonomies.clear()
                del st.session_state.bulk_deleting_taxonomies
                st.cache_data.clear()
                st.rerun()

            except APIError as e:
                st.error(f"Failed to delete taxonomies: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

    with col2:
        if st.button("⬅️ Cancel", width='stretch'):
            del st.session_state.bulk_deleting_taxonomies
            st.rerun()