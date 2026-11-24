"""
Taxonomy Display Module.
Handles all taxonomy display and viewing functionality.
"""

import streamlit as st
import json
import logging
from typing import Dict, Any, List
from src.compileo.features.gui.services.api_client import api_client, APIError
from src.compileo.features.gui.components.taxonomy_components import render_taxonomy_tree

logger = logging.getLogger(__name__)

MAX_INLINE_TAXONOMY_TREE_NODES = 1500
LARGE_TAXONOMY_PAGE_SIZE = 25


def _taxonomy_tree_stats(node: Dict[str, Any], depth: int = 1) -> Dict[str, int]:
    """Return cheap structural stats for taxonomy rendering diagnostics."""
    if not isinstance(node, dict):
        return {"nodes": 0, "max_depth": max(0, depth - 1), "max_children": 0, "max_label_len": 0}

    children = node.get("children")
    if not isinstance(children, list):
        children = []

    stats = {
        "nodes": 1,
        "max_depth": depth,
        "max_children": len(children),
        "max_label_len": len(str(node.get("name") or "")),
    }
    for child in children:
        child_stats = _taxonomy_tree_stats(child, depth + 1)
        stats["nodes"] += child_stats["nodes"]
        stats["max_depth"] = max(stats["max_depth"], child_stats["max_depth"])
        stats["max_children"] = max(stats["max_children"], child_stats["max_children"])
        stats["max_label_len"] = max(stats["max_label_len"], child_stats["max_label_len"])
    return stats


def _taxonomy_children(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    children = node.get("children") if isinstance(node, dict) else []
    return [child for child in children if isinstance(child, dict)] if isinstance(children, list) else []


def _taxonomy_node_at_path(root: Dict[str, Any], path: List[int]) -> Dict[str, Any]:
    node = root
    for index in path:
        children = _taxonomy_children(node)
        if index < 0 or index >= len(children):
            return root
        node = children[index]
    return node


def _taxonomy_path_labels(root: Dict[str, Any], path: List[int]) -> List[str]:
    labels = [str(root.get("name") or "Root")]
    node = root
    for index in path:
        children = _taxonomy_children(node)
        if index < 0 or index >= len(children):
            break
        node = children[index]
        labels.append(str(node.get("name") or f"Category {index + 1}"))
    return labels


def _short_label(value: Any, max_chars: int = 120) -> str:
    label = " ".join(str(value or "Untitled").split())
    return label if len(label) <= max_chars else f"{label[: max_chars - 1]}…"


def _render_large_taxonomy_tree(taxonomy_data: Dict[str, Any], taxonomy_id: str, stats: Dict[str, int]) -> None:
    """Render a large taxonomy with lazy, paginated navigation instead of all nodes."""
    st.warning(
        "This taxonomy is too large for full recursive rendering. "
        "Using safe paginated browsing to avoid overloading the Streamlit websocket."
    )
    st.caption(
        f"Nodes: {stats['nodes']:,} · Max depth: {stats['max_depth']} · "
        f"Largest sibling group: {stats['max_children']:,}"
    )

    key_prefix = f"large_taxonomy_{taxonomy_id}"
    path_key = f"{key_prefix}_path"
    page_key = f"{key_prefix}_page"

    if path_key not in st.session_state or not isinstance(st.session_state[path_key], list):
        st.session_state[path_key] = []
    if page_key not in st.session_state:
        st.session_state[page_key] = 0

    current_path = []
    for path_item in st.session_state.get(path_key, []):
        try:
            current_path.append(int(path_item))
        except (TypeError, ValueError):
            continue
    current_node = _taxonomy_node_at_path(taxonomy_data, current_path)
    if current_node is taxonomy_data and current_path:
        current_path = []
        st.session_state[path_key] = []

    path_labels = _taxonomy_path_labels(taxonomy_data, current_path)
    st.markdown(f"**Current path:** {' › '.join(_short_label(label, 80) for label in path_labels)}")

    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 4])
    with nav_col1:
        if st.button("⬅️ Up", key=f"{key_prefix}_up", disabled=not current_path):
            st.session_state[path_key] = current_path[:-1]
            st.session_state[page_key] = 0
            st.rerun()
    with nav_col2:
        if st.button("🏠 Root", key=f"{key_prefix}_root", disabled=not current_path):
            st.session_state[path_key] = []
            st.session_state[page_key] = 0
            st.rerun()

    current_children = _taxonomy_children(current_node)
    with st.expander(f"Current node: {_short_label(current_node.get('name'), 160)}", expanded=True):
        description = current_node.get("description")
        if description:
            st.write(f"**Description:** {description}")
        st.metric("Direct children", len(current_children))

    if not current_children:
        st.info("This category has no subcategories.")
        return

    total_pages = max(1, (len(current_children) + LARGE_TAXONOMY_PAGE_SIZE - 1) // LARGE_TAXONOMY_PAGE_SIZE)
    page = max(0, min(int(st.session_state.get(page_key, 0) or 0), total_pages - 1))
    st.session_state[page_key] = page

    page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
    with page_col1:
        if st.button("Previous", key=f"{key_prefix}_prev", disabled=page <= 0):
            st.session_state[page_key] = page - 1
            st.rerun()
    with page_col2:
        st.markdown(f"**Page {page + 1} of {total_pages}**")
    with page_col3:
        if st.button("Next", key=f"{key_prefix}_next", disabled=page >= total_pages - 1):
            st.session_state[page_key] = page + 1
            st.rerun()

    start = page * LARGE_TAXONOMY_PAGE_SIZE
    end = min(start + LARGE_TAXONOMY_PAGE_SIZE, len(current_children))
    for absolute_index, child in enumerate(current_children[start:end], start=start):
        child_children = _taxonomy_children(child)
        row_col1, row_col2, row_col3 = st.columns([5, 1, 1])
        with row_col1:
            st.markdown(f"**{_short_label(child.get('name'), 180)}**")
            if child.get("description"):
                st.caption(_short_label(child.get("description"), 220))
        with row_col2:
            st.caption(f"{len(child_children)} child nodes")
        with row_col3:
            if st.button("Open", key=f"{key_prefix}_open_{'_'.join(map(str, current_path))}_{absolute_index}"):
                st.session_state[path_key] = current_path + [absolute_index]
                st.session_state[page_key] = 0
                st.rerun()


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
        if not taxonomy_detail:
            logger.debug(
                "DEBUG: taxonomy_viewer_detail_empty taxonomy_id=%s name=%s",
                taxonomy.get("id"),
                taxonomy.get("name"),
            )
            st.warning("Taxonomy details could not be loaded.")
            return

        taxonomy_data = taxonomy_detail.get('taxonomy', {})
        stats = _taxonomy_tree_stats(taxonomy_data)
        logger.debug(
            "DEBUG: taxonomy_viewer_render_start taxonomy_id=%s name=%s listed_categories=%s nodes=%s max_depth=%s max_children=%s max_label_len=%s",
            taxonomy.get("id"),
            taxonomy.get("name"),
            taxonomy.get("categories_count"),
            stats["nodes"],
            stats["max_depth"],
            stats["max_children"],
            stats["max_label_len"],
        )
        is_large_tree = stats["nodes"] > MAX_INLINE_TAXONOMY_TREE_NODES
        search_term = st.text_input("🔍 Search categories", placeholder="Enter category name...", key=f"search_{taxonomy['id']}")
        if is_large_tree:
            if search_term:
                st.info("Search is disabled in safe large-taxonomy mode. Navigate pages to inspect categories.")
            _render_large_taxonomy_tree(taxonomy_data, str(taxonomy['id']), stats)
        else:
            render_taxonomy_tree(taxonomy_data, taxonomy_id=taxonomy['id'], search_term=search_term.lower() if search_term else None)
        logger.debug(
            "DEBUG: taxonomy_viewer_render_done taxonomy_id=%s nodes=%s",
            taxonomy.get("id"),
            stats["nodes"],
        )
    except APIError as e:
        st.error(f"Failed to load taxonomy details: {e}")


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
    logger.debug(
        "DEBUG: taxonomy_browser_management_loaded count=%s names=%s",
        len(taxonomies) if isinstance(taxonomies, list) else "not_list",
        [t.get("name") for t in taxonomies[:5]] if isinstance(taxonomies, list) else [],
    )

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
        view_mode = st.selectbox("View mode", ["List", "Tree"], key="unified_view_mode")

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

                # 3 columns since View Details is removed in tree view (it's already shown)
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("✏️ Edit", key=f"edit_selected_{selected_taxonomy['id']}", width='stretch'):
                        st.session_state.editing_taxonomy = selected_taxonomy
                        st.rerun()
                with col2:
                    dl_state_key = f"dl_taxonomy_ready_{selected_taxonomy['id']}"
                    if not st.session_state.get(dl_state_key):
                        if st.button("📤 Export JSON", key=f"export_selected_{selected_taxonomy['id']}", width='stretch'):
                            st.session_state[dl_state_key] = True
                            st.rerun()
                    
                    if st.session_state.get(dl_state_key):
                        try:
                            response = api_client.get(f"/api/v1/taxonomy/{selected_taxonomy['id']}")
                            if response:
                                json_str = json.dumps(response, indent=2)
                                st.download_button(
                                    label="📥 Download",
                                    data=json_str,
                                    file_name=f"taxonomy_{selected_taxonomy['id']}_{int(time.time())}.json",
                                    mime="application/json",
                                    width='stretch',
                                    key=f"real_dl_taxonomy_{selected_taxonomy['id']}"
                                )
                            else:
                                st.error("No data to export.")
                                st.session_state[dl_state_key] = False
                        except APIError as e:
                            st.error(f"Failed to export taxonomy: {e}")
                            st.session_state[dl_state_key] = False
                with col3:
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
