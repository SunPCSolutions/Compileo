"""Interactive taxonomy tree selector component with checkbox selection."""

import streamlit as st
from typing import Dict, Any, List, Optional, Set, Tuple
import time


class TaxonomyTreeSelector:
    """Interactive taxonomy tree component with checkbox selection and filtering."""

    def __init__(self, taxonomy_data: Dict[str, Any], key_prefix: str = "taxonomy_selector"):
        """
        Initialize the taxonomy tree selector.

        Args:
            taxonomy_data: Taxonomy data with hierarchical structure
            key_prefix: Unique prefix for session state keys
        """
        self.taxonomy_data = taxonomy_data
        self.key_prefix = key_prefix
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize session state for selection management."""
        # Selected category IDs
        selected_key = f"{self.key_prefix}_selected"
        if selected_key not in st.session_state:
            st.session_state[selected_key] = set()

        # Expanded nodes
        expanded_key = f"{self.key_prefix}_expanded"
        if expanded_key not in st.session_state:
            st.session_state[expanded_key] = set()

        # Search term
        search_key = f"{self.key_prefix}_search"
        if search_key not in st.session_state:
            st.session_state[search_key] = ""

    @property
    def selected_categories(self) -> Set[str]:
        """Get currently selected category IDs."""
        return st.session_state.get(f"{self.key_prefix}_selected", set())

    @selected_categories.setter
    def selected_categories(self, value: Set[str]):
        """Set selected category IDs."""
        st.session_state[f"{self.key_prefix}_selected"] = value

    @property
    def expanded_nodes(self) -> Set[str]:
        """Get currently expanded node paths."""
        return st.session_state.get(f"{self.key_prefix}_expanded", set())

    @expanded_nodes.setter
    def expanded_nodes(self, value: Set[str]):
        """Set expanded node paths."""
        st.session_state[f"{self.key_prefix}_expanded"] = value

    @property
    def search_term(self) -> str:
        """Get current search term."""
        return st.session_state.get(f"{self.key_prefix}_search", "")

    @search_term.setter
    def search_term(self, value: str):
        """Set search term."""
        st.session_state[f"{self.key_prefix}_search"] = value

    def render(self) -> Set[str]:
        """
        Render the taxonomy tree selector.

        Returns:
            Set of selected category IDs
        """
        st.markdown("### 🏷️ Taxonomy Category Selector")

        # Search and filter controls
        self._render_search_controls()

        # Selection summary
        self._render_selection_summary()

        # Tree rendering
        with st.container():
            if self.taxonomy_data:
                self._render_tree_node(self.taxonomy_data, path="", level=0)
            else:
                st.info("No taxonomy data available.")

        return self.selected_categories

    def _render_search_controls(self):
        """Render search and filter controls."""
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            search = st.text_input(
                "🔍 Search categories",
                value=self.search_term,
                placeholder="Enter category name...",
                key=f"{self.key_prefix}_search_input",
                help="Filter categories by name or description"
            )
            if search != self.search_term:
                self.search_term = search

        with col2:
            if st.button("🔄 Clear Search", key=f"{self.key_prefix}_clear_search"):
                self.search_term = ""
                st.rerun()

        with col3:
            if st.button("📂 Expand All", key=f"{self.key_prefix}_expand_all"):
                self._expand_all_nodes()
                st.rerun()
            if st.button("📁 Collapse All", key=f"{self.key_prefix}_collapse_all"):
                self.expanded_nodes.clear()
                st.rerun()

    def _render_selection_summary(self):
        """Render selection summary and bulk actions."""
        selected_count = len(self.selected_categories)

        if selected_count > 0:
            st.info(f"✅ {selected_count} categories selected")

            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("❌ Clear All", key=f"{self.key_prefix}_clear_all"):
                    self.selected_categories.clear()
                    st.rerun()

            with col2:
                if st.button("📋 Copy IDs", key=f"{self.key_prefix}_copy_ids"):
                    # Copy selected IDs to clipboard (would need JavaScript)
                    st.code(", ".join(sorted(self.selected_categories)), language=None)

            with col3:
                st.write("")  # Spacer

    def _render_tree_node(self, node: Dict[str, Any], path: str, level: int):
        """
        Recursively render a tree node.

        Args:
            node: Current node data
            path: Path to this node (for unique keys)
            level: Depth level (for indentation)
        """
        if not node:
            return

        node_name = node.get('name', 'Unknown')
        node_desc = node.get('description', '')
        confidence = node.get('confidence_threshold', 0.0)
        children = node.get('children', [])

        # Apply search filter
        if self.search_term:
            search_lower = self.search_term.lower()
            if (search_lower not in node_name.lower() and
                search_lower not in node_desc.lower() and
                not any(search_lower in child.get('name', '').lower() or
                       search_lower in child.get('description', '').lower()
                       for child in children)):
                return

        # Generate unique node ID using taxonomy ID instead of path
        taxonomy_id = node.get('id')
        if taxonomy_id:
            node_id = str(taxonomy_id)
        else:
            node_id = path if path else "0"

        # Determine selection state
        selection_state = self._get_node_selection_state(node, node_id)

        # Color coding based on confidence
        color = "🟢" if confidence >= 0.8 else "🟡" if confidence >= 0.6 else "🔴"

        # Create expander for this node
        is_expanded = node_id in self.expanded_nodes
        has_children = bool(children)

        # Node header with checkbox and expand/collapse
        col1, col2, col3 = st.columns([0.5, 0.5, 4])

        with col1:
            # Checkbox for selection
            checkbox_state = self._get_checkbox_state(selection_state)
            if st.checkbox(
                f"Select {node_name}",
                value=checkbox_state[0],
                key=f"checkbox_{node_id}",
                label_visibility="collapsed",
                help=f"Select {node_name}"
            ) != checkbox_state[0]:
                self._toggle_node_selection(node, node_id, selection_state)
                st.rerun()

        with col2:
            if has_children:
                # Expand/collapse button
                icon = "📂" if is_expanded else "📁"
                if st.button(icon, key=f"expand_{node_id}", help="Toggle expansion"):
                    if is_expanded:
                        self.expanded_nodes.discard(node_id)
                    else:
                        self.expanded_nodes.add(node_id)
                    st.rerun()

        with col3:
            # Node title with confidence
            st.markdown(f"**{color} {node_name}** (Confidence: {confidence:.2f})")

            # Description
            if node_desc:
                st.caption(node_desc)

        # Render children if expanded
        if is_expanded and has_children:
            with st.container():
                st.markdown("---")
                for i, child in enumerate(children):
                    # Use taxonomy ID for child path if available, otherwise fall back to index
                    child_taxonomy_id = child.get('id')
                    if child_taxonomy_id:
                        child_path = str(child_taxonomy_id)
                    else:
                        child_path = f"{node_id}_{i}"
                    self._render_tree_node(child, child_path, level + 1)

    def _get_node_selection_state(self, node: Dict[str, Any], node_id: str) -> Tuple[bool, bool, bool]:
        """
        Get selection state for a node.

        Returns:
            Tuple of (is_selected, has_selected_children, has_unselected_children)
        """
        is_selected = node_id in self.selected_categories
        has_selected_children = False
        has_unselected_children = False

        children = node.get('children', [])
        if children:
            for i, child in enumerate(children):
                # Use taxonomy ID for child ID if available, otherwise fall back to index
                child_taxonomy_id = child.get('id')
                if child_taxonomy_id:
                    child_id = str(child_taxonomy_id)
                else:
                    child_id = f"{node_id}_{i}"
                child_selected = child_id in self.selected_categories

                if child_selected:
                    has_selected_children = True
                else:
                    has_unselected_children = True

                # Recursively check grandchildren
                child_state = self._get_node_selection_state(child, child_id)
                if child_state[1]:  # has selected children
                    has_selected_children = True
                if child_state[2]:  # has unselected children
                    has_unselected_children = True

        return is_selected, has_selected_children, has_unselected_children

    def _get_checkbox_state(self, selection_state: Tuple[bool, bool, bool]) -> Tuple[bool, bool]:
        """
        Get checkbox state from selection state.

        For individual selection, checkbox reflects only the node's own selection state.

        Returns:
            Tuple of (checked, indeterminate)
        """
        is_selected, _, _ = selection_state
        return is_selected, False

    def _toggle_node_selection(self, node: Dict[str, Any], node_id: str, selection_state: Tuple[bool, bool, bool]):
        """
        Toggle selection for a node individually (without affecting children).
        """
        if node_id in self.selected_categories:
            self.selected_categories.discard(node_id)
        else:
            self.selected_categories.add(node_id)

    def _select_node_recursive(self, node: Dict[str, Any], node_id: str):
        """Recursively select a node and all its children."""
        self.selected_categories.add(node_id)

        children = node.get('children', [])
        for i, child in enumerate(children):
            child_path = f"{node_id}_{i}"
            child_node_id = child_path
            self._select_node_recursive(child, child_node_id)

    def _deselect_node_recursive(self, node: Dict[str, Any], node_id: str):
        """Recursively deselect a node and all its children."""
        self.selected_categories.discard(node_id)

        children = node.get('children', [])
        for i, child in enumerate(children):
            child_path = f"{node_id}_{i}"
            child_node_id = child_path
            self._deselect_node_recursive(child, child_node_id)

    def _expand_all_nodes(self):
        """Expand all nodes in the tree."""
        all_expanded = set()

        def expand_recursive(node: Dict[str, Any], path: str):
            node_id = path if path else "0"
            all_expanded.add(node_id)

            children = node.get('children', [])
            for i, child in enumerate(children):
                child_path = f"{node_id}_{i}"
                expand_recursive(child, child_path)

        if self.taxonomy_data:
            expand_recursive(self.taxonomy_data, "")

        self.expanded_nodes = all_expanded

    def select_all_at_level(self, level: int):
        """
        Select all categories at a specific level.

        Args:
            level: The level to select all categories from (0 = root)
        """
        def select_at_level_recursive(node: Dict[str, Any], current_level: int, path: str):
            if current_level == level:
                node_id = path if path else "0"
                current_selected = self.selected_categories.copy()
                current_selected.add(node_id)
                self.selected_categories = current_selected
                return

            children = node.get('children', [])
            node_id = path if path else "0"
            for i, child in enumerate(children):
                child_path = f"{node_id}_{i}"
                select_at_level_recursive(child, current_level + 1, child_path)

        if self.taxonomy_data:
            select_at_level_recursive(self.taxonomy_data, 0, "")

    def _build_node_name_mapping(self) -> Dict[str, str]:
        """
        Build a mapping from node_id to full category name in dot-separated format.

        Returns:
            Dictionary mapping node_id to full category path
        """
        mapping = {}

        def traverse_node(node: Dict[str, Any], path: str, dot_path: str):
            node_name = node.get('name', 'Unknown')
            node_id = path if path else "0"
            mapping[node_id] = dot_path

            children = node.get('children', [])
            for i, child in enumerate(children):
                child_path = f"{node_id}_{i}"
                child_dot_path = f"{dot_path}.{child.get('name', 'Unknown')}"
                traverse_node(child, child_path, child_dot_path)

        if self.taxonomy_data:
            root_name = self.taxonomy_data.get('name', 'Unknown')
            traverse_node(self.taxonomy_data, "", root_name)

        return mapping

    def get_selected_category_names(self) -> List[str]:
        """Get names of selected categories in dot-separated format."""
        category_names = []

        for selected_path in self.selected_categories:
            # Parse the selected path format: "RootName_index_ChildName_index_..."
            # Convert to dot-separated: "RootName.ChildName.GrandChildName..."
            parsed_name = self._parse_selected_path_to_name(selected_path)
            if parsed_name:
                category_names.append(parsed_name)

        return category_names

    def _parse_selected_path_to_name(self, selected_path: str) -> Optional[str]:
        """
        Parse a selected path string into a dot-separated category name.

        Args:
            selected_path: Path in format like "0_0_1" (indices)

        Returns:
            Dot-separated name like "Disease.Symptoms.Acute" or None if invalid
        """
        if not selected_path:
            return None

        # Split by underscore to get indices
        indices = selected_path.split('_')

        try:
            # Convert to integers
            indices = [int(idx) for idx in indices]
        except ValueError:
            return None

        # Traverse the taxonomy tree using indices
        current_node = self.taxonomy_data
        category_names = []

        for idx in indices:
            if current_node is None:
                return None

            category_names.append(current_node.get('name', 'Unknown'))

            # Move to child at this index
            children = current_node.get('children', [])
            if idx >= len(children):
                return None
            current_node = children[idx]

        return '.'.join(category_names)

    def clear_selection(self):
        """Clear all selections."""
        self.selected_categories.clear()

    def get_selection_stats(self) -> Dict[str, Any]:
        """Get selection statistics."""
        total_selected = len(self.selected_categories)

        # Count total categories
        def count_categories(node: Dict[str, Any]) -> int:
            count = 1
            for child in node.get('children', []):
                count += count_categories(child)
            return count

        total_categories = count_categories(self.taxonomy_data) if self.taxonomy_data else 0

        return {
            "selected": total_selected,
            "total": total_categories,
            "percentage": (total_selected / total_categories * 100) if total_categories > 0 else 0.0
        }


def render_taxonomy_tree_selector(
    taxonomy_data: Dict[str, Any],
    key_prefix: str = "taxonomy_selector"
) -> Set[str]:
    """
    Convenience function to render a taxonomy tree selector.

    Args:
        taxonomy_data: Taxonomy data with hierarchical structure
        key_prefix: Unique prefix for session state keys

    Returns:
        Set of selected category IDs
    """
    selector = TaxonomyTreeSelector(taxonomy_data, key_prefix)
    return selector.render()