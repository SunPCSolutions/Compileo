"""
Optimized taxonomy tree selector with lazy loading and virtualization for large hierarchies.
"""

import streamlit as st
from typing import Dict, Any, List, Optional, Set, Tuple, Iterator
import time
from functools import lru_cache
from dataclasses import dataclass


@dataclass
class TreeNode:
    """Lightweight tree node for performance."""
    id: str
    name: str
    description: str
    confidence: float
    children_count: int
    level: int
    parent_id: Optional[str] = None


class OptimizedTaxonomyTreeSelector:
    """High-performance taxonomy tree with lazy loading and virtualization."""

    def __init__(
        self,
        taxonomy_data: Dict[str, Any],
        key_prefix: str = "opt_taxonomy",
        page_size: int = 50,
        max_cache_size: int = 1000
    ):
        self.taxonomy_data = taxonomy_data
        self.key_prefix = key_prefix
        self.page_size = page_size
        self.max_cache_size = max_cache_size

        # Performance caches
        self._node_cache: Dict[str, TreeNode] = {}
        self._children_cache: Dict[str, List[str]] = {}
        self._search_cache: Dict[str, List[str]] = {}

        # Session state
        self._init_session_state()

        # Performance metrics
        self.render_times: List[float] = []
        self.cache_hits = 0
        self.cache_misses = 0

    def _init_session_state(self):
        """Initialize session state."""
        keys = [
            f"{self.key_prefix}_selected",
            f"{self.key_prefix}_expanded",
            f"{self.key_prefix}_search",
            f"{self.key_prefix}_page",
            f"{self.key_prefix}_lazy_loaded"
        ]

        for key in keys:
            if key not in st.session_state:
                if key.endswith("_selected") or key.endswith("_expanded") or key.endswith("_lazy_loaded"):
                    st.session_state[key] = set()
                elif key.endswith("_page"):
                    st.session_state[key] = 0
                else:
                    st.session_state[key] = ""

    @property
    def selected_categories(self) -> Set[str]:
        return st.session_state.get(f"{self.key_prefix}_selected", set())

    @selected_categories.setter
    def selected_categories(self, value: Set[str]):
        st.session_state[f"{self.key_prefix}_selected"] = value

    @property
    def expanded_nodes(self) -> Set[str]:
        return st.session_state.get(f"{self.key_prefix}_expanded", set())

    @expanded_nodes.setter
    def expanded_nodes(self, value: Set[str]):
        st.session_state[f"{self.key_prefix}_expanded"] = value

    @property
    def search_term(self) -> str:
        return st.session_state.get(f"{self.key_prefix}_search", "")

    @search_term.setter
    def search_term(self, value: str):
        st.session_state[f"{self.key_prefix}_search"] = value
        # Clear search cache when search changes
        self._search_cache.clear()

    @property
    def current_page(self) -> int:
        return st.session_state.get(f"{self.key_prefix}_page", 0)

    @current_page.setter
    def current_page(self, value: int):
        st.session_state[f"{self.key_prefix}_page"] = value

    def render(self) -> Set[str]:
        """Render the optimized taxonomy tree."""
        start_time = time.time()

        st.markdown("### 🏷️ Optimized Taxonomy Selector")

        # Controls
        self._render_controls()

        # Performance metrics
        if st.checkbox("Performance Metrics", key=f"{self.key_prefix}_metrics"):
            self._render_metrics()

        # Main content
        with st.container():
            if not self.taxonomy_data:
                st.info("No taxonomy data available.")
                return self.selected_categories

            # Determine rendering strategy
            total_nodes = self._count_nodes(self.taxonomy_data)
            if total_nodes > self.page_size * 3:
                self._render_paginated()
            else:
                self._render_full()

        # Track performance
        render_time = time.time() - start_time
        self.render_times.append(render_time)
        if len(self.render_times) > 10:
            self.render_times.pop(0)

        return self.selected_categories

    def _render_controls(self):
        """Render search and control elements."""
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

        with col1:
            search = st.text_input(
                "🔍 Search",
                value=self.search_term,
                placeholder="Filter categories...",
                key=f"{self.key_prefix}_search_input"
            )
            if search != self.search_term:
                self.search_term = search

        with col2:
            if st.button("❌ Clear", key=f"{self.key_prefix}_clear"):
                self.search_term = ""
                self.selected_categories.clear()
                st.rerun()

        with col3:
            if st.button("📂 Expand All", key=f"{self.key_prefix}_expand_all"):
                self._expand_all()
                st.rerun()

        with col4:
            if st.button("📁 Collapse All", key=f"{self.key_prefix}_collapse_all"):
                self.expanded_nodes.clear()
                st.rerun()

        # Selection summary
        if self.selected_categories:
            st.success(f"✅ {len(self.selected_categories)} categories selected")

    def _render_paginated(self):
        """Render tree with pagination for large datasets."""
        st.markdown("**📄 Paginated View** (Optimized for large taxonomies)")

        # Get paginated nodes
        visible_nodes = self._get_visible_nodes_paginated()

        if not visible_nodes:
            st.info("No matching nodes found.")
            return

        # Render current page
        for node in visible_nodes:
            self._render_node_compact(node)

        # Pagination controls
        self._render_pagination()

    def _render_full(self):
        """Render full tree for smaller datasets."""
        st.markdown("**🌳 Full Tree View**")

        def render_recursive(node_data: Dict[str, Any], path: str, level: int):
            node = self._get_cached_node(path, node_data, level)
            self._render_node_compact(node)

            # Render children if expanded
            if node.id in self.expanded_nodes:
                for i, child in enumerate(node_data.get('children', [])):
                    child_path = f"{path}_{i}"
                    render_recursive(child, child_path, level + 1)

        render_recursive(self.taxonomy_data, "root", 0)

    def _render_node_compact(self, node: TreeNode):
        """Render a node in compact format."""
        indent = "&nbsp;" * (node.level * 3)

        # Color coding
        color = "🟢" if node.confidence >= 0.8 else "🟡" if node.confidence >= 0.6 else "🔴"

        # Layout
        col1, col2, col3 = st.columns([0.5, 0.5, 4])

        with col1:
            is_selected = node.id in self.selected_categories
            if st.checkbox(
                f"select_{node.id}",
                value=is_selected,
                key=f"chk_{node.id}",
                label_visibility="collapsed"
            ) != is_selected:
                if is_selected:
                    self.selected_categories.discard(node.id)
                else:
                    self.selected_categories.add(node.id)
                st.rerun()

        with col2:
            if node.children_count > 0:
                is_expanded = node.id in self.expanded_nodes
                icon = "📂" if is_expanded else "📁"
                if st.button(icon, key=f"exp_{node.id}"):
                    if is_expanded:
                        self.expanded_nodes.discard(node.id)
                    else:
                        self.expanded_nodes.add(node.id)
                    st.rerun()

        with col3:
            title = f"{indent}**{color} {node.name}**"
            if node.confidence > 0:
                title += f" ({node.confidence:.2f})"
            st.markdown(title)

            if node.description:
                st.caption(f"{indent}{node.description}")

    def _render_pagination(self):
        """Render pagination controls."""
        total_pages = max(1, (len(self._get_all_matching_nodes()) + self.page_size - 1) // self.page_size)

        if total_pages <= 1:
            return

        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if self.current_page > 0:
                if st.button("⬅️ Previous", key=f"{self.key_prefix}_prev"):
                    self.current_page -= 1
                    st.rerun()

        with col2:
            st.markdown(f"**Page {self.current_page + 1} of {total_pages}**")

        with col3:
            if self.current_page < total_pages - 1:
                if st.button("Next ➡️", key=f"{self.key_prefix}_next"):
                    self.current_page += 1
                    st.rerun()

    def _render_metrics(self):
        """Render performance metrics."""
        st.markdown("#### Performance Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_render = sum(self.render_times) / len(self.render_times) if self.render_times else 0
            st.metric("Avg Render Time", ".3f")

        with col2:
            st.metric("Cache Size", len(self._node_cache))

        with col3:
            total_nodes = self._count_nodes(self.taxonomy_data)
            st.metric("Total Nodes", total_nodes)

        with col4:
            hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
            st.metric("Cache Hit Rate", ".1%")

    @lru_cache(maxsize=512)
    def _get_cached_node(self, path: str, node_data: tuple, level: int) -> TreeNode:
        """Get cached node with LRU eviction."""
        # Convert tuple to dict
        node_dict = dict(node_data)

        node_id = f"{path}_{node_dict.get('name', 'unknown')}" if path != "root" else "root"

        if node_id in self._node_cache:
            self.cache_hits += 1
            return self._node_cache[node_id]

        self.cache_misses += 1

        # Create new node
        node = TreeNode(
            id=node_id,
            name=node_dict.get('name', 'Unknown'),
            description=node_dict.get('description', ''),
            confidence=node_dict.get('confidence_threshold', 0.0),
            children_count=len(node_dict.get('children', [])),
            level=level
        )

        # Cache it
        if len(self._node_cache) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._node_cache))
            del self._node_cache[oldest_key]

        self._node_cache[node_id] = node
        return node

    def _get_visible_nodes_paginated(self) -> List[TreeNode]:
        """Get visible nodes for current page."""
        all_matching = self._get_all_matching_nodes()
        start_idx = self.current_page * self.page_size
        end_idx = start_idx + self.page_size

        visible_nodes = []
        for i, node_path in enumerate(all_matching[start_idx:end_idx]):
            node_data = self._get_node_data_by_path(node_path)
            if node_data:
                level = node_path.count('_')  # Simple level calculation
                node = self._get_cached_node(node_path, tuple(node_data.items()), level)
                visible_nodes.append(node)

        return visible_nodes

    def _get_all_matching_nodes(self) -> List[str]:
        """Get all node paths that match current search."""
        cache_key = f"search_{self.search_term}_{len(self.expanded_nodes)}"

        if cache_key in self._search_cache:
            return self._search_cache[cache_key]

        matching_paths = []

        def collect_matching(node_data: Dict[str, Any], path: str):
            node_name = node_data.get('name', '')
            node_desc = node_data.get('description', '')

            # Check if matches search
            if self.search_term:
                search_lower = self.search_term.lower()
                if (search_lower not in node_name.lower() and
                    search_lower not in node_desc.lower()):
                    return  # Skip this branch

            matching_paths.append(path)

            # Add children if expanded
            node_id = f"{path}_{node_name}" if path != "root" else "root"
            if node_id in self.expanded_nodes:
                for i, child in enumerate(node_data.get('children', [])):
                    child_path = f"{path}_{i}"
                    collect_matching(child, child_path)

        collect_matching(self.taxonomy_data, "root")
        self._search_cache[cache_key] = matching_paths
        return matching_paths

    def _get_node_data_by_path(self, path: str) -> Optional[Dict[str, Any]]:
        """Get node data by path."""
        if path == "root":
            return self.taxonomy_data

        parts = path.split('_')[1:]  # Skip 'root'
        current = self.taxonomy_data

        try:
            for part in parts:
                if part.isdigit():
                    current = current['children'][int(part)]
                else:
                    # Find by name
                    current = next(
                        (c for c in current.get('children', []) if c.get('name') == part),
                        None
                    )
                    if current is None:
                        return None
            return current
        except (KeyError, IndexError, TypeError):
            return None

    def _count_nodes(self, node_data: Dict[str, Any]) -> int:
        """Count total nodes in subtree."""
        count = 1
        for child in node_data.get('children', []):
            count += self._count_nodes(child)
        return count

    def _expand_all(self):
        """Expand all nodes."""
        def collect_all_paths(node_data: Dict[str, Any], path: str):
            node_id = f"{path}_{node_data.get('name', 'unknown')}" if path != "root" else "root"
            self.expanded_nodes.add(node_id)

            for i, child in enumerate(node_data.get('children', [])):
                child_path = f"{path}_{i}"
                collect_all_paths(child, child_path)

        collect_all_paths(self.taxonomy_data, "root")


def render_optimized_taxonomy_tree(
    taxonomy_data: Dict[str, Any],
    key_prefix: str = "opt_taxonomy"
) -> Set[str]:
    """
    Convenience function for optimized taxonomy tree rendering.

    Args:
        taxonomy_data: Taxonomy hierarchy data
        key_prefix: Unique key prefix for session state

    Returns:
        Set of selected category IDs
    """
    selector = OptimizedTaxonomyTreeSelector(taxonomy_data, key_prefix)
    return selector.render()