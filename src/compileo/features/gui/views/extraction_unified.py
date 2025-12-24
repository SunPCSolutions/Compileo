# Unified Extraction Interface for Compileo GUI
"""
Consolidated extraction interface that combines:
- Taxonomy-based extraction initiation
- Job monitoring and management
- Results viewing with taxonomy tree structure
"""

import streamlit as st
import time
import json
from typing import List, Dict, Any, Optional

from ..services.api_client import api_client, APIError
from ..services.document_api_service import get_projects
from ..services.job_monitoring_service import monitor_job_synchronously
from ..state.session_state import session_state
from ...extraction.error_logging import gui_logger


def render_extraction_unified():
    """Render the unified extraction interface with two main tabs."""
    st.markdown('<h1 class="section-header">🔍 Extraction</h1>', unsafe_allow_html=True)

    st.markdown("""
    Unified extraction interface for taxonomy-based content classification.
    Run extractions, monitor progress, and explore results in one consolidated view.
    """)

    # Create the two main tabs
    tab1, tab2 = st.tabs([
        "🏃 Run Extraction",
        "📋 Browse & Manage Extractions"
    ])

    with tab1:
        render_run_extraction()

    with tab2:
        render_extraction_browser_management()


def render_run_extraction():
    """Render the Run Extraction tab with taxonomy selector and parameters."""
    st.subheader("🏃 Run Taxonomy-Based Extraction")

    # Project Selection and Extraction Options
    projects = get_projects()
    if not projects:
        st.warning("No projects available. Please create a project first.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        project_options = {f"{p['name']} (ID: {p['id']})": p['id'] for p in projects}
        selected_project_display = st.selectbox(
            "Select Project",
            options=list(project_options.keys()),
            help="Choose the project for extraction",
            key="extraction_project_select"
        )
        selected_project_id = project_options[selected_project_display]

    with col2:
        # Extraction Type Selection
        extraction_type_options = {
            "Named Entity Recognition (NER)": "ner",
            "Whole Text Extraction": "whole_text"
        }

        selected_extraction_display = st.selectbox(
            "Extraction Type",
            options=list(extraction_type_options.keys()),
            index=0,  # Default to NER
            help="Choose the type of extraction: 'NER' extracts individual named entities, 'Whole Text' extracts complete text portions for AI fine-tuning",
            key="extraction_type_select"
        )
        selected_extraction_type = extraction_type_options[selected_extraction_display]

    with col3:
        # Extraction Mode Selection
        extraction_mode_options = {
            "Contextual Extraction": "contextual",
            "Document-Wide Extraction": "document_wide"
        }

        selected_mode_display = st.selectbox(
            "Extraction Mode",
            options=list(extraction_mode_options.keys()),
            index=0,  # Default to contextual
            help="Contextual: Filter by parent context. Document-Wide: Process all chunks.",
            key="extraction_mode_select"
        )

        selected_extraction_mode = extraction_mode_options[selected_mode_display]

    st.divider()

    st.divider()

    # Taxonomy Selection Section
    st.subheader("📂 Select Taxonomy Categories")

    # Get available taxonomies for the project
    taxonomies = get_available_taxonomies(selected_project_id)

    if not taxonomies:
        st.warning("No taxonomies available for this project. Please create a taxonomy first.")
        return

    taxonomy_options = {f"{t['name']} (ID: {t['id']})": t['id'] for t in taxonomies}
    selected_taxonomy_display = st.selectbox(
        "Select Taxonomy",
        options=list(taxonomy_options.keys()),
        help="Choose the taxonomy to use for extraction",
        key="extraction_taxonomy_select"
    )
    selected_taxonomy_id = taxonomy_options[selected_taxonomy_display]

    # Load and display taxonomy tree for category selection
    render_taxonomy_tree_selector(selected_taxonomy_id, selected_project_id, selected_project_display)

    st.divider()

    # AI Model Selection
    st.subheader("🤖 AI Models for Extraction")

    col1, col2 = st.columns(2)

    with col1:
        initial_classifier = st.selectbox(
            "Primary Classifier",
            options=["grok", "gemini", "openai", "ollama"],
            index=0,  # grok as default
            help="AI model for the initial classification stage",
            key="initial_classifier"
        )

    with col2:
        enable_validation = st.checkbox(
            "Enable Validation Stage",
            value=False,
            help="Use a second classifier to validate results",
            key="enable_validation"
        )

    validation_classifier = None
    if enable_validation:
        validation_classifier = st.selectbox(
            "Validation Classifier",
            options=["grok", "gemini", "openai", "ollama"],
            index=0,  # grok as default for validation
            help="AI model for the validation classification stage",
            key="validation_classifier"
        )

    st.divider()

    # Extraction Parameters
    st.subheader("⚙️ Extraction Parameters")

    col1, col2 = st.columns(2)

    with col1:
        extraction_depth = st.slider(
            "Extraction Depth",
            min_value=1,
            max_value=5,
            value=3,
            help="How deep in the taxonomy hierarchy to extract",
            key="extraction_depth"
        )

        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Minimum confidence score for classifications",
            key="confidence_threshold"
        )

    with col2:
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=100,
            value=10,
            help="Number of chunks to process per batch",
            key="batch_size"
        )

        max_chunks = st.number_input(
            "Max Chunks",
            min_value=1,
            max_value=10000,
            value=1000,
            help="Maximum number of chunks to process",
            key="max_chunks"
        )

    # Start Extraction Button
    if st.button("🚀 Start Extraction", type="primary", key="start_extraction"):
        start_extraction_job(
            selected_project_id,
            selected_taxonomy_id,
            extraction_depth,
            confidence_threshold,
            batch_size,
            max_chunks,
            initial_classifier,
            enable_validation,
            validation_classifier,
            selected_extraction_type,
            selected_extraction_mode
        )


def render_taxonomy_tree_selector(taxonomy_id: int, project_id: int, project_display: str):
    """Render interactive taxonomy tree for category selection using the proper taxonomy tree component."""
    try:
        # Get taxonomy structure
        taxonomy_data = api_client.get(f"/api/v1/taxonomy/{taxonomy_id}")

        if not taxonomy_data or 'taxonomy' not in taxonomy_data:
            st.error("Failed to load taxonomy structure.")
            return

        # Extract the actual taxonomy from the response
        taxonomy = taxonomy_data['taxonomy']
        if not taxonomy or 'children' not in taxonomy:
            st.error("Taxonomy structure is invalid or empty.")
            return

        # Add search functionality and batch controls
        col1, col2 = st.columns([3, 1])

        with col1:
            search_term = st.text_input(
                "🔍 Search categories",
                placeholder="Enter category name...",
                key=f"extraction_search_{taxonomy_id}"
            )

        with col2:
            # All batch control buttons in one row: Expand/Collapse | Select/Deselect
            btn1, btn2, spacer, btn3, btn4 = st.columns([1, 1, 0.2, 1, 1])
            with btn1:
                if st.button("📂 Expand All", key=f"expand_all_{taxonomy_id}", help="Expand all category nodes"):
                    st.session_state[f'expansion_state_{taxonomy_id}'] = 'expanded'
                    st.rerun()
            with btn2:
                if st.button("📁 Collapse All", key=f"collapse_all_{taxonomy_id}", help="Collapse all category nodes"):
                    st.session_state[f'expansion_state_{taxonomy_id}'] = 'collapsed'
                    st.rerun()
            with spacer:
                st.write("")  # Small spacer
            with btn3:
                if st.button("✅ Select All", key=f"select_all_{taxonomy_id}", help="Select all visible categories"):
                    select_all_categories(taxonomy_id, taxonomy.get('children', []), search_term.lower() if search_term else None)
                    st.rerun()
            with btn4:
                if st.button("❌ Deselect All", key=f"deselect_all_{taxonomy_id}", help="Deselect all categories"):
                    deselect_all_categories(taxonomy_id)
                    st.rerun()

        # Render taxonomy tree with selection checkboxes, skipping the root level
        st.markdown("**Select categories to extract from:**")
        # Skip the root taxonomy node and start from its children
        root_children = taxonomy.get('children', [])
        if root_children:
            expansion_state = st.session_state.get(f'expansion_state_{taxonomy_id}', 'default')
            selected_categories = render_selectable_taxonomy_tree_from_children(
                root_children, taxonomy_id, project_id, search_term.lower() if search_term else None, expansion_state
            )
        else:
            st.info("This taxonomy has no categories to select from.")
            selected_categories = []

        # Show selection summary
        if selected_categories:
            st.success(f"✅ Selected {len(selected_categories)} categories for extraction")
            with st.expander("View Selected Categories"):
                for cat in selected_categories:
                    st.write(f"• {cat}")
        else:
            st.info("ℹ️ No categories selected. Expand categories above and check the boxes to select them for extraction.")

    except Exception as e:
        st.error(f"Failed to load taxonomy: {e}")


def render_selectable_taxonomy_tree_from_children(children: List[Dict], taxonomy_id: int, project_id: int, search_term: Optional[str] = None, expansion_state: str = 'default') -> List[str]:
    """Render taxonomy tree starting from children (skipping root level) with selection checkboxes."""
    selected_categories = []

    # Initialize selected categories in session state
    if f'selected_categories_{taxonomy_id}' not in st.session_state:
        st.session_state[f'selected_categories_{taxonomy_id}'] = []

    # Render each child as a top-level selectable category
    for i, child in enumerate(children):
        child_selected = render_taxonomy_tree_with_selection(child, taxonomy_id, search_term, str(i), expansion_state)
        selected_categories.extend(child_selected)

    return selected_categories


def render_taxonomy_tree_with_selection(node: Dict[str, Any], taxonomy_id: int, search_term: Optional[str] = None, path: str = "", expansion_state: str = 'default', depth: int = 0) -> List[str]:
    """Recursively render taxonomy tree with selection checkboxes, similar to the taxonomy browser."""
    if not node:
        return []

    selected_categories = []
    node_name = node.get('name', 'Unknown')
    node_desc = node.get('description', '')
    confidence = node.get('confidence_threshold', 0.0)
    children = node.get('children', [])

    # Check search filter
    if search_term and search_term not in node_name.lower() and search_term not in node_desc.lower():
        if not any(search_term in child.get('name', '').lower() or search_term in child.get('description', '').lower() for child in children):
            return []

    # Determine confidence color
    color = "🟢" if confidence >= 0.8 else "🟡" if confidence >= 0.6 else "🔴"

    # Create a container for this category with checkbox and expander
    with st.container():
        col1, col2 = st.columns([0.1, 0.9])

        with col1:
            # Checkbox for category selection - ALWAYS use category ID as the identifier
            category_id = node.get('id')
            if category_id is None:
                # If no ID is available, show warning and skip checkbox
                st.warning(f"⚠️ Category '{node_name}' has no ID and cannot be selected.")
            else:
                # Ensure category_id is a string for consistent handling
                category_id = str(category_id)

                is_selected = st.checkbox(
                    f"Select {node_name}",  # Provide meaningful label for accessibility
                    key=f"select_cat_{taxonomy_id}_{category_id}_{path}",
                    value=category_id in st.session_state.get(f'selected_categories_{taxonomy_id}', []),
                    label_visibility="collapsed"  # Hide the label visually
                )

                if is_selected:
                    selected_categories.append(category_id)
                    # Update session state with category IDs
                    current_selected = st.session_state.get(f'selected_categories_{taxonomy_id}', [])
                    if category_id not in current_selected:
                        current_selected.append(category_id)
                        st.session_state[f'selected_categories_{taxonomy_id}'] = current_selected

        with col2:
            # Expander for category details (similar to taxonomy browser)
            expanded = (expansion_state == 'expanded') or (expansion_state == 'default' and depth == 0)
            with st.expander(f"{color} {node_name}", expanded=expanded):
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
                        child_selected = render_taxonomy_tree_with_selection(child, taxonomy_id, search_term, child_path, expansion_state, depth + 1)
                        selected_categories.extend(child_selected)

    return selected_categories








# Utility functions

def get_available_taxonomies(project_id: int) -> List[Dict[str, Any]]:
    """Get available taxonomies for a project."""
    try:
        response = api_client.get(f"/api/v1/taxonomy", params={"project_id": project_id})
        if response and isinstance(response, dict) and 'taxonomies' in response:
            return response['taxonomies']
        return []
    except Exception:
        return []


def get_extraction_jobs(project_id: int) -> List[Dict[str, Any]]:
    """Get extraction jobs for a project."""
    try:
        response = api_client.get(f"/api/v1/extraction/projects/{project_id}/jobs")
        return response if isinstance(response, list) else []
    except Exception:
        return []


def get_completed_extraction_jobs(project_id: int) -> List[Dict[str, Any]]:
    """Get completed extraction jobs for a project."""
    jobs = get_extraction_jobs(project_id)
    return [job for job in jobs if isinstance(job, dict) and job.get('status') == 'completed']


def start_extraction_job(project_id: int, taxonomy_id: int, depth: int,
                         confidence: float, batch_size: int, max_chunks: int,
                         initial_classifier: str, enable_validation: bool,
                         validation_classifier: Optional[str], extraction_type: str,
                         extraction_mode: str):
    """Start a new extraction job."""
    try:
        # Get selected categories from session state
        selected_categories = st.session_state.get(f'selected_categories_{taxonomy_id}', [])

        if not selected_categories:
            st.error("❌ No categories selected. Please select at least one category for extraction.")
            return

        with st.spinner("🚀 Starting extraction job..."):
            job_data = {
                "project_id": project_id,
                "taxonomy_id": taxonomy_id,
                "selected_categories": selected_categories,
                "parameters": {
                    "extraction_depth": depth,
                    "confidence_threshold": confidence,
                    "batch_size": batch_size,
                    "max_chunks": max_chunks
                },
                "initial_classifier": initial_classifier,
                "enable_validation_stage": enable_validation,
                "validation_classifier": validation_classifier,
                "extraction_type": extraction_type,
                "extraction_mode": extraction_mode
            }

            response = api_client.post("/api/v1/extraction/", data=job_data)

            if response and 'job_id' in response:
                job_id = response['job_id']
                st.success(f"✅ Extraction job started successfully! Job ID: {job_id}")
                
                # Monitor job synchronously (Wizard-style)
                success = monitor_job_synchronously(job_id, success_text="Extraction completed!")
                
                if success:
                    # Switch to Browse tab and refresh
                    st.session_state['extraction_active_tab'] = 1
                    st.session_state['selected_result_job'] = job_id
                    st.rerun()
            else:
                st.error("❌ Failed to start extraction job. Please try again.")

    except Exception as e:
        st.error(f"❌ Failed to start extraction job: {e}")


def restart_job(job_id: int):
    """Restart a failed or cancelled job."""
    try:
        with st.spinner(f"🔄 Restarting job {job_id}..."):
            response = api_client.post(f"/api/v1/extraction/{job_id}/restart")
            if response and response.get('status') == 'pending':
                st.success(f"✅ Job {job_id} restarted successfully.")
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"❌ Failed to restart job {job_id}.")
    except Exception as e:
        st.error(f"❌ Failed to restart job: {e}")


def cancel_job(job_id: int):
    """Cancel a pending or running job."""
    try:
        with st.spinner(f"❌ Cancelling job {job_id}..."):
            response = api_client.delete(f"/api/v1/extraction/{job_id}")
            if response and response.get('status') == 'cancelled':
                st.success(f"✅ Job {job_id} cancelled successfully.")
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"❌ Failed to cancel job {job_id}.")
    except Exception as e:
        st.error(f"❌ Failed to cancel job: {e}")


def get_confidence_color(confidence: float) -> str:
    """Get color for confidence score."""
    if confidence >= 0.8:
        return "#28a745"  # Green
    elif confidence >= 0.6:
        return "#ffc107"  # Yellow
    else:
        return "#dc3545"  # Red


def export_results_json(job_id: int, results_data: Dict[str, Any]):
    """Export current page results as JSON."""
    try:
        json_data = json.dumps(results_data, indent=2, default=str)
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"extraction_results_job_{job_id}.json",
            mime="application/json",
            key=f"download_json_{job_id}"
        )
    except Exception as e:
        st.error(f"Failed to prepare JSON export: {e}")


def export_results_csv(job_id: int, results_data: Dict[str, Any]):
    """Export current page results as CSV."""
    try:
        import pandas as pd

        results = results_data.get('results', [])
        if not results:
            st.error("No results to export.")
            return

        # Flatten the data for CSV
        csv_data = []
        for result in results:
            row = {
                'chunk_id': result.get('chunk_id', ''),
                'confidence_score': result.get('confidence_score', 0),
                'categories': ', '.join(result.get('categories_matched', [])),
                'chunk_text': result.get('chunk_text', ''),
            }
            csv_data.append(row)

        df = pd.DataFrame(csv_data)
        csv_string = df.to_csv(index=False)

        st.download_button(
            label="📊 Download CSV",
            data=csv_string,
            file_name=f"extraction_results_job_{job_id}.csv",
            mime="text/csv",
            key=f"download_csv_{job_id}"
        )
    except Exception as e:
        st.error(f"Failed to prepare CSV export: {e}")


def export_all_results(job_id: int):
    """Export all results for the job."""
    try:
        st.info("Full export functionality requires additional API endpoint implementation.")
    except Exception as e:
        st.error(f"Failed to export all results: {e}")


def select_all_categories(taxonomy_id: int, children: List[Dict], search_term: Optional[str] = None):
    """Select all visible categories for extraction using category IDs."""
    selected_ids = []

    def collect_category_ids(node: Dict[str, Any], path: str = ""):
        """Recursively collect category IDs that match search criteria."""
        node_name = node.get('name', 'Unknown')
        node_desc = node.get('description', '')

        # Check search filter
        if search_term and search_term not in node_name.lower() and search_term not in node_desc.lower():
            # If search term doesn't match this node, check children
            for i, child in enumerate(node.get('children', [])):
                child_path = f"{path}_{i}" if path else str(i)
                collect_category_ids(child, child_path)
            return

        # Add this category ID - only if it exists
        category_id = node.get('id')
        if category_id is not None:
            selected_ids.append(str(category_id))

        # Add children recursively
        for i, child in enumerate(node.get('children', [])):
            child_path = f"{path}_{i}" if path else str(i)
            collect_category_ids(child, child_path)

    # Collect all category IDs from children
    for i, child in enumerate(children):
        collect_category_ids(child, str(i))

    # Update session state
    st.session_state[f'selected_categories_{taxonomy_id}'] = selected_ids


def deselect_all_categories(taxonomy_id: int):
    """Deselect all categories for extraction."""
    st.session_state[f'selected_categories_{taxonomy_id}'] = []


def generate_qa_dataset(job_id: int, relationships: List[Dict[str, Any]]):
    """Generate Q&A dataset from entity relationships."""
    try:
        from ...features.taxonomy.qa_generator import generate_qa_from_relationships

        # Generate Q&A pairs
        qa_pairs = generate_qa_from_relationships(relationships, max_pairs_per_relationship=2)

        if not qa_pairs:
            st.warning("No Q&A pairs could be generated from the relationships.")
            return

        # Display statistics
        st.success(f"Generated {len(qa_pairs)} Q&A pairs!")

        # Show sample Q&A pairs
        st.markdown("### 📝 Sample Q&A Pairs")
        for i, pair in enumerate(qa_pairs[:5]):  # Show first 5
            with st.expander(f"Q&A Pair {i+1} (Confidence: {pair.confidence:.2f})", expanded=(i==0)):
                st.markdown(f"**Question:** {pair.question}")
                st.markdown(f"**Answer:** {pair.answer}")
                st.markdown(f"**Context:** {pair.context}")
                st.markdown(f"**Type:** {pair.qa_type.value.replace('_', ' ').title()}")

        # Export options
        st.markdown("### 💾 Export Q&A Dataset")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📄 Export JSONL", key=f'export_qa_jsonl_{job_id}'):
                from ...features.taxonomy.qa_generator import qa_generator
                dataset_content = qa_generator.export_qa_dataset(qa_pairs, "jsonl")

                st.download_button(
                    label="Download JSONL",
                    data=dataset_content,
                    file_name=f"qa_dataset_{job_id}.jsonl",
                    mime="application/jsonl",
                    key=f'download_qa_jsonl_{job_id}'
                )

        with col2:
            if st.button("📊 Export JSON", key=f'export_qa_json_{job_id}'):
                from ...features.taxonomy.qa_generator import qa_generator
                dataset_content = qa_generator.export_qa_dataset(qa_pairs, "json")

                st.download_button(
                    label="Download JSON",
                    data=dataset_content,
                    file_name=f"qa_dataset_{job_id}.json",
                    mime="application/json",
                    key=f'download_qa_json_{job_id}'
                )

        with col3:
            if st.button("📈 Export CSV", key=f'export_qa_csv_{job_id}'):
                from ...features.taxonomy.qa_generator import qa_generator
                dataset_content = qa_generator.export_qa_dataset(qa_pairs, "csv")

                st.download_button(
                    label="Download CSV",
                    data=dataset_content,
                    file_name=f"qa_dataset_{job_id}.csv",
                    mime="text/csv",
                    key=f'download_qa_csv_{job_id}'
                )

    except Exception as e:
        st.error(f"Error generating Q&A dataset: {str(e)}")
        st.exception(e)


def render_extraction_browser_management():
    """Render unified extraction browser and management interface."""
    from src.compileo.features.gui.services.api_client import api_client, APIError
    import time
    import json

    st.subheader("📋 Browse & Manage Extractions")

    # Project Selection
    projects = get_projects()
    if not projects:
        st.info("No projects available. Create your first project to start extractions.")
        return

    project_options = {f"{p['name']} (ID: {p['id']})": p['id'] for p in projects}
    selected_project_display = st.selectbox(
        "Select Project",
        options=list(project_options.keys()),
        help="Choose the project to view extraction jobs for",
        key="extraction_browser_project_select"
    )
    selected_project_id = project_options[selected_project_display]

    # Get all extraction jobs for this project
    jobs = get_extraction_jobs(selected_project_id)

    if not jobs:
        st.info("No extraction jobs found for this project.")
        st.markdown("""
        **To create an extraction job:**
        1. Go to the **🏃 Run Extraction** tab
        2. Select a taxonomy and categories
        3. Configure extraction parameters
        4. Click **Start Extraction**
        """)
        return

    # Search and filter controls
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        search_query = st.text_input("🔍 Search jobs", placeholder="Enter job ID or status...", key="extraction_search")

    with col2:
        status_filter = st.selectbox("Filter by Status", ["All", "Completed", "Running", "Failed", "Pending", "Cancelled"], key="extraction_status_filter")

    with col3:
        sort_by = st.selectbox("Sort by", ["Newest", "Oldest", "Status"], key="extraction_sort")

    # Filter jobs based on search and status
    filtered_jobs = jobs
    if search_query:
        filtered_jobs = [j for j in jobs if search_query.lower() in str(j.get('job_id', '')).lower() or search_query.lower() in j.get('status', '').lower()]

    if status_filter != "All":
        filtered_jobs = [j for j in filtered_jobs if j.get('status', '').lower() == status_filter.lower()]

    # Sort jobs
    if sort_by == "Newest":
        filtered_jobs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    elif sort_by == "Oldest":
        filtered_jobs.sort(key=lambda x: x.get('created_at', ''))
    elif sort_by == "Status":
        status_order = {'running': 0, 'pending': 1, 'completed': 2, 'failed': 3, 'cancelled': 4}
        filtered_jobs.sort(key=lambda x: status_order.get(x.get('status', 'unknown'), 5))

    # Bulk selection controls
    render_bulk_selection_controls_extractions(filtered_jobs)

    # Display jobs
    for job in filtered_jobs:
        render_extraction_job_card(job)

    # Export functionality
    st.divider()
    st.subheader("📤 Export Extraction Jobs")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📄 Export All Jobs as JSON", width='stretch'):
            export_extraction_jobs_json(filtered_jobs)

    with col2:
        if st.button("📊 Export Summary as CSV", width='stretch'):
            export_extraction_jobs_csv(filtered_jobs)


def render_bulk_selection_controls_extractions(jobs: List[Dict[str, Any]]):
    """Render bulk selection controls for extraction jobs."""
    if not jobs:
        return

    # Initialize bulk selection state
    if 'extraction_bulk_selected' not in st.session_state:
        st.session_state.extraction_bulk_selected = []

    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        if st.button("✅ Select All", key="extraction_select_all", help="Select all visible jobs"):
            st.session_state.extraction_bulk_selected = [str(j['job_id']) for j in jobs]
            st.rerun()

    with col2:
        if st.button("❌ Deselect All", key="extraction_deselect_all", help="Deselect all jobs"):
            st.session_state.extraction_bulk_selected = []
            st.rerun()

    with col3:
        selected_count = len(st.session_state.extraction_bulk_selected)
        if selected_count > 0:
            if st.button(f"🗑️ Delete Selected ({selected_count})", key="extraction_bulk_delete", type="secondary"):
                st.session_state.extraction_bulk_delete_confirm = True
                st.rerun()

    with col4:
        if selected_count > 0:
            st.success(f"{selected_count} job{'s' if selected_count > 1 else ''} selected")

    # Bulk delete confirmation
    if st.session_state.get('extraction_bulk_delete_confirm', False):
        render_bulk_delete_confirmation_extractions()


def render_bulk_delete_confirmation_extractions():
    """Render bulk delete confirmation dialog for extraction jobs."""
    selected_ids = st.session_state.get('extraction_bulk_selected', [])
    if not selected_ids:
        return

    st.warning(f"Are you sure you want to delete {len(selected_ids)} extraction job{'s' if len(selected_ids) > 1 else ''}?")
    st.error("This action cannot be undone. All extraction results and associated files will be permanently removed.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("❌ Yes, Delete All", type="primary", width='stretch'):
            try:
                with st.spinner(f"Deleting {len(selected_ids)} extraction jobs..."):
                    for job_id in selected_ids:
                        api_client.delete(f"/api/v1/extraction/{job_id}/delete")

                st.success(f"Successfully deleted {len(selected_ids)} extraction jobs!")
                st.session_state.extraction_bulk_selected = []
                del st.session_state.extraction_bulk_delete_confirm
                st.cache_data.clear()
                st.rerun()

            except APIError as e:
                st.error(f"Failed to delete extraction jobs: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

    with col2:
        if st.button("⬅️ Cancel", width='stretch'):
            del st.session_state.extraction_bulk_delete_confirm
            st.rerun()


def render_extraction_job_card(job: Dict[str, Any]):
    """Render an extraction job card with management actions."""
    with st.container():
        # Checkbox for bulk selection
        col_checkbox, col_content = st.columns([0.1, 0.9])

        with col_checkbox:
            job_id_str = str(job['job_id'])
            is_selected = st.checkbox(
                f"Select job {job_id_str}",
                key=f"extraction_select_{job_id_str}",
                value=job_id_str in st.session_state.get('extraction_bulk_selected', []),
                label_visibility="collapsed"
            )
            if is_selected and job_id_str not in st.session_state.extraction_bulk_selected:
                st.session_state.extraction_bulk_selected.append(job_id_str)
            elif not is_selected and job_id_str in st.session_state.extraction_bulk_selected:
                st.session_state.extraction_bulk_selected.remove(job_id_str)

        with col_content:
            # Job header
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.markdown(f"**Job {job['job_id']}**")
                created_date = job['created_at'].split('T')[0] if 'T' in job['created_at'] else job['created_at']
                st.caption(f"Created: {created_date}")

            with col2:
                status = job['status'].capitalize()
                status_color = {
                    'Completed': 'green',
                    'Running': 'blue',
                    'Failed': 'red',
                    'Pending': 'orange',
                    'Cancelled': 'gray'
                }.get(status, 'black')
                st.markdown(f"<span style='color: {status_color}; font-weight: bold;'>{status}</span>", unsafe_allow_html=True)

            with col3:
                status_lower = job['status'].lower()
                if status_lower == 'running':
                    st.info("🔄 Running...")
                elif status_lower == 'pending':
                    st.warning("⏳ Pending...")
                elif status_lower == 'completed':
                    st.success("✅ Done")
                elif status_lower == 'failed':
                    st.error("❌ Failed")
                elif status_lower == 'cancelled':
                    st.text("🚫 Cancelled")
                else:
                    st.text(f"Status: {status}")

            # Management actions
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.button("👁️ View Details", key=f"view_extraction_{job['job_id']}", width='stretch'):
                    # Switch to Browse & Manage tab and set job
                    st.session_state['extraction_active_tab'] = 1  # Browse & Manage tab index
                    st.session_state['selected_result_job'] = job['job_id']
                    st.rerun()

            with col2:
                if st.button("📤 Export JSON", key=f"export_extraction_{job['job_id']}", width='stretch'):
                    try:
                        response = api_client.get(f"/api/v1/extraction/{job['job_id']}/results?page=1&page_size=100")
                        if response and 'results' in response:
                            import json
                            json_str = json.dumps(response, indent=2, default=str)
                            st.download_button(
                                label="📥 Download",
                                data=json_str,
                                file_name=f"extraction_results_job_{job['job_id']}.json",
                                mime="application/json",
                                width='stretch'
                            )
                    except APIError as e:
                        st.error(f"Failed to export job results: {e}")

            with col3:
                if job['status'] in ['failed', 'cancelled']:
                    if st.button("🔄 Restart", key=f"restart_extraction_{job['job_id']}", width='stretch'):
                        restart_job(job['job_id'])
                elif job['status'] in ['pending', 'running']:
                    if st.button("❌ Cancel", key=f"cancel_extraction_{job['job_id']}", width='stretch'):
                        cancel_job(job['job_id'])

            with col4:
                if st.button("🗑️ Delete", key=f"delete_extraction_{job['job_id']}", width='stretch', type="secondary"):
                    st.session_state[f'extraction_delete_confirm_{job["job_id"]}'] = True
                    st.rerun()

            # Individual delete confirmation
            if st.session_state.get(f'extraction_delete_confirm_{job["job_id"]}', False):
                render_individual_delete_confirmation_extractions(job)

        st.divider()


def render_individual_delete_confirmation_extractions(job: Dict[str, Any]):
    """Render individual delete confirmation for an extraction job."""
    job_id = job['job_id']

    st.warning(f"Are you sure you want to delete extraction job {job_id}?")
    st.error("This action cannot be undone. All extraction results and associated files will be permanently removed.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button(f"❌ Yes, Delete Job {job_id}", type="primary", width='stretch'):
            try:
                with st.spinner(f"Deleting extraction job {job_id}..."):
                    api_client.delete(f"/api/v1/extraction/{job_id}/delete")

                st.success(f"Extraction job {job_id} deleted successfully!")
                del st.session_state[f'extraction_delete_confirm_{job_id}']
                st.cache_data.clear()
                st.rerun()

            except APIError as e:
                st.error(f"Failed to delete extraction job: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

    with col2:
        if st.button("⬅️ Cancel", width='stretch'):
            del st.session_state[f'extraction_delete_confirm_{job_id}']
            st.rerun()


def export_extraction_jobs_json(jobs: List[Dict[str, Any]]):
    """Export extraction jobs as JSON."""
    try:
        import json
        json_str = json.dumps(jobs, indent=2, default=str)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"extraction_jobs_{int(time.time())}.json",
            mime="application/json"
        )
    except Exception as e:
        st.error(f"Failed to export jobs as JSON: {e}")


def export_extraction_jobs_csv(jobs: List[Dict[str, Any]]):
    """Export extraction jobs summary as CSV."""
    try:
        import pandas as pd

        # Create summary data
        csv_data = []
        for job in jobs:
            row = {
                'job_id': job.get('job_id', ''),
                'status': job.get('status', ''),
                'created_at': job.get('created_at', ''),
                'progress_percentage': job.get('progress_percentage', 0),
                'project_id': job.get('project_id', ''),
                'taxonomy_id': job.get('parameters', {}).get('taxonomy_id', ''),
                'extraction_type': job.get('parameters', {}).get('extraction_type', ''),
                'extraction_mode': job.get('parameters', {}).get('extraction_mode', '')
            }
            csv_data.append(row)

        df = pd.DataFrame(csv_data)
        csv_string = df.to_csv(index=False)

        st.download_button(
            label="📊 Download CSV",
            data=csv_string,
            file_name=f"extraction_jobs_summary_{int(time.time())}.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Failed to export jobs as CSV: {e}")