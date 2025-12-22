"""
Taxonomy Tree Module.
Handles taxonomy tree operations, extraction, and complex tree manipulations.
"""

import streamlit as st
import time
from typing import Dict, Any, List, Optional
from src.compileo.features.gui.services.api_client import api_client, APIError
from src.compileo.features.gui.components.taxonomy_tree_selector import TaxonomyTreeSelector


def render_classification_extraction():
    """Render the Classification & Extraction controls."""
    st.subheader("🔍 Classification & Extraction")

    # Get available projects
    from src.compileo.features.gui.utils.taxonomy_utils import get_available_projects
    projects = get_available_projects()
    if not projects:
        st.warning("No projects available. Create a project first.")
        return

    # Project selection
    project_options = {f"{p['name']} (ID: {p['id']})": p for p in projects}
    selected_project_key = st.selectbox(
        "Select Project",
        options=list(project_options.keys()),
        key="classification_project_select",
        help="Choose the project for classification and extraction"
    )

    if selected_project_key:
        selected_project = project_options[selected_project_key]
        project_id = selected_project["id"]

        st.divider()

        # Extraction Controls
        st.markdown("### 📤 Extraction Controls")

        # Get available taxonomies for the project
        taxonomies = []
        try:
            response = api_client.get(f"/api/v1/taxonomy?project_id={project_id}")
            taxonomies = response.get("taxonomies", [])
        except APIError:
            taxonomies = []

        if taxonomies:
            taxonomy_options = {f"{t['name']} ({t['categories_count']} categories)": t for t in taxonomies}
            selected_taxonomy_key = st.selectbox(
                "Select Taxonomy",
                options=list(taxonomy_options.keys()),
                help="Choose taxonomy for selective extraction - you'll be able to select specific categories below"
            )

            if selected_taxonomy_key:
                selected_taxonomy_data = taxonomy_options[selected_taxonomy_key]

                # Load full taxonomy data for the tree selector
                try:
                    taxonomy_response = api_client.get(f"/api/v1/taxonomy/{selected_taxonomy_data['id']}")
                    taxonomy_tree_data = taxonomy_response.get('taxonomy', {})

                    if taxonomy_tree_data:
                        # Initialize TaxonomyTreeSelector
                        selector_key = f"extraction_selector_{selected_taxonomy_data['id']}"
                        tree_selector = TaxonomyTreeSelector(taxonomy_tree_data, selector_key)

                        # Render the taxonomy tree selector
                        st.markdown("#### 🎯 Category Selection")
                        st.markdown("Select specific categories from the taxonomy tree for extraction:")
                        st.markdown("*Check/uncheck categories to include them in extraction*")

                        selected_categories = tree_selector.render()

                        # Show selection summary with more details
                        if selected_categories:
                            stats = tree_selector.get_selection_stats()
                            st.success(f"✅ {stats['selected']} of {stats['total']} categories selected ({stats['percentage']:.1f}%)")
                        else:
                            st.info("ℹ️ No categories selected - all categories will be used for extraction")

                        # Extraction Parameters Section
                        st.markdown("---")
                        st.markdown("#### ⚙️ Extraction Parameters")
                        # Two-stage classifier selection
                        st.markdown("**🤖 Classifier Selection:**")
                        initial_classifier = st.selectbox(
                            "Select Initial Classifier",
                            options=["grok", "gemini", "ollama"],
                            index=0,
                            help="Choose the AI classifier for the initial classification stage"
                        )

                        enable_validation_stage = st.checkbox(
                            "Enable Validation Stage",
                            value=False,
                            help="Enable a second classification stage to validate the initial results"
                        )

                        validation_classifier = None
                        if enable_validation_stage:
                            validation_classifier = st.selectbox(
                                "Select Validation Classifier",
                                options=["grok", "gemini", "ollama"],
                                index=0,
                                help="Choose the AI classifier for the validation stage"
                            )

                        col1, col2 = st.columns(2)

                        with col1:
                            extraction_depth = st.slider(
                                "Extraction Depth",
                                min_value=1,
                                max_value=5,
                                value=3,
                                help="Maximum depth to traverse in taxonomy hierarchy during extraction"
                            )

                        with col2:
                            confidence_threshold = st.slider(
                                "Confidence Threshold",
                                min_value=0.0,
                                max_value=1.0,
                                value=0.7,
                                step=0.05,
                                help="Minimum confidence score required for extraction results"
                            )

                        # Advanced parameters in expandable section
                        with st.expander("🔧 Advanced Parameters", expanded=False):
                            col4, col5 = st.columns(2)

                            with col4:
                                max_chunks = st.number_input(
                                    "Max Chunks to Process",
                                    min_value=1,
                                    max_value=10000,
                                    value=1000,
                                    step=100,
                                    help="Maximum number of document chunks to process (0 = all)"
                                )

                            with col5:
                                batch_size = st.slider(
                                    "Processing Batch Size",
                                    min_value=10,
                                    max_value=200,
                                    value=50,
                                    help="Number of chunks to process in each batch"
                                )

                        # Extraction Actions
                        st.markdown("---")
                        col_action1, col_action2, col_action3 = st.columns([2, 2, 1])

                        with col_action1:
                            if st.button("📤 Run Selective Extraction", type="primary", width='stretch'):
                                if not selected_categories and len(selected_categories) == 0:
                                    st.warning("No categories selected. Please select at least one category or leave all unselected for full extraction.")
                                else:
                                    # Get selected category names for API
                                    selected_category_names = tree_selector.get_selected_category_names()

                                    run_selective_extraction(
                                        project_id=project_id,
                                        taxonomy_id=selected_taxonomy_data['id'],
                                        selected_categories=list(selected_categories),
                                        extraction_depth=extraction_depth,
                                        confidence_threshold=confidence_threshold,
                                        initial_classifier=initial_classifier,
                                        enable_validation_stage=enable_validation_stage,
                                        validation_classifier=validation_classifier,
                                        max_chunks=max_chunks if max_chunks > 0 else None,
                                        batch_size=batch_size
                                    )

                        with col_action2:
                            if st.button("📊 Preview Selection", width='stretch'):
                                preview_category_selection(tree_selector, taxonomy_tree_data)

                        with col_action3:
                            if st.button("🔄 Reset", help="Clear all selections and parameters"):
                                tree_selector.clear_selection()
                                st.rerun()

                    else:
                        st.error("Failed to load taxonomy structure for selection.")

                except APIError as e:
                    st.error(f"Failed to load taxonomy details: {e}")
        else:
            st.info("No taxonomies available for this project. Create a taxonomy first.")


def run_selective_extraction(
    project_id: int,
    taxonomy_id: int,
    selected_categories: List[str],
    extraction_depth: int,
    confidence_threshold: float,
    initial_classifier: str,
    enable_validation_stage: bool,
    validation_classifier: Optional[str],
    max_chunks: Optional[int] = None,
    batch_size: int = 50
):
    """Run selective taxonomy-based extraction with advanced parameters."""
    try:
        # Prepare request data for selective extraction API
        data = {
            "taxonomy_id": taxonomy_id,
            "selected_categories": selected_categories,
            "parameters": {
                "extraction_depth": extraction_depth,
                "confidence_threshold": confidence_threshold,
                "max_chunks": max_chunks,
                "batch_size": batch_size
            },
            "initial_classifier": initial_classifier,
            "enable_validation_stage": enable_validation_stage,
            "validation_classifier": validation_classifier
        }

        with st.spinner("🚀 Starting selective extraction..."):
            # Call the selective extraction API
            response = api_client.post("/api/v1/extraction/", data=data)

            if response and "job_id" in response:
                job_id = response["job_id"]
                st.success(f"✅ Selective extraction job started! Job ID: {job_id}")

                # Store job info in session state for persistence
                if "extraction_jobs" not in st.session_state:
                    st.session_state.extraction_jobs = {}
                st.session_state.extraction_jobs[job_id] = {
                    "taxonomy_id": taxonomy_id,
                    "selected_categories": selected_categories,
                    "parameters": data["parameters"],
                    "start_time": time.time()
                }

                # Poll for completion with enhanced monitoring
                poll_selective_extraction_status(job_id)
            else:
                st.error("❌ Failed to start selective extraction job")
                if response:
                    st.error(f"API Response: {response}")

    except APIError as e:
        st.error(f"❌ API Error during extraction: {e}")
    except Exception as e:
        st.error(f"❌ Unexpected error during extraction: {e}")


def preview_category_selection(tree_selector: TaxonomyTreeSelector, taxonomy_data: Dict[str, Any]):
    """Preview the selected categories and their hierarchy."""
    selected_categories = tree_selector.selected_categories

    if not selected_categories:
        st.info("No categories selected for preview.")
        return

    st.markdown("### 📋 Selection Preview")

    # Build a hierarchical view of selected categories
    selected_tree = {}

    def build_selected_tree(node: Dict[str, Any], path: str = "", level: int = 0):
        node_name = node.get('name', 'Unknown')
        node_id = f"{path}_{node_name}" if path else node_name

        if node_id in selected_categories:
            if level not in selected_tree:
                selected_tree[level] = []
            selected_tree[level].append({
                'name': node_name,
                'description': node.get('description', ''),
                'confidence': node.get('confidence_threshold', 0.0),
                'path': node_id
            })

        for i, child in enumerate(node.get('children', [])):
            child_path = f"{node_id}_{i}"
            build_selected_tree(child, child_path, level + 1)

    if taxonomy_data:
        build_selected_tree(taxonomy_data)

    # Display selected categories by level
    for level in sorted(selected_tree.keys()):
        level_name = ["Root", "Level 1", "Level 2", "Level 3", "Level 4", "Level 5"][level] if level < 6 else f"Level {level}"

        with st.expander(f"📂 {level_name} ({len(selected_tree[level])} categories)", expanded=level <= 1):
            for category in selected_tree[level]:
                confidence_color = "🟢" if category['confidence'] >= 0.8 else "🟡" if category['confidence'] >= 0.6 else "🔴"
                st.markdown(f"- **{category['name']}** {confidence_color} (Confidence: {category['confidence']:.2f})")
                if category['description']:
                    st.caption(category['description'])


def poll_selective_extraction_status(job_id: int):
    """Poll selective extraction job status with enhanced monitoring and results display."""
    max_attempts = 120  # 10 minutes max for selective extraction
    attempt = 0

    # Create containers for dynamic updates
    progress_container = st.empty()
    status_container = st.empty()
    results_container = st.empty()

    with progress_container.container():
        progress_bar = st.progress(0)
        status_text = st.empty()

    while attempt < max_attempts:
        try:
            response = api_client.get(f"/api/v1/extraction/{job_id}")

            if response:
                status = response.get("status", "unknown")
                progress_info = response.get("progress", {})

                # Update progress bar
                if status == "running" and progress_info:
                    progress_percent = progress_info.get("progress_percentage", 0)
                    progress_bar.progress(min(progress_percent / 100, 1.0))

                    # Show detailed progress
                    processed_chunks = progress_info.get("processed_chunks", 0)
                    total_chunks = progress_info.get("total_chunks", 0)
                    current_category = progress_info.get("current_category", "Unknown")

                    status_text.info(f"🔄 Processing: {processed_chunks}/{total_chunks} chunks | Current: {current_category}")

                elif status == "completed":
                    progress_bar.progress(1.0)
                    status_text.success("✅ Selective extraction completed successfully!")

                    # Show completion summary
                    with results_container.container():
                        display_selective_extraction_results(job_id, response)
                    break

                elif status == "failed":
                    progress_bar.empty()
                    status_text.error("❌ Selective extraction failed")
                    error_msg = response.get("error_message")
                    if error_msg:
                        st.error(f"Error: {error_msg}")
                    break

                elif status in ["pending", "queued"]:
                    status_text.info("⏳ Job queued, waiting to start...")
                    progress_bar.progress(0.1)

                else:
                    status_text.info(f"📋 Job status: {status}")

            else:
                status_text.warning("Unable to check job status")

        except APIError as e:
            status_text.error(f"Error checking job status: {e}")
            break

        time.sleep(5)  # Wait 5 seconds before checking again
        attempt += 1

    if attempt >= max_attempts:
        progress_bar.empty()
        status_text.warning("⏰ Job is taking longer than expected. Check back later or contact support.")


def display_selective_extraction_results(job_id: int, job_response: Dict[str, Any]):
    """Display selective extraction results organized by selected categories."""
    try:
        # Get detailed results
        response = api_client.get(f"/api/v1/extraction/{job_id}/results?page=1&page_size=100")

        if response and "results" in response:
            results = response.get("results", [])
            total_results = response.get("total_results", 0)

            st.subheader(f"📊 Selective Extraction Results ({total_results} total)")

            if results:
                # Get job info for category context
                job_info = st.session_state.get("extraction_jobs", {}).get(job_id, {})
                selected_categories = job_info.get("selected_categories", [])

                # Organize results by category
                results_by_category = {}
                for result in results:
                    categories = result.get("categories_matched", [])
                    for category in categories:
                        if category not in results_by_category:
                            results_by_category[category] = []
                        results_by_category[category].append(result)

                # Display results organized by category
                if results_by_category:
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Results", total_results)
                    with col2:
                        st.metric("Categories Found", len(results_by_category))
                    with col3:
                        avg_confidence = sum(r.get("confidence", 0) for r in results) / len(results) if results else 0
                        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                    with col4:
                        st.metric("Selected Categories", len(selected_categories) if selected_categories else "All")

                    # Results by category
                    for category, category_results in results_by_category.items():
                        with st.expander(f"📂 {category} ({len(category_results)} results)", expanded=len(results_by_category) <= 5):
                            # Category results table
                            result_data = []
                            for result in category_results[:10]:  # Show first 10 per category
                                result_data.append({
                                    "Chunk ID": result.get("chunk_id", ""),
                                    "Confidence": f"{result.get('confidence', 0):.2f}",
                                    "Text Preview": result.get("chunk_text", "")[:150] + "..." if len(result.get("chunk_text", "")) > 150 else result.get("chunk_text", ""),
                                    "Document": result.get("document_name", "Unknown")
                                })

                            if result_data:
                                st.dataframe(result_data, width='stretch')

                            if len(category_results) > 10:
                                st.info(f"Showing first 10 results for {category}. Total: {len(category_results)}")

                    # Export options
                    st.markdown("---")
                    col_export1, col_export2 = st.columns(2)

                    with col_export1:
                        if st.button("📄 Export All Results (JSON)", width='stretch'):
                            import json
                            export_data = {
                                "job_id": job_id,
                                "extraction_type": "selective",
                                "selected_categories": selected_categories,
                                "total_results": total_results,
                                "results_by_category": results_by_category,
                                "exported_at": time.time()
                            }
                            st.download_button(
                                label="📥 Download",
                                data=json.dumps(export_data, indent=2),
                                file_name=f"selective_extraction_{job_id}_{int(time.time())}.json",
                                mime="application/json",
                                width='stretch'
                            )

                    with col_export2:
                        if st.button("📊 Export Summary (CSV)", width='stretch'):
                            # Create summary CSV
                            summary_data = []
                            for category, cat_results in results_by_category.items():
                                summary_data.append({
                                    "Category": category,
                                    "Result Count": len(cat_results),
                                    "Avg Confidence": sum(r.get("confidence", 0) for r in cat_results) / len(cat_results) if cat_results else 0
                                })

                            import csv
                            import io
                            output = io.StringIO()
                            writer = csv.DictWriter(output, fieldnames=["Category", "Result Count", "Avg Confidence"])
                            writer.writeheader()
                            writer.writerows(summary_data)

                            st.download_button(
                                label="📥 Download Summary",
                                data=output.getvalue(),
                                file_name=f"extraction_summary_{job_id}_{int(time.time())}.csv",
                                mime="text/csv",
                                width='stretch'
                            )

                else:
                    st.info("No categorized results found.")
            else:
                st.info("No extraction results found.")
        else:
            st.error("Failed to load extraction results")

    except APIError as e:
        st.error(f"Error loading results: {e}")


def run_extraction(project_id: int, taxonomy_name: str, depth: int, confidence: float, skip_fine_classification: bool = False, selected_categories: Optional[List[str]] = None):
    """Run taxonomy-based extraction (legacy function for backwards compatibility)."""
    try:
        # Get taxonomy ID by name
        taxonomies = []
        try:
            response = api_client.get(f"/api/v1/taxonomy?project_id={project_id}")
            taxonomies = response.get("taxonomies", [])
        except APIError:
            st.error("Failed to load taxonomies")
            return

        # Find the taxonomy by name
        taxonomy_id = None
        for tax in taxonomies:
            if tax.get("name") == taxonomy_name:
                taxonomy_id = tax.get("id")
                break

        if not taxonomy_id:
            st.error(f"Taxonomy '{taxonomy_name}' not found")
            return

        # Use selective extraction with empty selection (all categories)
        run_selective_extraction(
            project_id=project_id,
            taxonomy_id=taxonomy_id,
            selected_categories=[],  # Empty list means all categories
            extraction_depth=depth,
            confidence_threshold=confidence,
            initial_classifier="gemini",
            enable_validation_stage=False,
            validation_classifier=None
        )

    except Exception as e:
        st.error(f"Extraction failed: {e}")


def poll_extraction_status(job_id: int):
    """Poll extraction job status and display results."""
    import time

    max_attempts = 60  # 5 minutes max
    attempt = 0

    progress_bar = st.progress(0)
    status_text = st.empty()

    while attempt < max_attempts:
        try:
            response = api_client.get(f"/api/v1/extraction/{job_id}")

            if response:
                status = response.get("status", "unknown")

                if status == "completed":
                    progress_bar.progress(100)
                    status_text.success("✅ Extraction completed successfully!")

                    # Show results summary
                    progress_info = response.get("progress", {})
                    if progress_info:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Results", progress_info.get("total_results", 0))
                        with col2:
                            st.metric("Avg Confidence", f"{progress_info.get('avg_confidence', 0):.2f}")
                        with col3:
                            st.metric("Categories Found", len(progress_info.get("categories_found", [])))

                    # Option to view detailed results
                    if st.button("📊 View Detailed Results"):
                        show_extraction_results(job_id)

                    break

                elif status == "failed":
                    progress_bar.empty()
                    status_text.error("❌ Extraction failed")
                    error_msg = response.get("error_message")
                    if error_msg:
                        st.error(f"Error: {error_msg}")
                    break

                elif status == "processing":
                    # Show processing status
                    status_text.info("🔄 Processing extraction job...")
                    progress_bar.progress(50)  # Show some progress

                else:
                    status_text.info(f"📋 Job status: {status}")

            else:
                status_text.warning("Unable to check job status")

        except APIError as e:
            status_text.error(f"Error checking job status: {e}")
            break

        time.sleep(5)  # Wait 5 seconds before checking again
        attempt += 1

    if attempt >= max_attempts:
        progress_bar.empty()
        status_text.warning("⏰ Job is taking longer than expected. Check back later.")


def show_extraction_results(job_id: int):
    """Display detailed extraction results (legacy function)."""
    try:
        # Get paginated results
        response = api_client.get(f"/api/v1/extraction/{job_id}/results?page=1&page_size=50")

        if response and "results" in response:
            results = response.get("results", [])
            total_results = response.get("total_results", 0)

            st.subheader(f"📊 Extraction Results ({total_results} total)")

            if results:
                # Display results in a table
                result_data = []
                for result in results[:20]:  # Show first 20 results
                    result_data.append({
                        "Chunk ID": result.get("chunk_id", ""),
                        "Confidence": f"{result.get('confidence', 0):.2f}",
                        "Categories": ", ".join(result.get("categories_matched", [])),
                        "Text Preview": result.get("chunk_text", "")[:100] + "..." if len(result.get("chunk_text", "")) > 100 else result.get("chunk_text", "")
                    })

                st.dataframe(result_data, width='stretch')

                if total_results > 20:
                    st.info(f"Showing first 20 results. Total: {total_results}")
            else:
                st.info("No extraction results found.")
        else:
            st.error("Failed to load extraction results")

    except APIError as e:
        st.error(f"Error loading results: {e}")