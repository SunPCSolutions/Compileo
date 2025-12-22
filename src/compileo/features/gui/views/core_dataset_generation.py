"""
Core Dataset Generation view for Compileo GUI.
Handles advanced dataset generation with comprehensive controls.
"""

import streamlit as st
from typing import Dict, Any, List
import time

from ..services.api_client import api_client, APIError
from ..services.document_api_service import get_projects
from ..state.session_state import session_state


def render_core_dataset_generation():
    """Render the Core Dataset Generation page with tabs for generation and management."""
    st.markdown('<h1 class="section-header">🔧 Dataset Generation</h1>', unsafe_allow_html=True)

    st.markdown("""
    Generate and manage datasets from processed document chunks.
    Create training data for AI models with comprehensive controls and management tools.
    """)

    # Create tabs for different views
    tab1, tab2 = st.tabs([
        "🚀 Generate Dataset",
        "📋 Browse & Manage Datasets"
    ])

    with tab1:
        render_generate_dataset()

    with tab2:
        render_dataset_browser_management()


def render_generate_dataset():
    """Render the Generate Dataset tab with compact, extraction-style UI."""
    # Project and Document Selection - Side by Side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📁 Project**")
        projects = get_projects()
        if projects:
            project_options = {f"{p['name']} (ID: {p['id']})": p['id'] for p in projects}
            selected_project_display = st.selectbox(
                "Select Project",
                options=list(project_options.keys()),
                key="core_gen_project_select",
                help="Choose the project containing documents for dataset generation"
            )
            selected_project_id = project_options[selected_project_display]
        else:
            st.error("⚠️ No projects available. Please create a project first.")
            return

    with col2:
        st.markdown("**📄 Documents**")
        try:
            docs_response = api_client.get(f"/api/v1/documents?project_id={selected_project_id}")
            documents = docs_response.get('documents', [])

            if documents:
                doc_options = {f"{d['file_name']} (ID: {d['id']})": d['id'] for d in documents}
                selected_documents = st.multiselect(
                    "Select Documents",
                    options=list(doc_options.keys()),
                    help="Select one or more documents to generate dataset entries from. The system will use existing text chunks from these documents."
                )
                selected_doc_ids = [doc_options[doc] for doc in selected_documents] if selected_documents else []
            else:
                st.warning("⚠️ No documents found")
                selected_doc_ids = []
        except Exception as e:
            st.error(f"❌ Failed to load documents: {e}")
            selected_doc_ids = []

    st.divider()

    # Dataset Generation Configuration - Compact Layout
    col3, col4, col5 = st.columns([2, 2, 2])

    with col3:
        st.markdown("**🤖 Generation Model**")
        classification_model = st.selectbox(
            "Model",
            options=["grok", "gemini", "ollama", "openai"],
            index=0,
            help="AI model used to generate dataset entries. Grok provides detailed reasoning, Gemini is fast and accurate, Ollama runs locally."
        )

    with col4:
        st.markdown("**📝 Generation Mode**")
        generation_mode = st.selectbox(
            "Mode",
            options=["instruction following", "question and answer", "question", "answer", "summarization"],
            index=0,
            help="Type of dataset to generate: Q&A pairs, individual questions/answers, summaries, or instruction-response pairs for AI training."
        )

    with col5:
        st.markdown("**📄 Output Format**")
        # Get available formats dynamically including plugins
        try:
            formats_response = api_client.get("/api/v1/plugins/dataset-formats")
            available_formats = formats_response.get("formats", ["jsonl", "parquet"])
        except Exception as e:
            st.warning(f"Could not load plugin formats: {e}. Using default formats.")
            available_formats = ["jsonl", "parquet"]

        format_type = st.selectbox(
            "Format",
            options=available_formats,
            index=0,
            help="Output file format: JSONL for line-delimited JSON (common for AI training), Parquet for compressed columnar storage, or plugin formats like 'anki' for specialized outputs."
        )

    st.divider()

    # Processing Configuration - New Section
    col_batch1, col_batch2 = st.columns([2, 2])

    with col_batch1:
        st.markdown("**⚡ Batch Size**")
        batch_size = st.slider(
            "Chunks per Batch",
            min_value=0,
            max_value=200,
            value=50,
            step=10,
            help="Number of chunks to process per batch. Smaller batches use less memory but may take longer. Set to 0 to process all chunks at once."
        )

        # Display formatted value
        if batch_size == 0:
            st.caption("📊 All chunks at once")
        else:
            st.caption(f"📊 {batch_size} chunks per batch")

    with col_batch2:
        st.markdown("**🔄 Workers per Batch**")
        concurrency = st.slider(
            "Concurrent Workers",
            min_value=1,
            max_value=5,
            value=2,
            help="Number of parallel workers within each batch. Higher values process faster but use more API calls.",
            label_visibility="collapsed"
        )

    st.divider()

    # Dataset Size and Quality - Compact Layout
    st.markdown("### 📊 Dataset Configuration")
    col6, col7, col8 = st.columns(3)

    with col6:
        st.markdown("**📊 Dataset Size**")
        datasets_per_chunk = st.number_input(
            "Entries per Chunk",
            min_value=1,
            value=3,
            step=1,
            help="Number of dataset entries to generate from each document chunk. Higher values create more data but increase processing time."
        )

    with col7:
        st.markdown("**🎯 Quality**")
        analyze_quality = st.checkbox(
            "Quality Filter",
            value=True,
            help="Enable quality analysis to filter out low-quality generated entries"
        )

        if analyze_quality:
            quality_threshold = st.slider(
                "Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Minimum quality score required (0.0-1.0). Higher values ensure better quality but may reduce dataset size."
            )
        else:
            quality_threshold = 0.7

    with col8:
        st.markdown("**📊 Data Source**")
        data_source = st.selectbox(
            "Data Source for dataset generation",
            options=["Chunks Only", "Taxonomy", "Extract"],
            index=0,
            help="Choose the data source for dataset generation: Chunks only (use raw chunks), Taxonomy (use chunks with taxonomy filtering), Extract (use only extracted entities for taxonomy-specific datasets)"
        )

        # Debug: Show current data source
        st.caption(f"Selected: {data_source}")

        taxonomy_name = None
        extraction_file_id = None

        if data_source == "Taxonomy":
            st.markdown("**🏷️ Taxonomy Selection**")
            taxonomies = get_project_taxonomies(selected_project_id)
            if taxonomies:
                taxonomy_options = {f"{t['name']}": t['name'] for t in taxonomies}
                taxonomy_name = st.selectbox(
                    "Select Taxonomy",
                    options=list(taxonomy_options.keys()),
                    help="Select a taxonomy to filter content by category. Only chunks matching selected taxonomy categories will be used."
                )
                st.caption(f"Selected taxonomy: {taxonomy_name}")
            else:
                st.warning("⚠️ No taxonomies available for this project")
                taxonomy_name = None

        elif data_source == "Extract":
            st.markdown("**📄 Extraction File Selection**")
            # Get available extraction files for the project
            extraction_files = get_project_extraction_files(selected_project_id)
            if extraction_files:
                st.info(f"Found {len(extraction_files)} completed extraction jobs")
                extraction_options = {f"{ef['display_name']}": ef['id'] for ef in extraction_files}
                selected_extraction_display = st.selectbox(
                    "Select Extraction File",
                    options=list(extraction_options.keys()),
                    help="Select the extraction file to use as data source for dataset generation. This contains pre-extracted entities from your documents."
                )
                extraction_file_id = extraction_options[selected_extraction_display]
                st.caption(f"Selected extraction file ID: {extraction_file_id}")
            else:
                st.warning("⚠️ No completed extraction files available for this project")
                st.info("💡 Create extraction jobs first to use the 'Extract' data source")
                extraction_file_id = None

    st.divider()

    # High-Level Prompts - Compact Layout
    col9, col10 = st.columns(2)

    with col9:
        st.markdown("**👥 Audience**")
        custom_audience = st.text_input(
            "Target Audience",
            placeholder="healthcare professionals, students",
            help="Specify the target audience for the generated content (e.g., 'healthcare professionals', 'medical students', 'patients'). This helps tailor the language and complexity."
        )

        st.markdown("**🎯 Purpose**")
        custom_purpose = st.text_input(
            "Purpose",
            placeholder="patient education, research",
            help="Describe how the dataset will be used (e.g., 'patient counseling', 'medical education', 'clinical research'). This guides content generation."
        )

    with col10:
        st.markdown("**🧠 Complexity**")
        complexity_level = st.selectbox(
            "Level",
            options=["beginner", "intermediate", "advanced", "expert"],
            index=1,
            help="Content complexity level: beginner (simple explanations), intermediate (balanced), advanced (detailed), expert (technical depth)."
        )

        st.markdown("**📚 Domain**")
        domain = st.text_input(
            "Domain",
            placeholder="preventive nutrition, cardiology",
            help="Specific subject domain or specialty area (e.g., 'preventive cardiology', 'sports nutrition', 'pediatric care'). Helps focus content generation."
        )

    st.divider()

    # User-Defined Prompt - New Feature
    st.markdown("**✏️ Custom Prompt** *(Optional)*")
    custom_prompt = st.text_area(
        "Enter your custom prompt for dataset generation",
        placeholder="e.g., Generate clinical questions that would help medical students prepare for board exams...",
        height=80,
        help="Override the default generation prompts with your own custom instructions. Use {chunk} to reference the source text. Leave empty to use system-generated prompts based on the mode and high-level parameters above.",
        label_visibility="visible"
    )

    # Generation Button - Compact Layout
    col11, col12, col13 = st.columns([1, 2, 1])

    with col12:
        generate_button = st.button(
            "🚀 Generate Dataset",
            type="primary",
            width='stretch'
        )

    if generate_button:
        if not selected_doc_ids:
            st.error("❌ Please select at least one document")
        else:
            generate_dataset(
                project_id=selected_project_id,
                selected_doc_ids=selected_doc_ids,
                generation_mode=generation_mode,
                format_type=format_type,
                classification_model=classification_model,
                batch_size=batch_size,
                concurrency=concurrency,
                analyze_quality=analyze_quality,
                quality_threshold=quality_threshold,
                data_source=data_source,
                taxonomy_name=taxonomy_name,
                extraction_file_id=extraction_file_id,
                custom_audience=custom_audience,
                custom_purpose=custom_purpose,
                complexity_level=complexity_level,
                domain=domain,
                datasets_per_chunk=datasets_per_chunk,
                custom_prompt=custom_prompt
            )

    # Display generation status if there's an active job
    if hasattr(session_state, 'generation_job_id') and session_state.generation_job_id:
        display_generation_status(session_state.generation_job_id)

    st.markdown('</div>', unsafe_allow_html=True)




def get_project_taxonomies(project_id: int) -> list:
    """Get taxonomies for a specific project."""
    try:
        response = api_client.get(f"/api/v1/taxonomy?project_id={project_id}")
        if response and "taxonomies" in response:
            return response["taxonomies"]
        return []
    except Exception as e:
        st.error(f"Failed to load taxonomies: {e}")
        return []


def get_project_extraction_files(project_id: int) -> list:
    """Get available extraction files for a specific project."""
    try:
        response = api_client.get(f"/api/v1/datasets/extraction-files/{project_id}")
        if response and "extraction_files" in response:
            return response["extraction_files"]
        return []
    except Exception as e:
        st.error(f"Failed to load extraction files: {e}")
        return []


def generate_dataset(**kwargs):
    """Generate dataset with provided parameters."""
    try:
        # Prepare the request data - simplified for compact design
        request_data = {
            "project_id": kwargs["project_id"],
            "generation_mode": kwargs["generation_mode"],
            "format_type": kwargs["format_type"],
            "classification_model": kwargs["classification_model"],
            "batch_size": kwargs.get("batch_size", 50),
            "concurrency": kwargs.get("concurrency", 2),
            "analyze_quality": kwargs["analyze_quality"],
            "quality_threshold": kwargs["quality_threshold"],
            "datasets_per_chunk": kwargs["datasets_per_chunk"],
            "data_source": kwargs["data_source"]
        }

        # Add optional fields
        optional_fields = [
            "taxonomy_name", "extraction_file_id", "custom_audience", "custom_purpose",
            "complexity_level", "domain", "custom_prompt"
        ]

        for field in optional_fields:
            if kwargs.get(field):
                # Map GUI field names to API field names
                if field == "custom_audience":
                    request_data["audience"] = kwargs[field]
                elif field == "custom_purpose":
                    request_data["purpose"] = kwargs[field]
                else:
                    request_data[field] = kwargs[field]

        # Start dataset generation
        with st.spinner("🚀 Starting dataset generation..."):
            response = api_client.post("/api/v1/datasets/generate", request_data)

            if response and "job_id" in response:
                job_id = response["job_id"]
                session_state.generation_job_id = job_id
                st.success(f"✅ Dataset generation started (Job ID: {job_id})")
                st.rerun()
            else:
                st.error("❌ Failed to start dataset generation")

    except Exception as e:
        st.error(f"❌ Error: {e}")


def display_generation_status(job_id: str):
    """Display dataset generation status and results with enhanced UI."""
    try:
        status_response = api_client.get(f"/api/v1/datasets/generate/{job_id}/status")

        if status_response:
            status = status_response.get("status", "unknown")
            progress = status_response.get("progress", 0)
            current_step = status_response.get("current_step", "Unknown")

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">📊 Generation Status</div>', unsafe_allow_html=True)

            # Progress bar with percentage
            st.markdown(f'<div class="progress-label">Progress: {progress}%</div>', unsafe_allow_html=True)
            st.progress(progress / 100)

            if status == "running":
                st.markdown(f'<div class="status-info">ℹ️ Current Step: {current_step}</div>', unsafe_allow_html=True)
            elif status == "completed":
                st.markdown('<div class="status-success">✅ Dataset generation completed successfully!</div>', unsafe_allow_html=True)

                # Clear the job ID
                session_state.generation_job_id = None
            elif status == "failed":
                st.markdown('<div class="status-error">❌ Dataset generation failed</div>', unsafe_allow_html=True)
                error = status_response.get("error")
                if error:
                    st.markdown(f'<div class="status-error">❌ Error: {error}</div>', unsafe_allow_html=True)
                session_state.generation_job_id = None

            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.markdown(f'<div class="status-error">❌ Error checking generation status: {e}</div>', unsafe_allow_html=True)


def render_dataset_browser_management():
    """Render the Browse & Manage Datasets tab with dataset listing and management."""
    import time
    import json

    st.subheader("📋 Browse & Manage Datasets")

    # Project Selection
    projects = get_projects()
    if not projects:
        st.info("No projects available. Create your first project to start generating datasets.")
        return

    project_options = {f"{p['name']} (ID: {p['id']})": p['id'] for p in projects}
    selected_project_display = st.selectbox(
        "Select Project",
        options=list(project_options.keys()),
        help="Choose the project to view datasets for",
        key="dataset_browser_project_select"
    )
    selected_project_id = project_options[selected_project_display]

    # Get all datasets for this project
    datasets = get_datasets_for_project(selected_project_id)

    if not datasets:
        st.info("No datasets found for this project.")
        st.markdown("""
        **To create a dataset:**
        1. Go to the **🚀 Generate Dataset** tab
        2. Configure dataset parameters
        3. Click **Generate Dataset**
        """)
        return

    # Search and filter controls
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        search_query = st.text_input("🔍 Search datasets", placeholder="Enter dataset name or ID...", key="dataset_search")

    with col2:
        format_filter = st.selectbox("Filter by Format", ["All", "jsonl", "parquet", "json"], key="dataset_format_filter")

    with col3:
        sort_by = st.selectbox("Sort by", ["Newest", "Oldest", "Name", "Size"], key="dataset_sort")

    # Filter datasets based on search and format
    filtered_datasets = datasets
    if search_query:
        filtered_datasets = [d for d in datasets if search_query.lower() in str(d.get('id', '')).lower() or search_query.lower() in d.get('name', '').lower()]

    if format_filter != "All":
        filtered_datasets = [d for d in filtered_datasets if d.get('format_type', '').lower() == format_filter.lower()]

    # Sort datasets
    if sort_by == "Newest":
        filtered_datasets.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    elif sort_by == "Oldest":
        filtered_datasets.sort(key=lambda x: x.get('created_at', ''))
    elif sort_by == "Name":
        filtered_datasets.sort(key=lambda x: x.get('name', ''))
    elif sort_by == "Size":
        filtered_datasets.sort(key=lambda x: x.get('file_size', 0), reverse=True)

    # Bulk selection controls
    render_bulk_selection_controls_datasets(filtered_datasets)

    # Display datasets
    for dataset in filtered_datasets:
        render_dataset_card(dataset)

    # Export functionality
    st.divider()
    st.subheader("📤 Export Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📄 Export All Datasets as JSON", width='stretch'):
            export_datasets_json(filtered_datasets)

    with col2:
        if st.button("📊 Export Summary as CSV", width='stretch'):
            export_datasets_csv(filtered_datasets)


def render_bulk_selection_controls_datasets(datasets: List[Dict[str, Any]]):
    """Render bulk selection controls for datasets."""
    if not datasets:
        return

    # Initialize bulk selection state
    if 'dataset_bulk_selected' not in st.session_state:
        st.session_state.dataset_bulk_selected = []

    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        if st.button("✅ Select All", key="dataset_select_all", help="Select all visible datasets"):
            st.session_state.dataset_bulk_selected = [str(d['id']) for d in datasets]
            st.rerun()

    with col2:
        if st.button("❌ Deselect All", key="dataset_deselect_all", help="Deselect all datasets"):
            st.session_state.dataset_bulk_selected = []
            st.rerun()

    with col3:
        selected_count = len(st.session_state.dataset_bulk_selected)
        if selected_count > 0:
            if st.button(f"🗑️ Delete Selected ({selected_count})", key="dataset_bulk_delete", type="secondary"):
                st.session_state.dataset_bulk_delete_confirm = True
                st.rerun()

    with col4:
        if selected_count > 0:
            st.success(f"{selected_count} dataset{'s' if selected_count > 1 else ''} selected")

    # Bulk delete confirmation
    if st.session_state.get('dataset_bulk_delete_confirm', False):
        render_bulk_delete_confirmation_datasets()


def render_bulk_delete_confirmation_datasets():
    """Render bulk delete confirmation dialog for datasets."""
    selected_ids = st.session_state.get('dataset_bulk_selected', [])
    if not selected_ids:
        return

    st.warning(f"Are you sure you want to delete {len(selected_ids)} dataset{'s' if len(selected_ids) > 1 else ''}?")
    st.error("This action cannot be undone. All dataset files, generation parameters, and database records will be permanently removed.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("❌ Yes, Delete All", type="primary", width='stretch'):
            try:
                with st.spinner(f"Deleting {len(selected_ids)} datasets..."):
                    for dataset_id in selected_ids:
                        api_client.delete(f"/api/v1/datasets/{dataset_id}/delete")

                st.success(f"Successfully deleted {len(selected_ids)} datasets!")
                st.session_state.dataset_bulk_selected = []
                del st.session_state.dataset_bulk_delete_confirm
                st.cache_data.clear()
                st.rerun()

            except APIError as e:
                st.error(f"Failed to delete datasets: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

    with col2:
        if st.button("⬅️ Cancel", width='stretch'):
            del st.session_state.dataset_bulk_delete_confirm
            st.rerun()


def render_dataset_card(dataset: Dict[str, Any]):
    """Render a dataset card with management actions."""
    with st.container():
        # Checkbox for bulk selection
        col_checkbox, col_content = st.columns([0.1, 0.9])

        with col_checkbox:
            dataset_id_str = str(dataset['id'])
            is_selected = st.checkbox(
                f"Select dataset {dataset_id_str}",
                key=f"dataset_select_{dataset_id_str}",
                value=dataset_id_str in st.session_state.get('dataset_bulk_selected', []),
                label_visibility="collapsed"
            )
            if is_selected and dataset_id_str not in st.session_state.dataset_bulk_selected:
                st.session_state.dataset_bulk_selected.append(dataset_id_str)
            elif not is_selected and dataset_id_str in st.session_state.dataset_bulk_selected:
                st.session_state.dataset_bulk_selected.remove(dataset_id_str)

        with col_content:
            # Dataset header
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.markdown(f"**Dataset {dataset['id']}**")
                name = dataset.get('name', f"Dataset {dataset['id']}")
                st.caption(f"Name: {name}")

            with col2:
                format_type = dataset.get('format_type', 'unknown').upper()
                st.markdown(f"**Format:** {format_type}")

            with col3:
                created_date = dataset['created_at'].split('T')[0] if 'T' in dataset['created_at'] else dataset['created_at']
                st.caption(f"Created: {created_date}")

            # Dataset details
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.button("👁️ View Details", key=f"view_dataset_{dataset['id']}", width='stretch'):
                    # Show dataset details in an expander
                    st.session_state[f'dataset_details_expanded_{dataset["id"]}'] = True
                    st.rerun()

            with col2:
                # Direct link to backend for downloading
                api_base = api_client.base_url
                download_url = f"{api_base}/api/v1/datasets/{dataset['id']}/download"
                st.markdown(
                    f'<a href="{download_url}" target="_self" style="text-decoration: none;">'
                    f'<button style="width: 100%; cursor: pointer; border: 1px solid rgba(255, 255, 255, 0.2); '
                    f'background-color: transparent; color: white; padding: 0.5rem; border-radius: 0.25rem;">'
                    f'📥 Download</button></a>',
                    unsafe_allow_html=True
                )

            with col3:
                if st.button("📊 Export JSON", key=f"export_dataset_{dataset['id']}", width='stretch'):
                    try:
                        response = api_client.get(f"/api/v1/datasets/{dataset['id']}/details")
                        if response:
                            import json
                            json_str = json.dumps(response, indent=2, default=str)
                            st.download_button(
                                label="📥 Download",
                                data=json_str,
                                file_name=f"dataset_{dataset['id']}_details.json",
                                mime="application/json",
                                width='stretch'
                            )
                    except APIError as e:
                        st.error(f"Failed to export dataset details: {e}")

            with col4:
                if st.button("🗑️ Delete", key=f"delete_dataset_{dataset['id']}", width='stretch', type="secondary"):
                    st.session_state[f'dataset_delete_confirm_{dataset["id"]}'] = True
                    st.rerun()

            # Individual delete confirmation
            if st.session_state.get(f'dataset_delete_confirm_{dataset["id"]}', False):
                render_individual_delete_confirmation_datasets(dataset)

            # Dataset details expander
            if st.session_state.get(f'dataset_details_expanded_{dataset["id"]}', False):
                with st.expander("📋 Dataset Details", expanded=True):
                    render_dataset_details(dataset)

        st.divider()


def render_individual_delete_confirmation_datasets(dataset: Dict[str, Any]):
    """Render individual delete confirmation for a dataset."""
    dataset_id = dataset['id']

    st.warning(f"Are you sure you want to delete dataset {dataset_id}?")
    st.error("This action cannot be undone. All dataset files, generation parameters, and database records will be permanently removed.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button(f"❌ Yes, Delete Dataset {dataset_id}", type="primary", width='stretch'):
            try:
                with st.spinner(f"Deleting dataset {dataset_id}..."):
                    api_client.delete(f"/api/v1/datasets/{dataset_id}/delete")

                st.success(f"Dataset {dataset_id} deleted successfully!")
                del st.session_state[f'dataset_delete_confirm_{dataset_id}']
                st.cache_data.clear()
                st.rerun()

            except APIError as e:
                st.error(f"Failed to delete dataset: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

    with col2:
        if st.button("⬅️ Cancel", width='stretch'):
            del st.session_state[f'dataset_delete_confirm_{dataset_id}']
            st.rerun()


def render_dataset_details(dataset: Dict[str, Any]):
    """Render detailed information about a dataset."""
    try:
        # Get full dataset details
        response = api_client.get(f"/api/v1/datasets/{dataset['id']}/details")
        if not response:
            st.error("Failed to load dataset details.")
            return

        # Display key information
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Basic Information:**")
            st.write(f"**ID:** {response.get('id', 'N/A')}")
            st.write(f"**Name:** {response.get('name', 'N/A')}")
            st.write(f"**Format:** {response.get('format_type', 'N/A')}")
            st.write(f"**Created:** {response.get('created_at', 'N/A')}")

        with col2:
            st.markdown("**Technical Details:**")
            st.write(f"**File Size:** {response.get('file_size', 'N/A')} bytes")
            st.write(f"**Record Count:** {response.get('record_count', 'N/A')}")
            st.write(f"**Chunking Strategy:** {response.get('chunking_strategy', 'N/A')}")
            st.write(f"**Embedding Model:** {response.get('embedding_model', 'N/A')}")

        # Show sample records if available
        if response.get('sample_records'):
            st.markdown("**Sample Records:**")
            with st.expander("View Sample Data", expanded=False):
                st.json(response['sample_records'])

    except APIError as e:
        st.error(f"Failed to load dataset details: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")


def get_datasets_for_project(project_id: int) -> List[Dict[str, Any]]:
    """Get datasets for a project."""
    try:
        response = api_client.get(f"/api/v1/datasets/list/{project_id}")
        if isinstance(response, dict) and "datasets" in response:
            return response["datasets"]
        return []
    except Exception:
        return []


def export_datasets_json(datasets: List[Dict[str, Any]]):
    """Export datasets as JSON."""
    try:
        import json
        json_str = json.dumps(datasets, indent=2, default=str)
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"datasets_{int(time.time())}.json",
            mime="application/json"
        )
    except Exception as e:
        st.error(f"Failed to export datasets as JSON: {e}")


def export_datasets_csv(datasets: List[Dict[str, Any]]):
    """Export datasets summary as CSV."""
    try:
        import pandas as pd

        # Create summary data
        csv_data = []
        for dataset in datasets:
            row = {
                'id': dataset.get('id', ''),
                'name': dataset.get('name', ''),
                'format_type': dataset.get('format_type', ''),
                'created_at': dataset.get('created_at', ''),
                'file_size': dataset.get('file_size', 0),
                'record_count': dataset.get('record_count', 0),
                'chunking_strategy': dataset.get('chunking_strategy', ''),
                'embedding_model': dataset.get('embedding_model', '')
            }
            csv_data.append(row)

        df = pd.DataFrame(csv_data)
        csv_string = df.to_csv(index=False)

        st.download_button(
            label="📊 Download CSV",
            data=csv_string,
            file_name=f"datasets_summary_{int(time.time())}.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Failed to export datasets as CSV: {e}")