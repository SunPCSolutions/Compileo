"""
Wizard step components for the dataset creation workflow.
"""

import os
import time
import streamlit as st
from typing import List, Dict, Any, Optional

from src.compileo.features.gui.state.wizard_state import wizard_state
from src.compileo.features.gui.services.api_client import api_client, APIError
from src.compileo.features.gui.services.unified_job_service import unified_job_service, JobResult
from src.compileo.features.gui.utils.settings_storage import settings_storage
from ...jobhandle.models import JobType, JobPriority

# Cache for settings to avoid repeated DB calls
_settings_cache = None
_settings_cache_time = 0
_SETTINGS_CACHE_TTL = 5  # 5 seconds - refresh quickly to pick up setting changes

def _get_settings():
    """Get settings from database with caching."""
    global _settings_cache, _settings_cache_time
    current_time = time.time()
    if _settings_cache is None or (current_time - _settings_cache_time) > _SETTINGS_CACHE_TTL:
        # Get current user ID from session state if available
        owner_id = None
        try:
            from ..state.session_state import session_state
            if session_state.user_id:
                owner_id = session_state.user_id
        except:
            pass
        _settings_cache = settings_storage.load_settings(owner_id=owner_id)
        _settings_cache_time = current_time
    return _settings_cache

def _clear_settings_cache():
    """Clear the settings cache to force a refresh on next access."""
    global _settings_cache, _settings_cache_time
    _settings_cache = None
    _settings_cache_time = 0

def _get_model_for_provider(task: str, provider: str) -> str:
    """Get the configured model name for a task and provider."""
    settings = _get_settings()
    key = f"{task}_{provider}_model"
    
    # Debug: Log settings lookup
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"Looking for setting: {key}")
    logger.debug(f"Available settings keys: {list(settings.keys())}")
    
    default_models = {
        "parsing_gemini": "gemini-2.5-flash",
        "parsing_grok": "grok-4-fast-reasoning",
        "parsing_openai": "gpt-4o",
        "parsing_ollama": "mistral:latest",
        "chunking_gemini": "gemini-2.5-flash",
        "chunking_grok": "grok-4-fast-reasoning",
        "chunking_openai": "gpt-4o",
        "chunking_ollama": "mistral:latest",
        "classification_gemini": "gemini-2.5-flash",
        "classification_grok": "grok-4-fast-reasoning",
        "classification_openai": "gpt-4o",
        "classification_ollama": "mistral:latest",
        "generation_gemini": "gemini-2.5-flash",
        "generation_grok": "grok-4-fast-reasoning",
        "generation_openai": "gpt-4o",
        "generation_ollama": "mistral:latest",
    }
    
    value = settings.get(key, default_models.get(key, provider))
    logger.debug(f"Found value for {key}: {value}")
    return value

# Project selection is now in wizard/project_selection.py

def _wait_for_job_completion(job_id: str, poll_interval: int = 3) -> bool:
    """
    Wait for a job to complete by polling the job status endpoint.
    Polls indefinitely until job completes or fails.
    
    Args:
        job_id: The job ID to wait for
        poll_interval: Seconds between status checks
        
    Returns:
        True if job completed successfully, False if failed
    """
    import time
    
    while True:
        try:
            status_response = api_client.get(f"/api/v1/jobs/status/{job_id}")
            
            if status_response:
                status = status_response.get("status", "unknown")
                
                if status == "completed":
                    return True
                elif status == "failed":
                    error = status_response.get("error", "Unknown error")
                    st.error(f"Job failed: {error}")
                    return False
                # Otherwise continue polling (pending, running, queued, processing)
            
            time.sleep(poll_interval)
            
        except Exception:
            # If endpoint fails, wait and retry
            time.sleep(poll_interval)

def render_document_upload():
    """Render the document upload step."""
    st.header("📤 Step 2: Document Upload")

    project_id = wizard_state.get_step_data("project_selection").get("project_id")
    if not project_id:
        st.error("Please select a project first")
        return

    st.markdown("Upload documents to process for your dataset. Supported formats: PDF, Office (DOCX, PPTX, XLSX), TXT, MD, CSV, JSON, XML")

    # Check for existing documents in the project
    try:
        documents_response = api_client.get(f"/api/v1/documents?project_id={project_id}")
        existing_documents = documents_response.get("documents", []) if documents_response else []
    except APIError as e:
        st.error(f"Failed to check existing documents: {e}")
        existing_documents = []
    except Exception as e:
        st.error(f"Error checking existing documents: {e}")
        existing_documents = []

    # Display existing documents if any
    if existing_documents:
        st.subheader("📁 Existing Project Documents")
        st.info(f"This project already has {len(existing_documents)} document(s). You can proceed to the next step or upload additional documents.")

        for doc in existing_documents:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"📄 {doc['file_name']}")
            with col2:
                st.write(f"ID: {doc['id']}")
            with col3:
                created_date = doc['created_at']
                if isinstance(created_date, str) and len(created_date) > 10:
                    created_date = created_date[:10]
                st.write(created_date)

        st.success("✅ Project has existing documents. You can proceed to the next step.")

        # Store existing documents in wizard state
        wizard_state.update_step_data("document_upload", "existing_documents", existing_documents)
        wizard_state.update_step_data("document_upload", "existing_count", len(existing_documents))

        st.divider()
        st.subheader("Or Upload Additional Documents")
    else:
        st.info("No documents found in this project. Please upload documents to proceed.")

    # Initialize session state for tracking uploaded files
    if 'uploaded_file_names' not in st.session_state:
        st.session_state.uploaded_file_names = set()
    
    # Track current project to detect project changes
    if 'wizard_current_project_id' not in st.session_state:
        st.session_state.wizard_current_project_id = None
    
    # Track if we need to clear state (on project change or first load)
    state_key = f"wizard_state_cleared_for_project_{project_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = False
    
    # Clear uploaded files if project has changed or on first load for this project
    if st.session_state.wizard_current_project_id != project_id or not st.session_state.get(state_key, False):
        st.session_state.uploaded_file_names = set()
        st.session_state.wizard_current_project_id = project_id
        st.session_state[state_key] = True
        # Also clear wizard state upload data for the new project
        wizard_state.set_step_data("document_upload", {
            "uploaded_files": [],
            "uploaded_count": 0,
            "existing_documents": existing_documents,
            "existing_count": len(existing_documents)
        })
        # Clear state for other projects to prevent memory buildup
        for key in list(st.session_state.keys()):
            if key.startswith("wizard_state_cleared_for_project_") and key != state_key:
                del st.session_state[key]

    # File uploader with automatic upload - use a unique key
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=["pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls", "txt", "md", "csv", "json", "xml"],
        help="Select one or more documents to upload. Files will be uploaded automatically.",
        key="wizard_file_uploader"
    )

    # Filter out already uploaded files to prevent duplicate uploads
    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_file_names]
    else:
        new_files = []

    # Auto-upload when new files are selected
    if new_files and not st.session_state.get('upload_in_progress', False):
        st.session_state.upload_in_progress = True

        st.subheader("📤 Uploading Files")
        with st.spinner("Uploading documents..."):
            try:
                # Prepare files for upload
                files = []
                for file in new_files:
                    file_content = file.read()
                    files.append(("files", (file.name, file_content, file.type)))

                # Prepare form data
                data = {"project_id": str(project_id)}

                # Upload via API
                response = api_client.post(
                    "/api/v1/documents/upload",
                    files=files,
                    data=data
                )

                uploaded_count = response.get("files_count", 0) if response else 0
                if response:
                    st.success(f"✅ Successfully uploaded {uploaded_count} documents")
                else:
                    st.error("❌ Failed to upload documents")
                    st.session_state.upload_in_progress = False
                    return

                # Track uploaded file names to prevent re-upload
                for file in new_files:
                    st.session_state.uploaded_file_names.add(file.name)

                # Merge with existing uploaded files in wizard state
                existing_uploaded = wizard_state.get_step_data("document_upload").get("uploaded_files", [])
                all_uploaded = list(set(existing_uploaded + [f.name for f in new_files]))
                wizard_state.update_step_data("document_upload", "uploaded_files", all_uploaded)
                wizard_state.update_step_data("document_upload", "uploaded_count", len(all_uploaded))

                # Clear the upload state and refresh
                st.session_state.upload_in_progress = False
                st.rerun()

            except APIError as e:
                st.error(f"❌ Failed to upload documents: {e}")
                st.session_state.upload_in_progress = False
            except Exception as e:
                st.error(f"❌ Error uploading documents: {e}")
                st.session_state.upload_in_progress = False

    # Show upload status
    elif uploaded_files and st.session_state.get('upload_in_progress', False):
        st.info("📤 Upload in progress...")

    # Show uploaded documents with delete option
    uploaded_docs = wizard_state.get_step_data("document_upload").get("uploaded_files", [])
    if uploaded_docs:
        st.subheader("📄 Uploaded Documents")

        # Get current project documents to find IDs for deletion
        try:
            docs_response = api_client.get(f"/api/v1/documents?project_id={project_id}")
            current_docs = docs_response.get("documents", []) if docs_response else []
        except:
            current_docs = []

        for doc_name in uploaded_docs:
            # Find the document in current docs
            doc_info = next((doc for doc in current_docs if doc['file_name'] == doc_name), None)

            if doc_info:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"📄 {doc_name}")
                with col2:
                    st.write(f"ID: {doc_info['id']}")
                with col3:
                    if st.button("🗑️", key=f"delete_uploaded_{doc_info['id']}", help=f"Delete {doc_name}"):
                        try:
                            api_client.delete(f"/api/v1/documents/{doc_info['id']}")
                            st.success(f"Deleted {doc_name}")

                            # Remove from wizard state
                            updated_uploaded = [d for d in uploaded_docs if d != doc_name]
                            wizard_state.update_step_data("document_upload", "uploaded_files", updated_uploaded)
                            wizard_state.update_step_data("document_upload", "uploaded_count", len(updated_uploaded))

                            st.rerun()
                        except APIError as e:
                            st.error(f"Failed to delete {doc_name}: {e}")
                        except Exception as e:
                            st.error(f"Error deleting {doc_name}: {e}")
            else:
                # Document not found in API (might have been deleted already)
                st.write(f"📄 {doc_name} (not found)")

def render_processing_config():
    """Render the processing configuration step."""
    # Clear settings cache to ensure we get fresh model settings
    _clear_settings_cache()
    
    st.header("⚙️ Step 3: Processing Configuration")

    # Check prerequisites
    project_data = wizard_state.get_step_data("project_selection")
    upload_data = wizard_state.get_step_data("document_upload")

    if not project_data.get("project_id"):
        st.error("❌ Please complete Project Selection first.")
        st.info("You need to select a project before configuring processing options.")
        return

    if not upload_data.get("uploaded_files") and not upload_data.get("existing_documents"):
        st.error("❌ Please complete Document Upload first.")
        st.info("You need to upload documents or have existing project documents before configuring processing.")
        return

    project_id = project_data.get("project_id")
    
    st.markdown("Configure how your documents will be processed and chunked.")

    # AI Model Selections
    st.subheader("AI Models & Software")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Determine supported parsers based on uploaded files (reactive filtering)
        all_parsers = ["grok", "gemini", "ollama", "openai", "huggingface", "unstructured", "pypdf", "novlm"]
        supported_parsers = all_parsers.copy()
        
        # Check uploaded files if they exist in session state (from Step 2)
        uploaded_files = wizard_state.get_step_data("document_upload").get("uploaded_files", [])
        if uploaded_files:
            has_non_pdf = any(not f.lower().endswith('.pdf') for f in uploaded_files)
            if has_non_pdf:
                # Restrict to parsers that handle non-PDFs
                text_extensions = {'.txt', '.md', '.csv', '.json', '.xml'}
                has_complex_format = any(os.path.splitext(f)[1].lower() not in text_extensions and not f.lower().endswith('.pdf') for f in uploaded_files)
                
                if has_complex_format:
                    supported_parsers = ["unstructured", "novlm"]
                else:
                    supported_parsers = ["unstructured", "novlm", "gemini", "ollama"]
        
        # Determine default index
        default_parser = "grok" if "grok" in supported_parsers else "novlm"
        try:
            default_index = supported_parsers.index(default_parser)
        except ValueError:
            default_index = 0

        parsing_model = st.selectbox(
            "Document Parsing",
            options=supported_parsers,
            index=default_index,
            help="AI model for document parsing. Available models filtered based on uploaded file types."
        )

    with col2:
        chunking_model = st.selectbox(
            "Text Chunking",
            options=["grok", "gemini", "openai", "ollama"],
            help="AI model for intelligent text chunking"
        )

    with col3:
        classification_model = st.selectbox(
            "Document Classification",
            options=["grok", "gemini", "openai", "ollama"],
            help="AI model for document classification"
        )

    with col4:
        generation_model = st.selectbox(
            "Dataset Generation",
            options=["grok", "gemini", "openai", "ollama"],
            help="AI model for generating dataset entries and responses"
        )

    # Document Selection
    st.subheader("Document Selection")

    project_id = wizard_state.get_step_data("project_selection").get("project_id")
    if project_id:
        try:
            documents_response = api_client.get(f"/api/v1/documents?project_id={project_id}")
            all_documents = documents_response.get("documents", []) if documents_response else []
        except APIError as e:
            st.error(f"Failed to load documents: {e}")
            all_documents = []
        except Exception as e:
            st.error(f"Error loading documents: {e}")
            all_documents = []

        if all_documents:
            # Create options for multiselect
            doc_options = [f"{doc['file_name']} (ID: {doc['id']})" for doc in all_documents]
            doc_ids = [doc['id'] for doc in all_documents]

            # Get previously selected documents or default to all
            selected_doc_ids = st.session_state.get('selected_document_ids', doc_ids)

            selected_options = st.multiselect(
                "Select documents to process",
                options=doc_options,
                default=[opt for opt, doc_id in zip(doc_options, doc_ids) if doc_id in selected_doc_ids],
                help="Choose which documents to include in dataset generation. All documents are selected by default."
            )

            # Convert selected options back to document IDs
            selected_document_ids = []
            for option in selected_options:
                # Extract ID from option string
                for doc in all_documents:
                    if f"{doc['file_name']} (ID: {doc['id']})" == option:
                        selected_document_ids.append(doc['id'])
                        break

            # Store selected document IDs
            st.session_state.selected_document_ids = selected_document_ids

            st.info(f"Selected {len(selected_document_ids)} of {len(all_documents)} documents")
        else:
            st.warning("No documents available in this project. Please upload documents first.")
            selected_document_ids = []
    else:
        st.error("No project selected")
        selected_document_ids = []

    # Initialize default values for all parameters BEFORE the UI section
    chunk_size = 512
    overlap = 50
    delimiters = ["\n\n", "\n"]
    semantic_prompt = ""
    schema_json = ""

    # Chunking strategy
    st.subheader("Text Splitting Strategy")
    chunk_strategy = st.selectbox(
        "Choose chunking method",
        options=["character", "token", "semantic", "delimiter", "schema"],
        help="Method to split documents into chunks"
    )

    # Parameters based on strategy
    if chunk_strategy == "character":
        col3, col4 = st.columns(2)
        with col3:
            chunk_size = st.slider("Chunk Size", 100, 4000, 1000, 100, help="Characters per chunk")
        with col4:
            overlap = st.slider("Overlap", 0, 500, 100, 25, help="Overlap characters")
    elif chunk_strategy == "token":
        col3, col4 = st.columns(2)
        with col3:
            chunk_size = st.slider("Chunk Size", 100, 2000, 512, 50, help="Tokens per chunk")
        with col4:
            overlap = st.slider("Overlap", 0, 200, 50, 10, help="Overlap tokens")
    elif chunk_strategy == "semantic":
        semantic_prompt = st.text_area(
            "Semantic Prompt",
            height=100,
            placeholder="Custom prompt for semantic chunking...",
            help="Leave empty for default",
            key="semantic_prompt"
        )
    elif chunk_strategy == "delimiter":
        st.markdown("**Common Delimiters**")
        col_delim1, col_delim2 = st.columns(2)

        with col_delim1:
            common_delimiters = st.multiselect(
                "Quick Select",
                ["# (Markdown headers)", "--- (Horizontal rule)", "### (H3 headers)", "## (H2 headers)", "\n\n (Double newline)", "\n (Single newline)", ". ", "! ", "? "],
                ["# (Markdown headers)"],
                help="Common split patterns - select what appears in parentheses",
                key="common_delimiters"
            )

            # Extract actual delimiter values from the display strings
            actual_delimiters = []
            for delim in common_delimiters:
                if " (Markdown headers)" in delim:
                    actual_delimiters.append("#")
                elif " (Horizontal rule)" in delim:
                    actual_delimiters.append("---")
                elif " (H3 headers)" in delim:
                    actual_delimiters.append("###")
                elif " (H2 headers)" in delim:
                    actual_delimiters.append("##")
                elif " (Double newline)" in delim:
                    actual_delimiters.append("\n\n")
                elif " (Single newline)" in delim:
                    actual_delimiters.append("\n")
                elif delim in [". ", "! ", "? "]:
                    actual_delimiters.append(delim)

            common_delimiters = actual_delimiters

        with col_delim2:
            custom_delimiter = st.text_input(
                "Custom Delimiter",
                placeholder="e.g., #, ---, <div>, etc.",
                help="Enter any custom delimiter or pattern",
                key="custom_delimiter"
            )

        # Combine common and custom delimiters
        delimiters = common_delimiters.copy()
        if custom_delimiter.strip():
            # Allow multiple custom delimiters separated by commas
            custom_list = [d.strip() for d in custom_delimiter.split(",") if d.strip()]
            delimiters.extend(custom_list)

        # Remove duplicates while preserving order
        seen = set()
        delimiters = [d for d in delimiters if not (d in seen or seen.add(d))]

        if delimiters:
            st.info(f"📋 Using {len(delimiters)} delimiter(s): {', '.join(repr(d) for d in delimiters[:3])}{'...' if len(delimiters) > 3 else ''}")
        else:
            st.warning("⚠️ No delimiters selected. Add common delimiters or enter a custom one.")
    else:  # schema
        schema_json = st.text_area(
            "Schema JSON",
            height=150,
            placeholder='{"rules": [{"type": "pattern", "value": "#"}], "combine": "any"}',
            help="JSON schema for chunking rules",
            key="schema_json"
        )

    # Auto-save configuration (will be saved when Next is pressed)
    config = {
        "parsing_model": parsing_model,
        "chunking_model": chunking_model,
        "classification_model": classification_model,
        "generation_model": generation_model,
        "chunk_strategy": chunk_strategy,
        "selected_document_ids": selected_document_ids
    }

    if chunk_strategy == "character":
        config.update({"chunk_size": chunk_size, "overlap": overlap})
    elif chunk_strategy == "token":
        config.update({"chunk_size": chunk_size, "overlap": overlap})
    elif chunk_strategy == "semantic":
        config.update({"semantic_prompt": semantic_prompt})
    elif chunk_strategy == "delimiter":
        config.update({"delimiters": delimiters})
    else:  # schema
        config.update({"schema_json": schema_json})

    # Store config in session state for auto-save
    st.session_state.processing_config = config

    st.markdown("---")

    # Get current document status (refresh from server)
    try:
        docs_response = api_client.get(f"/api/v1/documents?project_id={project_id}")
        documents = docs_response.get("documents", []) if docs_response else []
    except:
        documents = []

    # Filter to selected documents
    selected_docs = [doc for doc in documents if doc["id"] in selected_document_ids]
    
    if not selected_docs:
        st.warning("⚠️ No documents selected. Please select documents above.")
        return

    unparsed_docs = [doc for doc in selected_docs if doc.get("status") == "uploaded"]

    # Get wizard state for chunking status
    # IMPORTANT: Reset chunking_status if project has changed to prevent stale state
    # from previous projects being carried over
    processing_data = wizard_state.get_step_data("processing_config") or {}
    stored_project_id = processing_data.get("stored_project_id")
    if stored_project_id != project_id:
        # Project changed - reset chunking status
        chunking_status = "not_started"
        wizard_state.update_step_data("processing_config", "stored_project_id", project_id)
        wizard_state.update_step_data("processing_config", "chunking_status", "not_started")
    else:
        chunking_status = processing_data.get("chunking_status", "not_started")

    if chunking_status == "completed":
        st.success("✅ Documents have been processed and are ready for taxonomy generation!")
    elif chunking_status == "running":
        st.info("⏳ Processing is in progress... Please wait for it to complete.")

    button_label = "🔄 Re-Generate Chunks" if chunking_status == "completed" else "🚀 Generate Chunks"

    if st.button(button_label, type="primary", key="generate_chunks_btn"):
        wizard_state.update_step_data("processing_config", "chunking_status", "running")

        # Get user ID for job attribution (same as non-wizard document processing)
        try:
            from ..state.session_state import session_state
            user_id = session_state.user_id
            if not user_id:
                st.error("❌ You must be logged in to process documents.")
                return
        except Exception as e:
            st.error(f"❌ Authentication error: {e}")
            return

        # Step 1: Parse unparsed documents
        if unparsed_docs:
            st.info(f"🚀 Submitting parsing job for {len(unparsed_docs)} documents...")
            
            # NOTE: We intentionally do NOT pass "model" here to use the same pipeline as non-wizard
            # The non-wizard document processing (document_processing_service.py) does not pass "model"
            # and instead relies on the backend settings to determine the default model.
            # This ensures consistency and allows AI models to use their default parameters.
            
            parse_parameters = {
                "operation": "parse_documents",
                "project_id": project_id,
                "document_ids": [doc["id"] for doc in unparsed_docs],
                "parser": parsing_model,
                "pages_per_split": 50,
                "overlap_pages": 0,
                "page_selection": None
            }
            
            # Use unified job service (same as non-wizard document processing)
            parse_result = unified_job_service.submit_and_monitor(
                job_type=JobType.DOCUMENT_PROCESSING,
                parameters=parse_parameters,
                priority=JobPriority.NORMAL,
                user_id=user_id,
                title=f"Parsing {len(unparsed_docs)} Document(s)",
                timeout=1800,
                poll_interval=3
            )
            
            if not parse_result.success:
                st.error(f"❌ Parsing failed: {parse_result.error_message}")
                wizard_state.update_step_data("processing_config", "chunking_status", "failed")
                return
            
            st.success(f"✅ Successfully parsed {len(unparsed_docs)} document(s)!")

        # Step 2: Chunk all selected documents
        # In re-generate scenarios, we chunk everything again
        docs_to_chunk = [doc["id"] for doc in selected_docs]
        if docs_to_chunk:
                st.info(f"🚀 Submitting chunking job for {len(docs_to_chunk)} documents...")
                
                # NOTE: We intentionally do NOT pass "model" here to use the same pipeline as non-wizard
                # The non-wizard document processing does not pass "model" and instead relies on
                # the backend settings to determine the default model.
                
                # Determine effective chunk strategy
                effective_chunk_strategy = chunk_strategy
                if chunk_strategy == "schema" and config.get("schema_json"):
                    effective_chunk_strategy = "schema"
                
                chunk_parameters = {
                    "operation": "chunk_documents",
                    "project_id": project_id,
                    "document_ids": docs_to_chunk,
                    "chunker": chunking_model,
                    "chunk_strategy": effective_chunk_strategy,
                    "skip_parsing": True,
                    "sliding_window": True
                }
                
                # Add strategy-specific parameters
                if effective_chunk_strategy == "character":
                    chunk_parameters.update({
                        "chunk_size": config.get("chunk_size", 512),
                        "overlap": config.get("overlap", 50)
                    })
                elif effective_chunk_strategy == "token":
                    chunk_parameters.update({
                        "chunk_size": config.get("chunk_size", 512),
                        "overlap": config.get("overlap", 50)
                    })
                elif effective_chunk_strategy == "semantic":
                    chunk_parameters.update({
                        "semantic_prompt": config.get("semantic_prompt", "")
                    })
                elif effective_chunk_strategy == "schema":
                    chunk_parameters.update({
                        "schema_definition": config.get("schema_json", "")
                    })
                elif effective_chunk_strategy == "delimiter":
                    chunk_parameters.update({
                        "delimiters": config.get("delimiters", ["\n\n", "\n"])
                    })
                
                # Use unified job service (same as non-wizard document processing)
                chunk_result = unified_job_service.submit_and_monitor(
                    job_type=JobType.DOCUMENT_PROCESSING,
                    parameters=chunk_parameters,
                    priority=JobPriority.NORMAL,
                    user_id=user_id,
                    title=f"Chunking {len(docs_to_chunk)} Document(s)",
                    timeout=3600,
                    poll_interval=3
                )
                
                if chunk_result.success:
                    st.success("✅ Processing completed!")
                    wizard_state.update_step_data("processing_config", "chunking_status", "completed")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(f"❌ Processing failed: {chunk_result.error_message}")
                    wizard_state.update_step_data("processing_config", "chunking_status", "failed")

def render_taxonomy_selection():
    """Render the taxonomy selection step."""
    st.header("🏷️ Step 4: Taxonomy Selection")

    # Check prerequisites
    project_data = wizard_state.get_step_data("project_selection")
    processing_data = wizard_state.get_step_data("processing_config")

    if not project_data.get("project_id"):
        st.error("❌ Please complete Project Selection first.")
        st.info("You need to select a project before managing taxonomies.")
        return

    if not processing_data.get("selected_document_ids"):
        st.error("❌ Please complete Processing Configuration first.")
        st.info("You need to configure processing options and select documents before managing taxonomies.")
        return

    project_id = project_data.get("project_id")

    st.markdown("Choose or create a taxonomy for categorizing your dataset.")

    # Get available taxonomies
    try:
        taxonomies_response = api_client.get(f"/api/v1/taxonomy?project_id={project_id}")
        taxonomies = taxonomies_response.get("taxonomies", []) if taxonomies_response else []
    except APIError as e:
        st.error(f"Failed to load taxonomies: {e}")
        taxonomies = []

    if taxonomies:
        st.subheader("Available Taxonomies")
        for tax in taxonomies:
            with st.expander(f"🏷️ {tax['name']} - {tax['categories_count']} categories"):
                st.write(f"**Description:** {tax['description'] or 'No description'}")
                st.write(f"**Confidence:** {tax['confidence_score']:.2f}")
                st.write(f"**Created:** {tax['created_at']}")

                if st.button(f"Select {tax['name']}", key=f"select_tax_{tax['id']}"):
                    wizard_state.update_step_data("taxonomy_selection", "selected_taxonomy", tax)
                    st.success(f"Selected taxonomy: {tax['name']}")
    else:
        st.info("No taxonomies available for this project.")

    st.divider()

    # Create new taxonomy option
    st.subheader("Create New Taxonomy")
    if st.button("Generate Taxonomy from Documents"):
        # Get selected document IDs from processing config
        processing_config = wizard_state.get_step_data("processing_config") or {}
        selected_document_ids = processing_config.get("selected_document_ids", [])
        
        if not selected_document_ids:
            st.error("No documents selected. Please go back to Processing Configuration and select documents.")
            return

        # Generate taxonomy using API endpoint with complete processing mode
        with st.spinner("Generating taxonomy from chunks..."):
            try:
                # Get selected models for taxonomy generation
                taxonomy_generator = processing_config.get("classification_model", "gemini")
                # NOTE: We intentionally do NOT pass "model" here to use the same pipeline as non-wizard
                # The backend will use default model settings

                st.info(f"🏷️ **Taxonomy Generation:** Using {taxonomy_generator} model (Complete Mode)")

                # Prepare taxonomy generation request with complete processing mode
                # Do NOT pass "model" - let the backend use default settings (same as non-wizard)
                taxonomy_request = {
                    "project_id": project_id,
                    "name": f"Auto-generated taxonomy for project {project_id}",
                    "documents": selected_document_ids,
                    "depth": 3,
                    "generator": taxonomy_generator,
                    "domain": "general",
                    "batch_size": 10,
                    "specificity_level": 1,
                    "processing_mode": "complete"  # Use complete mode for full chunk coverage
                }

                taxonomy_response = api_client.post("/api/v1/taxonomy/generate", data=taxonomy_request)

                if taxonomy_response:
                    # Store the generated taxonomy in wizard state
                    wizard_state.update_step_data("taxonomy_selection", "selected_taxonomy", taxonomy_response)
                    st.success(f"Taxonomy '{taxonomy_response.get('name', 'Generated')}' generated successfully!")
                    st.rerun()
                else:
                    st.error("Failed to generate taxonomy")
                    return

            except APIError as e:
                st.error(f"Failed to generate taxonomy: {e}")
            except Exception as e:
                st.error(f"Error generating taxonomy: {e}")

def render_generation_params():
    """Render the generation parameters step - simplified version for wizard."""
    st.header("🔧 Step 5: Generation Parameters")

    # Check prerequisites - taxonomy is optional, will fallback to chunks
    project_data = wizard_state.get_step_data("project_selection")
    processing_data = wizard_state.get_step_data("processing_config")

    if not project_data.get("project_id"):
        st.error("❌ Please complete Project Selection first.")
        st.info("You need to select a project before configuring generation parameters.")
        return

    if not processing_data.get("selected_document_ids"):
        st.error("❌ Please complete Processing Configuration first.")
        st.info("You need to configure processing options and select documents before setting generation parameters.")
        return

    st.markdown("Configure the dataset generation parameters.")

    # Compact layout: Place all options except Custom Prompt side by side
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Dataset generation mode selection (required field) - match exact options from Dataset Generation GUI
        generation_mode = st.selectbox(
            "Dataset Generation Mode",
            options=["instruction following", "question and answer", "question", "answer", "summarization"],
            format_func=lambda x: {
                "instruction following": "Instruction Following",
                "question and answer": "Question and Answer",
                "question": "Questions Only",
                "answer": "Answers Only",
                "summarization": "Summarization"
            }.get(x, x),
            help="Choose the type of dataset to generate"
        )

    with col2:
        # Output format (required field)
        output_format = st.selectbox(
            "Output Format",
            options=["jsonl", "parquet"],
            help="Format for the generated dataset"
        )

    with col3:
        # Entries per chunk (required field) - changed to number_input for free text with no max
        entries_per_chunk = st.number_input(
            "Entries per Chunk",
            min_value=0,
            value=3,
            step=1,
            help="""Number of dataset entries to generate per document chunk.

• **1-10**: Fixed number of entries per chunk
• **0**: Let AI Decide - AI determines the appropriate number based on content richness"""
        )

    with col4:
        # Quality filter (required field)
        quality_threshold = st.slider(
            "Quality Filter",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Minimum quality score for generated entries (0.0 = accept all, 1.0 = highest quality only)"
        )

    # Custom prompt (optional field) - placed below in full width
    custom_prompt = st.text_area(
        "Custom Prompt",
        value="",
        height=100,
        placeholder="Optional: Provide a custom prompt template. Use {chunk} as placeholder for text content.",
        help="Optional custom prompt for dataset generation. Leave blank to use default prompts."
    )

    # Store parameters in wizard state
    params = {
        "generation_mode": generation_mode,
        "output_format": output_format,
        "entries_per_chunk": entries_per_chunk,
        "quality_threshold": quality_threshold,
        "custom_prompt": custom_prompt,
        # Fixed defaults as specified
        "chunks_per_batch": 1,
        "workers_per_batch": 1,
        "complexity_level": ""  # Leave blank unless mandatory
    }

    # Store params in session state for auto-save
    st.session_state.generation_params = params

    # Validate required parameters
    validation_errors = []
    if not generation_mode:
        validation_errors.append("Dataset Generation Mode is required")
    if not output_format:
        validation_errors.append("Output Format is required")
    if entries_per_chunk < 0:
        validation_errors.append("Entries per Chunk must be >= 0 (0 = AI decides)")
    if quality_threshold < 0.0 or quality_threshold > 1.0:
        validation_errors.append("Quality Filter must be between 0.0 and 1.0")

    if validation_errors:
        for error in validation_errors:
            st.error(f"⚠️ {error}")
        st.warning("Please correct the errors above before proceeding.")
    else:
        st.success("✅ Parameters are valid. You can proceed to the next step.")

def render_review_generate():
    """Render the review and generate step."""
    st.header("✅ Step 6: Review & Generate")

    # Check prerequisites
    project_data = wizard_state.get_step_data("project_selection")
    processing_data = wizard_state.get_step_data("processing_config")
    gen_data = wizard_state.get_step_data("generation_params")

    if not project_data.get("project_id"):
        st.error("❌ Please complete Project Selection first.")
        st.info("You need to select a project before generating datasets.")
        return

    if not processing_data.get("selected_document_ids"):
        st.error("❌ Please complete Processing Configuration first.")
        st.info("You need to configure processing options and select documents before generating datasets.")
        return

    if not gen_data.get("generation_mode"):
        st.error("❌ Please complete Generation Parameters first.")
        st.info("You need to configure generation parameters before creating datasets.")
        return

    st.markdown("Review your configuration and start dataset generation.")

    st.subheader("📋 Configuration Summary")

    # Display summary
    st.subheader("Configuration Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Project:**")
        project_data = wizard_state.get_step_data("project_selection")
        st.info(project_data.get("project_name", "Not selected"))

        st.markdown("**Documents:**")
        processing_data = wizard_state.get_step_data("processing_config")
        selected_document_ids = processing_data.get("selected_document_ids", [])
        selected_count = len(selected_document_ids)
        st.info(f"{selected_count} documents selected")

        st.markdown("**Processing:**")
        processing_data = wizard_state.get_step_data("processing_config")
        chunk_strategy = processing_data.get("chunk_strategy", "Not configured")
        parsing_model = processing_data.get("parsing_model", "Not configured")
        st.info(f"Strategy: {chunk_strategy} | Parser: {parsing_model}")

    with col2:
        st.markdown("**Taxonomy:**")
        taxonomy_data = wizard_state.get_step_data("taxonomy_selection")
        selected_tax = taxonomy_data.get("selected_taxonomy")
        if selected_tax:
            st.info(selected_tax.get("name", "Unknown"))
        else:
            st.warning("No taxonomy selected")

        st.markdown("**Generation:**")
        gen_data = wizard_state.get_step_data("generation_params")
        generation_mode = gen_data.get("generation_mode", "instruction following")
        mode_display = {
            "instruction following": "Instruction Following",
            "question and answer": "Question and Answer",
            "question": "Questions Only",
            "answer": "Answers Only",
            "summarization": "Summarization"
        }.get(generation_mode, generation_mode)

        # Determine data source for display
        taxonomy_data = wizard_state.get_step_data("taxonomy_selection")
        selected_tax = taxonomy_data.get("selected_taxonomy")
        data_source_display = "Taxonomy" if selected_tax else "Chunks Only"

        st.info(f"Mode: {mode_display} | Source: {data_source_display}")

        st.markdown("**Output:**")
        output_format = gen_data.get("output_format", "jsonl")
        st.info(f"Format: {output_format.upper()}")

    # Generate button
    st.divider()
    if st.button("🚀 Start Dataset Generation", type="primary", width='stretch'):
        # Collect all configuration data
        project_id = project_data.get("project_id")
        if not project_id:
            st.error("No project selected")
            return

        # Get taxonomy info
        taxonomy_id = None
        taxonomy_name = None
        if selected_tax:
            taxonomy_id = selected_tax.get("id")
            taxonomy_name = selected_tax.get("name")

        # Get processing configuration for AI models
        processing_data = wizard_state.get_step_data("processing_config")

        # Debug: Log the model selections
        parsing_model = processing_data.get("parsing_model", "gemini")
        chunking_model = processing_data.get("chunking_model", "gemini")
        classification_model = processing_data.get("classification_model", "gemini")
        generation_model = processing_data.get("generation_model", "gemini")

        st.info(f"🔧 **AI Models Selected:** Parsing: {parsing_model} | Chunking: {chunking_model} | Classification: {classification_model} | Generation: {generation_model}")

        # Get high-level parameters from wizard state
        gen_data = wizard_state.get_step_data("generation_params")
        high_level_params = gen_data.get("high_level_params", {})

        # First, save dataset parameters to the database
        dataset_params_request = {
            "project_id": project_id,
            "purpose": gen_data.get("dataset_purpose", "question_answering"),
            "audience": gen_data.get("target_audience", "intermediate"),
            "extraction_rules": "default",  # Default value
            "dataset_format": output_format,
            "question_style": "factual",  # Default value
            "answer_style": "comprehensive",  # Default value
            "negativity_ratio": 0.1,  # Default value
            "data_augmentation": "none",  # Default value
            # High-level prompt parameters
            "custom_audience": high_level_params.get("custom_audience", ""),
            "custom_purpose": high_level_params.get("custom_purpose", ""),
            "complexity_level": high_level_params.get("complexity_level", "intermediate"),
            "domain": high_level_params.get("domain", "general")
        }

        try:
            st.info("💾 Saving dataset parameters...")
            params_response = api_client.post("/api/v1/datasets/parameters", data=dataset_params_request)
            if params_response:
                st.success(f"✅ Dataset parameters saved (ID: {params_response.get('parameter_id')})")
            else:
                st.error("Failed to save dataset parameters")
                return
        except APIError as e:
            st.error(f"Failed to save dataset parameters: {e}")
            return
        except Exception as e:
            st.error(f"Error saving dataset parameters: {e}")
            return

        # Get custom prompt - only send to backend if user explicitly provided one
        # This allows the backend to use its own default prompts based on generation_mode
        # which properly respects datasets_per_chunk parameter
        custom_prompt = gen_data.get("custom_prompt", "").strip()
        generation_mode = gen_data.get("generation_mode", "instruction following")
        
        # Map wizard UI mode names to backend expected mode names
        if generation_mode == "instruction following":
            generation_mode = "instruction_following"

        # Get selected document IDs from processing config
        processing_data = wizard_state.get_step_data("processing_config")
        selected_document_ids = processing_data.get("selected_document_ids", [])

        # Determine data source: default to "Taxonomy" with fallback to "Chunks Only"
        data_source = "Taxonomy" if taxonomy_name else "Chunks Only"

        # Build the dataset generation request
        # Get entries per chunk and map to datasets_per_chunk
        entries_per_chunk = gen_data.get("entries_per_chunk", 3)
        
        generation_request = {
            "project_id": project_id,
            # Only include custom_prompt if user explicitly provided it
            # Otherwise, let the backend use its default prompts based on generation_mode
            # which properly respects the datasets_per_chunk parameter
            **({"custom_prompt": custom_prompt} if custom_prompt else {}),
            "generation_mode": generation_mode,
            "format_type": output_format,
            "data_source": data_source,  # Use Taxonomy if available, otherwise Chunks Only
            "datasets_per_chunk": entries_per_chunk,  # Map entries_per_chunk to datasets_per_chunk
            "concurrency": 1,  # Fixed default as per wizard specification
            "batch_size": 50,  # Default batch size
            "include_evaluation_sets": gen_data.get("include_evaluation_sets", False),
            "taxonomy_project": str(project_id) if taxonomy_name else None,  # Use project ID as taxonomy project
            "taxonomy_name": taxonomy_name,
            "output_dir": ".",
            "analyze_quality": True,
            "quality_threshold": gen_data.get("quality_threshold", 0.7),
            "enable_versioning": gen_data.get("enable_versioning", False),
            "run_benchmarks": gen_data.get("run_benchmarks", False),
            "benchmark_suite": "glue",
            # Pass generation_model as classification_model because the background job expects the provider there
            "classification_model": generation_model,
            # Pass the selected documents directly to the worker
            "document_ids": selected_document_ids,
            # Add required fields with defaults
            "prompt_name": "qa_pairs",
            "dataset_name": None,
            "extraction_file_id": None,
            "selected_categories": None,
            "only_validated": False
        }

        # Start dataset generation using unified job service (same as non-wizard)
        with st.spinner("Starting dataset generation..."):
            try:
                # Get user ID for job attribution
                try:
                    from ..state.session_state import session_state
                    user_id = session_state.user_id
                    if not user_id:
                        st.error("❌ You must be logged in to generate datasets.")
                        return
                except Exception as e:
                    st.error(f"❌ Authentication error: {e}")
                    return

                # Use unified job service for submission and monitoring
                result = unified_job_service.submit_and_monitor(
                    job_type=JobType.DATASET_GENERATION,
                    parameters=generation_request,
                    priority=JobPriority.NORMAL,
                    user_id=user_id,
                    title="Generating Dataset",
                    timeout=7200,  # 2 hours
                    poll_interval=5
                )

                if result.success:
                    dataset_result = result.result or {}
                    # Unwrap nested result if present (common pattern from worker response)
                    if "result" in dataset_result and isinstance(dataset_result["result"], dict):
                        dataset_result = dataset_result["result"]
                    
                    dataset_id = dataset_result.get("dataset_id", "Unknown")
                    entries_count = dataset_result.get("entries_count", 0)
                    
                    st.success(f"🎉 Dataset generation completed! Dataset ID: {dataset_id} with {entries_count} entries")
                    
                    # Store job info in wizard state - we use the API job_id for fetching files
                    # since filenames are generated using the job_id, not the version_id
                    wizard_state.update_step_data("review_generate", "job_id", result.job_id)
                    wizard_state.update_step_data("review_generate", "generation_started", True)
                    wizard_state.update_step_data("review_generate", "entries_count", entries_count)
                    
                    # Try to pre-fetch the download URL now so we can render st.download_button immediately
                    try:
                        response = api_client.get(f"/api/v1/datasets/{result.job_id}/download?project_id={project_id}")
                        if response and response.get('download_url'):
                            wizard_state.update_step_data("review_generate", "download_url", response['download_url'])
                    except Exception as e:
                        pass
                else:
                    st.error(f"❌ Dataset generation failed: {result.error_message}")

            except Exception as e:
                st.error(f"Error starting dataset generation: {e}")

    # Check if dataset was successfully generated in a previous interaction
    review_data = wizard_state.get_step_data("review_generate")
    if review_data and review_data.get("generation_started") and review_data.get("job_id"):
        job_id = review_data.get("job_id")
        entries_count = review_data.get("entries_count", 0)
        project_id = project_data.get("project_id")
        download_url = review_data.get("download_url")
        
        st.success(f"✅ Your dataset (job id: {job_id}) is ready with {entries_count} entries!")
        
        # If we didn't fetch the URL previously, try now
        if not download_url:
            try:
                response = api_client.get(f"/api/v1/datasets/{job_id}/download?project_id={project_id}")
                if response and response.get('download_url'):
                    download_url = response['download_url']
                    wizard_state.update_step_data("review_generate", "download_url", download_url)
            except Exception:
                pass
                
        if download_url:
            col1, col2 = st.columns(2)
            with col1:
                # Use a state variable to track if we should fetch
                if not st.session_state.get(f"fetch_{job_id}"):
                    if st.button("📥 Download Dataset", type="primary", use_container_width=True, help="Fetch and save the dataset file"):
                        st.session_state[f"fetch_{job_id}"] = True
                        st.rerun()
                
                if st.session_state.get(f"fetch_{job_id}"):
                    # Need to use caching to avoid downloading a large file multiple times
                    @st.cache_data(ttl=3600, show_spinner="Fetching dataset file from server...")
                    def get_cached_file(url):
                        return api_client.get_file(url)
                        
                    file_content = get_cached_file(download_url)
                    
                    if file_content:
                        filename = download_url.split("/")[-2]
                        st.download_button(
                            label=f"💾 Click to save '{filename}'",
                            data=file_content,
                            file_name=filename,
                            mime="application/octet-stream",
                            use_container_width=True,
                            key=f"save_{job_id}"
                        )
                    else:
                        st.error("Dataset file could not be downloaded from server.")
        else:
            st.warning("Could not locate the generated file automatically. Please check the 'Browse & Manage Datasets' tab.")

