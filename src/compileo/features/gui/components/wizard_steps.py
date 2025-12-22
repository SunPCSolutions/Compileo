"""
Wizard step components for the dataset creation workflow.
"""

import streamlit as st
import os
from typing import List, Dict, Any
import json
from ....core.logging import get_logger

logger = get_logger(__name__)

from src.compileo.features.gui.state.wizard_state import wizard_state
from src.compileo.features.gui.services.api_client import api_client, APIError
from src.compileo.features.gui.state.session_state import session_state

# Project selection is now in wizard/project_selection.py

def check_project_has_chunks(project_id: str) -> bool:
    """Check if a project already has chunks in the database."""
    try:
        resp = api_client.get(f"/api/v1/chunks/project/{project_id}?limit=1")
        if resp and resp.get("chunks") and len(resp.get("chunks", [])) > 0:
            return True
    except:
        pass
    return False

def render_combined_setup():
    """Render the combined setup step with project, models, upload, and chunking configuration."""
    st.header("⚙️ Step 2: Parse & Chunk & Taxonomy")

    # Check if we just completed a job (session state flag) - auto-refresh like document processing
    job_just_completed = hasattr(session_state, 'job_just_completed') and session_state.job_just_completed
    if job_just_completed:
        session_state.job_just_completed = False  # Clear the flag
        st.success("✅ Processing completed! Validation updated.")
        st.rerun()  # Refresh to update validation status

    # ============================================================================
    # PROJECT SELECTION
    # ============================================================================
    st.subheader("🏗️ Project")

    # Get available projects
    try:
        projects_response = api_client.get("/api/v1/projects?per_page=1000")
        if projects_response:
            projects = projects_response.get("projects", [])
        else:
            projects = []
    except Exception as e:
        st.error(f"Failed to load projects: {e}")
        projects = []

    if not projects:
        st.error("❌ No projects available. Please create a project first.")
        return

    # Project selection dropdown
    project_options = {f"{p['name']} (ID: {p['id']})": p['id'] for p in projects}
    selected_project_display = st.selectbox(
        "Select Project",
        options=list(project_options.keys()),
        help="Choose the project for dataset creation",
        key="combined_project_select"
    )
    selected_project_id = project_options[selected_project_display]

    # Store project selection
    wizard_state.update_step_data("project_selection", "project_id", selected_project_id)
    wizard_state.update_step_data("project_selection", "project_name", selected_project_display.split(" (ID: ")[0])

    st.success(f"✅ Selected project: {selected_project_display.split(' (ID: ')[0]}")

    # Check if chunks already exist
    if selected_project_id and check_project_has_chunks(selected_project_id):
        st.success("✨ This project already has processed chunks. You can upload more or proceed to the next step.")
        
    # ============================================================================
    # AI MODEL SELECTION
    # ============================================================================
    st.subheader("🤖 AI Models")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Determine supported parsers based on uploaded files (reactive filtering)
        all_parsers = ["grok", "gemini", "ollama", "openai", "huggingface", "unstructured", "pypdf", "novlm"]
        supported_parsers = all_parsers.copy()
        
        # Check uploaded files if they exist in session state (Streamlit retains them)
        check_files = st.session_state.get(f"combined_upload_{st.session_state.get('combined_upload_key', 0)}")
        if check_files:
            if not isinstance(check_files, list):
                check_files = [check_files]
                
            has_non_pdf = any(not f.name.lower().endswith('.pdf') for f in check_files)
            if has_non_pdf:
                # Restrict to parsers that handle non-PDFs
                # gemini and ollama support .txt/.md; unstructured/novlm support everything
                text_extensions = {'.txt', '.md', '.csv', '.json', '.xml'}
                has_complex_format = any(os.path.splitext(f.name)[1].lower() not in text_extensions and not f.name.lower().endswith('.pdf') for f in check_files)
                
                if has_complex_format:
                    supported_parsers = ["unstructured"]
                else:
                    supported_parsers = ["unstructured", "gemini", "ollama"]
        
        # Determine default index
        default_parser = "grok" if "grok" in supported_parsers else "novlm"
        try:
            default_index = supported_parsers.index(default_parser)
        except ValueError:
            default_index = 0

        parsing_model = st.selectbox(
            "Parsing",
            options=supported_parsers,
            index=default_index,
            help="AI model for document parsing. Available models filtered based on uploaded file types.",
            key="combined_parsing_model"
        )

    with col2:
        chunking_model = st.selectbox(
            "Chunking",
            options=["grok", "gemini", "openai", "ollama"],
            index=0,  # grok as default
            help="AI model for text chunking",
            key="combined_chunking_model"
        )

    with col3:
        classification_model = st.selectbox(
            "Classification",
            options=["grok", "gemini", "openai", "ollama"],
            index=0,  # grok as default
            help="AI model for document classification",
            key="combined_classification_model"
        )

    with col4:
        generation_model = st.selectbox(
            "Generation",
            options=["grok", "gemini", "openai", "ollama"],
            index=0,  # grok as default
            help="AI model for dataset generation",
            key="combined_generation_model"
        )

    # ============================================================================
    # DOCUMENT UPLOAD
    # ============================================================================
    st.subheader("📄 Documents")

    # Check for existing documents
    try:
        documents_response = api_client.get(f"/api/v1/documents?project_id={selected_project_id}")
        if documents_response:
            existing_documents = documents_response.get("documents", [])
        else:
            existing_documents = []
    except Exception as e:
        st.error(f"Error checking existing documents: {e}")
        existing_documents = []

    if existing_documents:
        st.info(f"📁 Project has {len(existing_documents)} existing document(s)")
        use_existing = st.checkbox("Use existing documents", value=True, key="use_existing_docs")
    else:
        use_existing = False
        st.info("📤 No existing documents. Upload new documents below.")

    # File uploader
    if 'combined_upload_key' not in st.session_state:
        st.session_state.combined_upload_key = 0

    uploaded_files = st.file_uploader(
        "Upload Documents",
        accept_multiple_files=True,
        type=["pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls", "txt", "md", "csv", "json", "xml"],
        help="Drag and drop files here • Limit 200MB per file • PDF, Office (DOCX, PPTX, XLSX), TXT, MD, CSV, JSON, XML",
        key=f"combined_upload_{st.session_state.combined_upload_key}"
    )

    # ============================================================================
    # CHUNKING CONFIGURATION
    # ============================================================================
    st.subheader("✂️ Chunking Strategy")

    chunk_strategy = st.selectbox(
        "Strategy",
        options=["character", "token", "semantic", "delimiter", "schema"],
        help="Method to split documents into chunks",
        key="combined_chunk_strategy"
    )

    # Initialize default values for all parameters
    chunk_size = 512
    overlap = 50
    semantic_prompt = ""
    common_delimiters = ["# (Headers)", "\n\n (Paragraphs)"]
    custom_delimiter = ""
    schema_json = ""

    # Strategy-specific parameters
    if chunk_strategy == "character":
        col_a, col_b = st.columns(2)
        with col_a:
            chunk_size = st.slider("Chunk Size", 100, 4000, 1000, 100, help="Characters per chunk", key="combined_char_size")
        with col_b:
            overlap = st.slider("Overlap", 0, 500, 100, 25, help="Overlap characters", key="combined_char_overlap")
    elif chunk_strategy == "token":
        col_a, col_b = st.columns(2)
        with col_a:
            chunk_size = st.slider("Chunk Size", 100, 2000, 512, 50, help="Tokens per chunk", key="combined_token_size")
        with col_b:
            overlap = st.slider("Overlap", 0, 200, 50, 10, help="Overlap tokens", key="combined_token_overlap")
    elif chunk_strategy == "semantic":
        semantic_prompt = st.text_area(
            "Prompt",
            height=80,
            placeholder="Custom prompt for semantic chunking...",
            help="Leave empty for default",
            key="combined_semantic_prompt"
        )
    elif chunk_strategy == "delimiter":
        st.write("**Delimiters**")
        col_a, col_b = st.columns(2)
        with col_a:
            common_delimiters = st.multiselect(
                "Common",
                ["# (Headers)", "--- (Rules)", "### (H3)", "\n\n (Paragraphs)", "\n (Lines)"],
                ["# (Headers)", "\n\n (Paragraphs)"],
                help="Select common delimiters",
                key="combined_common_delims"
            )
        with col_b:
            custom_delimiter = st.text_input(
                "Custom",
                placeholder="e.g., #, ---, <div>",
                help="Additional delimiters",
                key="combined_custom_delim"
            )
    else:  # schema
        schema_json = st.text_area(
            "Schema JSON",
            height=100,
            placeholder='{"rules": [{"type": "pattern", "value": "#"}], "combine": "any"}',
            help="JSON schema for chunking rules",
            key="combined_schema_json"
        )

    # ============================================================================
    # AUTO-SAVE CURRENT STATE (for navigation)
    # ============================================================================
    current_doc_ids = []
    if use_existing and existing_documents:
        current_doc_ids = [d["id"] for d in existing_documents]
        
    st.session_state.processing_config = {
        "parsing_model": parsing_model,
        "chunking_model": chunking_model,
        "classification_model": classification_model,
        "generation_model": generation_model,
        "chunk_strategy": chunk_strategy,
        "selected_document_ids": current_doc_ids,
        "chunk_job_id": st.session_state.get("processing_config", {}).get("chunk_job_id")
    }

    # ============================================================================
    # SUBMIT BUTTON
    # ============================================================================
    st.divider()

    # Validate inputs
    has_documents = (uploaded_files and len(uploaded_files) > 0) or (use_existing and existing_documents)
    can_proceed = has_documents

    if can_proceed:
        if st.button("🚀 Start Processing: Parse, Chunk, and Generate Taxonomy", type="primary", width='stretch'):
            st.info("🔄 Starting document processing workflow...")

            # Step 1: Upload files if any
            document_ids_to_process = []

            if uploaded_files:
                with st.spinner("📤 Uploading documents..."):
                    try:
                        files = []
                        for file in uploaded_files:
                            file_content = file.read()
                            files.append(("files", (file.name, file_content, file.type)))

                        data = {"project_id": str(selected_project_id)}
                        response = api_client.post("/api/v1/documents/upload", files=files, data=data)

                        if response and "uploaded_files" in response:
                            document_ids_to_process.extend([doc["id"] for doc in response["uploaded_files"]])
                            st.success(f"✅ Uploaded {len(response['uploaded_files'])} documents")
                        else:
                            st.error("❌ Failed to upload documents")
                            return
                    except Exception as e:
                        st.error(f"❌ Upload error: {e}")
                        return

            # Step 2: Add existing documents if selected
            if use_existing and existing_documents:
                document_ids_to_process.extend([doc["id"] for doc in existing_documents])

            if not document_ids_to_process:
                st.error("❌ No documents to process")
                return

            # Step 3: Submit parsing job
            with st.spinner("📄 Starting document parsing..."):
                try:
                    parse_request = {
                        "project_id": selected_project_id,
                        "document_ids": document_ids_to_process,
                        "parser": parsing_model,
                        "skip_parsing": False
                    }

                    parse_response = api_client.post("/api/v1/documents/process", data=parse_request)

                    if parse_response and parse_response.get("job_id"):
                        parse_job_id = parse_response["job_id"]
                        st.success(f"✅ Parsing job started: {parse_job_id}")

                        # Wait for parsing to complete
                        import time
                        max_wait = 300  # 5 minutes max
                        wait_count = 0

                        # Create placeholders for status updates
                        progress_placeholder = st.empty()
                        status_placeholder = st.empty()

                        while wait_count < max_wait:
                            try:
                                status_response = api_client.get(f"/api/v1/jobs/status/{parse_job_id}")
                                if status_response:
                                    status = status_response.get("status", "unknown")
                                    progress = status_response.get("progress", 0)
                                    current_step = status_response.get("current_step", "Processing...")

                                    # Update placeholders (replaces content instead of adding new rows)
                                    progress_placeholder.progress(progress / 100)
                                    status_placeholder.info(f"📄 Parsing {status}... ({wait_count}s)")

                                    if status == "completed":
                                        progress_placeholder.empty()
                                        status_placeholder.success("✅ Parsing completed!")
                                        break
                                    elif status == "failed":
                                        progress_placeholder.empty()
                                        status_placeholder.error("❌ Parsing failed")
                                        return
                                else:
                                    status_placeholder.warning("Could not check parsing status")
                            except Exception as e:
                                status_placeholder.warning(f"Status check error: {e}")

                            time.sleep(5)
                            wait_count += 5

                        # Clear placeholders if timeout
                        if wait_count >= max_wait:
                            progress_placeholder.empty()
                            status_placeholder.warning("⚠️ Parsing is taking longer than expected. You can proceed and check status later.")

                        if wait_count >= max_wait:
                            st.warning("⚠️ Parsing is taking longer than expected. You can proceed and check status later.")
                    else:
                        st.error("❌ Failed to start parsing job")
                        return

                except Exception as e:
                    st.error(f"❌ Parsing error: {e}")
                    return

            # Step 4: Submit chunking job
            with st.spinner("✂️ Starting document chunking..."):
                try:
                    # Prepare chunking parameters
                    chunk_params = {
                        "project_id": selected_project_id,
                        "document_ids": document_ids_to_process,
                        "chunker": chunking_model,
                        "chunk_strategy": chunk_strategy,
                        "skip_parsing": True  # Documents already parsed
                    }

                    # Add strategy-specific parameters
                    if chunk_strategy == "character":
                        chunk_params.update({
                            "chunk_size": chunk_size,
                            "overlap": overlap
                        })
                    elif chunk_strategy == "token":
                        chunk_params.update({
                            "chunk_size": chunk_size,
                            "overlap": overlap
                        })
                    elif chunk_strategy == "semantic":
                        chunk_params["semantic_prompt"] = semantic_prompt or ""
                    elif chunk_strategy == "delimiter":
                        delimiters = []
                        # Process common delimiters
                        delim_map = {
                            "# (Headers)": "#",
                            "--- (Rules)": "---",
                            "### (H3)": "###",
                            "\n\n (Paragraphs)": "\n\n",
                            "\n (Lines)": "\n"
                        }
                        for delim in common_delimiters:
                            if delim in delim_map:
                                delimiters.append(delim_map[delim])

                        # Add custom delimiters
                        if custom_delimiter.strip():
                            custom_list = [d.strip() for d in custom_delimiter.split(",") if d.strip()]
                            delimiters.extend(custom_list)

                        chunk_params["delimiters"] = delimiters
                    else:  # schema
                        # Validate schema JSON
                        if not schema_json.strip():
                            st.error("❌ Schema JSON is required for schema chunking")
                            return

                        # Validate JSON
                        try:
                            json.loads(schema_json)
                            chunk_params["schema_definition"] = schema_json
                        except json.JSONDecodeError as e:
                            st.error(f"❌ Invalid JSON schema: {e}")
                            return

                    # Submit chunking job
                    chunk_response = api_client.post("/api/v1/documents/process", data=chunk_params)

                    if chunk_response and chunk_response.get("job_id"):
                        chunk_job_id = chunk_response["job_id"]
                        st.success(f"✅ Chunking job started: {chunk_job_id}")

                        # Store configuration in both session state (for wizard.py commit) and wizard state
                        config_data = {
                            "parsing_model": parsing_model,
                            "chunking_model": chunking_model,
                            "classification_model": classification_model,
                            "generation_model": generation_model,
                            "chunk_strategy": chunk_strategy,
                            "selected_document_ids": document_ids_to_process,
                            "chunk_job_id": chunk_job_id
                        }
                        st.session_state.processing_config = config_data
                        wizard_state.set_step_data("processing_config", config_data)

                        # Wait for chunking to complete
                        wait_count = 0
                        while wait_count < max_wait:
                            try:
                                status_response = api_client.get(f"/api/v1/jobs/status/{chunk_job_id}")
                                if status_response:
                                    status = status_response.get("status", "unknown")
                                    progress = status_response.get("progress", 0)
                                    progress_placeholder.progress(progress / 100)
                                    status_placeholder.info(f"✂️ Chunking {status}... ({wait_count}s)")

                                    if status == "completed":
                                        status_placeholder.success("✅ Chunking completed!")
                                        break
                                    elif status == "failed":
                                        status_placeholder.error("❌ Chunking failed")
                                        return
                                else:
                                    status_placeholder.warning("Could not check chunking status")
                            except Exception as e:
                                status_placeholder.warning(f"Status check error: {e}")

                            time.sleep(5)
                            wait_count += 5

                        # Step 5: Automatic Taxonomy Generation
                        with st.spinner("🏷️ Automatically generating taxonomy..."):
                            try:
                                taxonomy_request = {
                                    "project_id": selected_project_id,
                                    "name": f"Wizard Taxonomy {selected_project_id}",
                                    "documents": document_ids_to_process,
                                    "depth": None,  # Liberated: Let AI decide depth
                                    "generator": classification_model,
                                    "domain": "general",
                                    "sample_size": 5,
                                    "mode": "complete",
                                    "category_limits": None  # Liberated: Let AI decide category limits
                                }
                                taxonomy_response = api_client.post("/api/v1/taxonomy/generate", data=taxonomy_request)
                                if taxonomy_response:
                                    wizard_state.update_step_data("taxonomy_selection", "selected_taxonomy", taxonomy_response)
                                    st.success("🎉 Processing complete! Documents parsed, chunked, and taxonomy generated.")
                                else:
                                    st.warning("⚠️ Processing complete, but automatic taxonomy generation failed. You can select one manually.")
                            except Exception as e:
                                st.warning(f"⚠️ Automatic taxonomy generation failed: {e}. You can select or generate one below.")

                        # Set flag for auto-refresh
                        session_state.job_just_completed = True

                        # Reset uploader
                        st.session_state.combined_upload_key += 1
                        st.rerun()

                    else:
                        st.error("❌ Failed to start chunking job")

                except Exception as e:
                    st.error(f"❌ Chunking error: {e}")

    else:
        st.warning("⚠️ Please upload documents or select existing documents to proceed.")
        st.button("🚀 Start Processing: Parse & Chunk Documents", disabled=True, width='stretch')



def render_generation_params():
    """Render the generation parameters step - simplified version for wizard."""
    st.header("🔧 Step 4: Generation Parameters")

    # Check prerequisites - taxonomy is optional, will fallback to chunks
    project_data = wizard_state.get_step_data("project_selection")
    processing_data = wizard_state.get_step_data("processing_config")

    if not project_data.get("project_id"):
        st.error("❌ Please complete Project Selection first.")
        st.info("You need to select a project before configuring generation parameters.")
        return

    # Verify the project still exists
    project_id = project_data.get("project_id")
    try:
        projects_response = api_client.get("/api/v1/projects")
        if projects_response:
            projects = projects_response.get("projects", [])
            project_exists = any(p["id"] == project_id for p in projects)
            if not project_exists:
                st.error("❌ The selected project no longer exists. Please go back to Step 1 to select a different project.")
                # Clear the invalid project from wizard state
                wizard_state.update_step_data("project_selection", "project_id", None)
                wizard_state.update_step_data("project_selection", "project_name", None)
                return
    except Exception as e:
        st.warning(f"Could not verify project exists: {e}")

    # Robust check for selected documents - fallback to all project documents if state is missing
    selected_document_ids = processing_data.get("selected_document_ids", [])
    if not selected_document_ids and project_id:
        try:
            # Attempt to recover documents from the project if wizard state is empty
            docs_resp = api_client.get(f"/api/v1/documents?project_id={project_id}")
            if docs_resp and "documents" in docs_resp:
                # Filter documents that have chunks
                for d in docs_resp["documents"]:
                    chunks_resp = api_client.get(f"/api/v1/chunks/document/{d['id']}")
                    if chunks_resp and chunks_resp.get("chunks"):
                        selected_document_ids.append(d["id"])
                
                if selected_document_ids:
                    # Update wizard state with recovered documents
                    wizard_state.update_step_data("processing_config", "selected_document_ids", selected_document_ids)
                    st.info(f"🔄 Recovered {len(selected_document_ids)} processed documents from project state.")
        except Exception as e:
            logger.debug(f"State recovery failed: {e}")

    if not selected_document_ids:
        st.error("❌ Please complete Processing Configuration first.")
        st.info("You need to configure processing options and select documents before setting generation parameters.")
        return

    st.markdown("Configure the dataset generation parameters.")
    
    # Load existing parameters if not in state
    gen_data = wizard_state.get_step_data("generation_params")
    if (not gen_data.get("generation_mode") or wizard_state.needs_refresh) and project_id:
        try:
            resp = api_client.get(f"/api/v1/datasets/parameters/{project_id}")
            if resp and resp.get("parameters"):
                p = resp["parameters"]
                gen_data = {
                    "generation_mode": p.get("generation_mode") or "instruction following",
                    "output_format": p.get("dataset_format") or "jsonl",
                    "entries_per_chunk": gen_data.get("entries_per_chunk", 3),
                    "quality_threshold": gen_data.get("quality_threshold", 0.7),
                    "custom_prompt": gen_data.get("custom_prompt", ""),
                    "custom_audience": p.get("custom_audience", ""),
                    "custom_purpose": p.get("custom_purpose", ""),
                    "complexity_level": p.get("complexity_level", "intermediate"),
                    "domain": p.get("domain", "")
                }
                wizard_state.set_step_data("generation_params", gen_data)
                st.info("🔄 Loaded existing generation parameters from project.")
        except:
            pass

    # Compact layout: Place all options except Custom Prompt side by side
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Dataset generation mode selection (required field) - match exact options from Dataset Generation GUI
        mode_options = ["instruction following", "question and answer", "question", "answer", "summarization"]
        default_mode = gen_data.get("generation_mode", "instruction following")
        mode_index = mode_options.index(default_mode) if default_mode in mode_options else 0
        
        generation_mode = st.selectbox(
            "Dataset Generation Mode",
            options=mode_options,
            index=mode_index,
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
        st.markdown("**📄 Output Format**")
        # Get available formats dynamically including plugins
        try:
            formats_response = api_client.get("/api/v1/plugins/dataset-formats")
            available_formats = formats_response.get("formats", ["jsonl", "parquet"]) if formats_response else ["jsonl", "parquet"]
        except Exception as e:
            available_formats = ["jsonl", "parquet"]

        default_format = gen_data.get("output_format", "jsonl")
        format_index = available_formats.index(default_format) if default_format in available_formats else 0
        
        output_format = st.selectbox(
            "Format",
            options=available_formats,
            index=format_index,
            help="Output file format: JSONL for line-delimited JSON (common for AI training), Parquet for compressed columnar storage, or plugin formats like 'anki' for specialized outputs."
        )

    with col3:
        # Entries per chunk (required field) - changed to number_input for free text with no max
        entries_per_chunk = st.number_input(
            "Entries per Chunk",
            min_value=1,
            value=gen_data.get("entries_per_chunk", 3),
            step=1,
            help="Number of dataset entries to generate per document chunk"
        )

    with col4:
        # Quality filter (required field)
        quality_threshold = st.slider(
            "Quality Filter",
            min_value=0.0,
            max_value=1.0,
            value=gen_data.get("quality_threshold", 0.7),
            step=0.1,
            help="Minimum quality score for generated entries (0.0 = accept all, 1.0 = highest quality only)"
        )

    # High-Level Prompts - New Section aligned with Core Dataset Generation
    st.divider()
    st.subheader("📝 High-Level Prompts")
    
    # Fetch defaults from API
    try:
        hl_config = api_client.get("/api/v1/datasets/config/high-level-prompts")
        if hl_config:
            audience_opts = hl_config.get("audience_defaults", [])
            purpose_opts = hl_config.get("purpose_defaults", [])
            complexity_opts = hl_config.get("complexity_options", ["beginner", "intermediate", "advanced", "expert"])
            domain_opts = hl_config.get("domain_defaults", [])
        else:
            # Fallback defaults
            audience_opts = ["healthcare professionals", "students", "patients", "researchers"]
            purpose_opts = ["patient education", "research", "medical education", "clinical decision support"]
            complexity_opts = ["beginner", "intermediate", "advanced", "expert"]
            domain_opts = ["general"]
    except Exception:
        audience_opts = []
        purpose_opts = []
        complexity_opts = ["beginner", "intermediate", "advanced", "expert"]
        domain_opts = []

    col_hl1, col_hl2 = st.columns(2)

    with col_hl1:
        # Audience free text field
        custom_audience = st.text_input(
            "Target Audience",
            value=gen_data.get("custom_audience", ""),
            placeholder="e.g., policy makers, engineers",
            help="Specify the target audience."
        )

        # Purpose free text field
        custom_purpose = st.text_input(
            "Purpose",
            value=gen_data.get("custom_purpose", ""),
            placeholder="e.g., sentiment analysis, entity extraction",
            help="Describe how the dataset will be used."
        )

    with col_hl2:
        default_complexity = gen_data.get("complexity_level", "intermediate")
        complexity_index = complexity_opts.index(default_complexity) if default_complexity in complexity_opts else 1
        
        complexity_level = st.selectbox(
            "Complexity Level",
            options=complexity_opts,
            index=complexity_index,
            help="Content complexity level."
        )

        # Domain free text field
        domain = st.text_input(
            "Domain",
            value=gen_data.get("domain", ""),
            placeholder="e.g., finance, law",
            help="Specific subject domain."
        )

    # Custom prompt (optional field) - placed below in full width
    st.divider()
    st.markdown("**✏️ Custom Prompt** *(Optional)*")
    custom_prompt = st.text_area(
        "Enter your custom prompt for dataset generation",
        value=gen_data.get("custom_prompt", ""),
        placeholder="e.g., Generate clinical questions that would help medical students prepare for board exams...",
        height=100,
        help="Override the default generation prompts. Use {chunk} to reference source text. Leave empty for system-generated prompts.",
        key="wizard_custom_prompt"
    )

    # Store parameters in wizard state
    params = {
        "generation_mode": generation_mode,
        "output_format": output_format,
        "entries_per_chunk": entries_per_chunk,
        "quality_threshold": quality_threshold,
        "custom_prompt": custom_prompt,
        "custom_audience": custom_audience,
        "custom_purpose": custom_purpose,
        "complexity_level": complexity_level,
        "domain": domain,
        # Default batch settings
        "chunks_per_batch": 50,
        "workers_per_batch": 2
    }

    # Store params in session state for auto-save
    st.session_state.generation_params = params

    # Validate required parameters
    validation_errors = []
    if not generation_mode:
        validation_errors.append("Dataset Generation Mode is required")
    if not output_format:
        validation_errors.append("Output Format is required")
    if entries_per_chunk < 1:
        validation_errors.append("Entries per Chunk must be at least 1")
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
    st.header("✅ Step 5: Review & Generate")

    # Check prerequisites
    project_data = wizard_state.get_step_data("project_selection")
    processing_data = wizard_state.get_step_data("processing_config")
    gen_data = wizard_state.get_step_data("generation_params")

    if not project_data.get("project_id"):
        st.error("❌ Please complete Project Selection first.")
        st.info("You need to select a project before generating datasets.")
        return

    # Verify the project still exists
    project_id = project_data.get("project_id")
    try:
        projects_response = api_client.get("/api/v1/projects")
        if projects_response:
            projects = projects_response.get("projects", [])
            project_exists = any(p["id"] == project_id for p in projects)
            if not project_exists:
                st.error("❌ The selected project no longer exists. Please go back to Step 1 to select a different project.")
                # Clear the invalid project from wizard state
                wizard_state.update_step_data("project_selection", "project_id", None)
                wizard_state.update_step_data("project_selection", "project_name", None)
                return
    except Exception as e:
        st.warning(f"Could not verify project exists: {e}")

    # Robust check for selected documents
    selected_document_ids = processing_data.get("selected_document_ids", [])
    
    # Recovery fallback for documents
    if not selected_document_ids and project_id:
        try:
            docs_resp = api_client.get(f"/api/v1/documents?project_id={project_id}")
            if docs_resp and "documents" in docs_resp:
                for d in docs_resp["documents"]:
                    chunks_resp = api_client.get(f"/api/v1/chunks/document/{d['id']}")
                    if chunks_resp and chunks_resp.get("chunks"):
                        selected_document_ids.append(d["id"])
                if selected_document_ids:
                    wizard_state.update_step_data("processing_config", "selected_document_ids", selected_document_ids)
                    processing_data["selected_document_ids"] = selected_document_ids
        except:
            pass

    if not selected_document_ids:
        st.error("❌ Please complete Processing Configuration first.")
        st.info("You need to configure processing options and select documents before generating datasets.")
        return

    # Recovery fallback for generation params
    if not gen_data.get("generation_mode"):
        # Try to recover from session state
        if "generation_params" in st.session_state:
            gen_data = st.session_state.generation_params
            wizard_state.set_step_data("generation_params", gen_data)
        else:
            st.error("❌ Please complete Generation Parameters first.")
            st.info("You need to configure generation parameters before creating datasets.")
            return

    st.markdown("Review your configuration and start dataset generation.")

    st.subheader("📋 Configuration Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Project:**")
        st.info(project_data.get("project_name", "Not selected"))

        st.markdown("**Documents:**")
        selected_count = len(selected_document_ids)
        st.info(f"{selected_count} documents selected")

        st.markdown("**Processing:**")
        # Fallback to session state if wizard state is missing strategy
        chunk_strategy = processing_data.get("chunk_strategy")
        if not chunk_strategy and "processing_config" in st.session_state:
            chunk_strategy = st.session_state.processing_config.get("chunk_strategy")
        if not chunk_strategy and project_id:
            try:
                # Query chunks table to find what strategy was actually used
                resp = api_client.get(f"/api/v1/chunks/project/{project_id}?limit=1")
                if resp and resp.get("chunks"):
                    chunk_strategy = resp["chunks"][0].get("chunk_strategy")
                    if chunk_strategy:
                        wizard_state.update_step_data("processing_config", "chunk_strategy", chunk_strategy)
            except:
                pass

        if not chunk_strategy:
            chunk_strategy = "character" # Default fallback
            
        parsing_model = processing_data.get("parsing_model") or st.session_state.get("processing_config", {}).get("parsing_model", "grok")
        chunking_model = processing_data.get("chunking_model") or st.session_state.get("processing_config", {}).get("chunking_model", "grok")
        
        st.info(f"Strategy: {chunk_strategy}")
        st.info(f"Parser: {parsing_model} | Chunker: {chunking_model}")

    with col2:
        st.markdown("**Taxonomy:**")
        taxonomy_data = wizard_state.get_step_data("taxonomy_selection")
        selected_tax = taxonomy_data.get("selected_taxonomy")
        
        # Recovery fallback for taxonomy
        if not selected_tax and project_id:
            try:
                tax_resp = api_client.get(f"/api/v1/taxonomy?project_id={project_id}")
                if tax_resp and tax_resp.get("taxonomies"):
                    # Use the most recent taxonomy
                    selected_tax = tax_resp["taxonomies"][-1]
                    wizard_state.update_step_data("taxonomy_selection", "selected_taxonomy", selected_tax)
            except:
                pass

        if selected_tax:
            st.info(selected_tax.get("name", "Unknown"))
        else:
            st.warning("No taxonomy selected")

        st.markdown("**Generation:**")
        generation_mode = gen_data.get("generation_mode", "instruction following")
        mode_display = {
            "instruction following": "Instruction Following",
            "question and answer": "Question and Answer",
            "question": "Questions Only",
            "answer": "Answers Only",
            "summarization": "Summarization"
        }.get(generation_mode, generation_mode)

        # Determine data source for display
        data_source_display = "Taxonomy-aware Chunks" if selected_tax else "Chunks Only"

        st.info(f"Mode: {mode_display} | Source: {data_source_display}")
        
        classification_model = processing_data.get("classification_model") or st.session_state.get("processing_config", {}).get("classification_model", "grok")
        generation_model = processing_data.get("generation_model") or st.session_state.get("processing_config", {}).get("generation_model", "grok")
        st.info(f"Classifier: {classification_model} | Generator: {generation_model}")

        st.markdown("**Output:**")
        output_format = gen_data.get("output_format", "jsonl")
        st.info(f"Format: {output_format.upper()}")

    # High-level prompt summary
    st.markdown("**📝 High-Level Parameters:**")
    col_hl_sum1, col_hl_sum2 = st.columns(2)
    with col_hl_sum1:
        st.info(f"Audience: {gen_data.get('custom_audience') or 'Not specified'}")
        st.info(f"Purpose: {gen_data.get('custom_purpose') or 'Not specified'}")
    with col_hl_sum2:
        st.info(f"Complexity: {gen_data.get('complexity_level') or 'intermediate'}")
        st.info(f"Domain: {gen_data.get('domain') or 'Not specified'}")

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
        parsing_model = processing_data.get("parsing_model") or st.session_state.get("processing_config", {}).get("parsing_model", "grok")
        chunking_model = processing_data.get("chunking_model") or st.session_state.get("processing_config", {}).get("chunking_model", "grok")
        classification_model = processing_data.get("classification_model") or st.session_state.get("processing_config", {}).get("classification_model", "grok")
        generation_model = processing_data.get("generation_model") or st.session_state.get("processing_config", {}).get("generation_model", "grok")

        st.info(f"🔧 **AI Models Selected:** Parsing: {parsing_model} | Chunking: {chunking_model} | Classification: {classification_model} | Generation: {generation_model}")

        # Get parameters from wizard state
        gen_data = wizard_state.get_step_data("generation_params")

        # First, save dataset parameters to the database
        dataset_params_request = {
            "project_id": project_id,
            "purpose": gen_data.get("custom_purpose", "Dataset Generation"),
            "audience": gen_data.get("custom_audience", "General"),
            "extraction_rules": "default",
            "dataset_format": output_format,
            "question_style": "factual",
            "answer_style": "comprehensive",
            "negativity_ratio": 0.1,
            "data_augmentation": "none",
            # High-level prompt parameters
            "custom_audience": gen_data.get("custom_audience", ""),
            "custom_purpose": gen_data.get("custom_purpose", ""),
            "complexity_level": gen_data.get("complexity_level", "intermediate"),
            "domain": gen_data.get("domain", "general")
        }

        try:
            st.info("💾 Saving dataset parameters...")
            params_response = api_client.post("/api/v1/datasets/parameters", data=dataset_params_request)
            param_id = params_response.get('parameter_id') if params_response else "Unknown"
            st.success(f"✅ Dataset parameters saved (ID: {param_id})")
        except APIError as e:
            st.error(f"Failed to save dataset parameters: {e}")
            return
        except Exception as e:
            st.error(f"Error saving dataset parameters: {e}")
            return

        # Get custom prompt, provide default if empty
        custom_prompt = gen_data.get("custom_prompt", "").strip()
        generation_mode = gen_data.get("generation_mode", "instruction following")

        # Provide default prompts based on generation mode if no custom prompt is specified
        if not custom_prompt:
            # Fetch defaults from API
            try:
                prompts_resp = api_client.get("/api/v1/datasets/config/default-prompts")
                if prompts_resp and "prompts" in prompts_resp:
                    default_prompts = prompts_resp["prompts"]
                else:
                    raise Exception("Failed to fetch default prompts")
            except Exception:
                # Fallback defaults
                default_prompts = {
                    "instruction following": "Generate an instruction-response pair based on the following text. Create a natural language instruction that could be given to an AI assistant, and provide the appropriate response.\n\nText: {chunk}\n\nInstruction:",
                    "question and answer": "Generate a question and answer pair based on the following text.\n\nText: {chunk}\n\nQuestion:",
                    "question": "Generate a question based on the following text.\n\nText: {chunk}\n\nQuestion:",
                    "answer": "Provide a comprehensive answer based on the following text.\n\nText: {chunk}\n\nAnswer:",
                    "summarization": "Analyze the following text and provide concise summaries for different aspects or sections. If the text contains clear sections with headers (like # Section Title), summarize each section individually. If no clear sections exist, identify the key topics or aspects within the text and provide a separate summary for each topic.\n\nText: {chunk}\n\nSummaries:"
                }
            custom_prompt = default_prompts.get(generation_mode, default_prompts.get("question and answer", ""))

        # Get selected document IDs from processing config
        processing_data = wizard_state.get_step_data("processing_config")
        selected_document_ids = processing_data.get("selected_document_ids", [])

        # Wizard always uses Taxonomy as data source
        if not taxonomy_name:
            st.error("❌ No taxonomy selected. Please complete Step 4: Taxonomy Selection.")
            return
        data_source = "Taxonomy"

        # Build the dataset generation request
        generation_request = {
            "project_id": project_id,
            "document_ids": selected_document_ids,  # Use selected documents
            "custom_prompt": custom_prompt,
            "generation_mode": generation_mode,
            "format_type": output_format,
            "data_source": data_source,
            "concurrency": gen_data.get("workers_per_batch", 2),
            "batch_size": gen_data.get("chunks_per_batch", 50),
            "include_evaluation_sets": gen_data.get("include_evaluation_sets", False),
            "taxonomy_project": str(project_id) if taxonomy_name else None,  # Use project ID as taxonomy project
            "taxonomy_name": taxonomy_name,
            "output_dir": ".",
            "analyze_quality": True,
            "quality_threshold": gen_data.get("quality_threshold", 0.7),
            "enable_versioning": gen_data.get("enable_versioning", False),
            "run_benchmarks": gen_data.get("run_benchmarks", False),
            "benchmark_suite": "glue",
            "parsing_model": parsing_model,
            "chunking_model": chunking_model,
            "classification_model": classification_model,
            "generation_model": generation_model,
            "custom_audience": gen_data.get("custom_audience", ""),
            "custom_purpose": gen_data.get("custom_purpose", ""),
            "complexity_level": gen_data.get("complexity_level", "intermediate"),
            "domain": gen_data.get("domain", "general"),
            "datasets_per_chunk": gen_data.get("entries_per_chunk", 3)
        }

        # Start dataset generation
        with st.spinner("Starting dataset generation..."):
            try:
                response = api_client.post("/api/v1/datasets/generate", data=generation_request)

                job_id = response.get("job_id") if response else None
                if job_id:
                    st.success(f"✅ Dataset generation started! Job ID: {job_id}")
                    est_duration = response.get('estimated_duration', 'Unknown') if response else "Unknown"
                    st.info(f"Estimated duration: {est_duration}")

                    # Store job info in wizard state
                    wizard_state.update_step_data("review_generate", "job_id", job_id)
                    wizard_state.update_step_data("review_generate", "generation_started", True)

                    # Show status monitoring
                    st.subheader("Generation Status")
                    status_placeholder = st.empty()

                    # Poll for status updates
                    import time
                    max_polls = 60  # 5 minutes max
                    poll_count = 0
                    final_status = "running"
                    final_status_response = {}

                    while poll_count < max_polls:
                        try:
                            status_response = api_client.get(f"/api/v1/datasets/generate/{job_id}/status")
                            status = status_response.get("status", "unknown") if status_response else "unknown"
                            progress = status_response.get("progress", 0) if status_response else 0
                            current_step = status_response.get("current_step", "Processing...") if status_response else "Processing..."

                            with status_placeholder.container():
                                st.progress(progress / 100)
                                st.info(f"Status: {status} | {current_step}")

                            if status in ["completed", "failed"]:
                                final_status = status
                                final_status_response = status_response
                                break

                        except Exception as e:
                            st.warning(f"Could not get status update: {e}")

                        time.sleep(5)  # Wait 5 seconds between polls
                        poll_count += 1

                    # Final status
                    if final_status == "completed":
                        result = final_status_response.get("result", {}) if final_status_response else {}
                        dataset_id = result.get("dataset_id", "Unknown")
                        entries_count = result.get("entries_count", 0)
                        st.success(f"🎉 Dataset generation completed! Dataset ID: {dataset_id} with {entries_count} entries")
                        
                        # Add Download Button
                        try:
                            # Construct download URL
                            api_base = api_client.base_url
                            download_url = f"{api_base}/api/v1/datasets/{dataset_id}/download"
                            
                            # Display download link
                            st.markdown(f"### 📥 Result")
                            st.success(f"Dataset generated successfully with {entries_count} entries.")
                            st.markdown(f"**[📥 Click here to download the generated dataset]({download_url})**")
                            
                        except Exception as e:
                            logger.error(f"Failed to setup download: {e}")
                            
                    elif final_status == "failed":
                        error = final_status_response.get("error", "Unknown error") if final_status_response else "Unknown error"
                        st.error(f"❌ Dataset generation failed: {error}")
                    else:
                        st.warning("⚠️ Dataset generation is still running. Check back later for results.")

                else:
                    st.error("Failed to start dataset generation - no job ID received")

            except APIError as e:
                st.error(f"Failed to start dataset generation: {e}")
            except Exception as e:
                st.error(f"Error starting dataset generation: {e}")

def render_edit_taxonomy_step():
    """Render a simplified taxonomy selection and editing step for the wizard."""
    st.header("🏗️ Step 3: Edit Taxonomy")
    
    project_data = wizard_state.get_step_data("project_selection")
    project_id = project_data.get("project_id")
    
    if not project_id:
        st.error("Please select a project first.")
        return

    # ============================================================================
    # 1. TAXONOMY SELECTION
    # ============================================================================
    st.subheader("🏷️ Select Taxonomy")
    
    # Get available taxonomies for this project
    try:
        taxonomies_response = api_client.get(f"/api/v1/taxonomy?project_id={project_id}")
        taxonomies = taxonomies_response.get("taxonomies", []) if taxonomies_response else []
    except Exception as e:
        st.error(f"Failed to load taxonomies: {e}")
        taxonomies = []

    selected_tax = None
    if taxonomies:
        taxonomy_options = {f"{t['name']} (ID: {t['id']})": t for t in taxonomies}
        
        # Determine current selection for default
        current_step_data = wizard_state.get_step_data("taxonomy_selection")
        selected_tax_obj = current_step_data.get("selected_taxonomy") if current_step_data else None
        current_sel_id = selected_tax_obj.get("id") if selected_tax_obj else None
        
        default_index = 0
        for i, t in enumerate(taxonomies):
            if t['id'] == current_sel_id:
                default_index = i
                break
        
        # UI layout with buttons next to selectbox
        st.write("Choose a taxonomy to work with")
        col_sel, col_reg, col_del, col_spacer = st.columns([0.7, 0.05, 0.05, 0.2])
        
        with col_sel:
            selected_tax_key = st.selectbox(
                "Choose a taxonomy to work with",
                options=list(taxonomy_options.keys()),
                index=default_index,
                key="wizard_taxonomy_selectbox",
                label_visibility="collapsed"
            )
            selected_tax = taxonomy_options[selected_tax_key] if selected_tax_key else None
        
        with col_reg:
            if st.button("🔄", help="Regenerate this taxonomy (Overwrites original)", key="wizard_regen_tax_btn"):
                if selected_tax:
                    # Logic to regenerate
                    try:
                        # Load full data to get source docs
                        resp = api_client.get(f"/api/v1/taxonomy/{selected_tax['id']}")
                        metadata = resp.get('metadata', {}) if resp else {}
                        doc_ids = metadata.get('source_document_ids')
                        
                        if not doc_ids:
                            # Fallback to current wizard docs
                            proc_data = wizard_state.get_step_data("processing_config")
                            doc_ids = proc_data.get("selected_document_ids")
                            
                        if not doc_ids:
                            # Final fallback: get all documents in the project
                            try:
                                docs_resp = api_client.get(f"/api/v1/documents?project_id={selected_tax['project_id']}")
                                if docs_resp and "documents" in docs_resp:
                                    doc_ids = [d["id"] for d in docs_resp["documents"]]
                                    logger.debug(f"Regenerating using all {len(doc_ids)} project documents as fallback")
                            except Exception as e:
                                logger.debug(f"Failed fallback document fetch: {e}")
                            
                        if not doc_ids:
                            st.error("No source documents found to regenerate from.")
                        else:
                            with st.spinner("🔄 Regenerating..."):
                                taxonomy_request = {
                                    "project_id": selected_tax['project_id'],
                                    "name": selected_tax['name'],
                                    "documents": doc_ids,
                                    "depth": None,
                                    "generator": "grok", # Default to grok
                                    "domain": metadata.get('domain', 'general'),
                                    "sample_size": 100,
                                    "mode": "complete",
                                    "category_limits": None
                                }
                                new_resp = api_client.post("/api/v1/taxonomy/generate", data=taxonomy_request)
                                if new_resp and 'id' in new_resp:
                                    # Delete old
                                    api_client.delete(f"/api/v1/taxonomy/{selected_tax['id']}")
                                    # Update state
                                    wizard_state.update_step_data("taxonomy_selection", "selected_taxonomy", new_resp)
                                    if "unified_taxonomy" in st.session_state:
                                        del st.session_state.unified_taxonomy
                                    st.success("✅ Taxonomy regenerated!")
                                    import time
                                    time.sleep(1)
                                    st.rerun()
                    except Exception as e:
                        st.error(f"Regeneration failed: {e}")

        with col_del:
            if st.button("🗑️", help="Delete this taxonomy", key="wizard_delete_tax_btn"):
                if selected_tax:
                    try:
                        with st.spinner("🗑️ Deleting..."):
                            api_client.delete(f"/api/v1/taxonomy/{selected_tax['id']}")
                            wizard_state.update_step_data("taxonomy_selection", "selected_taxonomy", None)
                            if "unified_taxonomy" in st.session_state:
                                del st.session_state.unified_taxonomy
                            st.success("✅ Taxonomy deleted!")
                            import time
                            time.sleep(1)
                            st.rerun()
                    except Exception as e:
                        st.error(f"Deletion failed: {e}")

        if selected_tax:
            # Update wizard state and initialize editor if changed
            if current_sel_id != selected_tax['id']:
                wizard_state.update_step_data("taxonomy_selection", "selected_taxonomy", selected_tax)
                if "unified_taxonomy" in st.session_state:
                    del st.session_state.unified_taxonomy
                st.rerun()
    else:
        st.info("No taxonomies available. Please complete Step 2 to generate one automatically.")
        return

    # ============================================================================
    # 2. REACTIVE TAXONOMY EDITOR (Using Fragments)
    # ============================================================================
    if taxonomies and selected_tax:
        st.divider()
        render_reactive_taxonomy_editor(selected_tax)

@st.fragment()
def render_reactive_taxonomy_editor(selected_tax: Dict[str, Any]):
    """Fragment-based taxonomy editor for immediate GUI reactivity."""
    st.subheader("🏗️ Refine Taxonomy Structure")
    
    # Pre-populate unified_taxonomy in session state
    if "unified_taxonomy" not in st.session_state:
        try:
            response = api_client.get(f"/api/v1/taxonomy/{selected_tax['id']}")
            full_tax_data = response.get('taxonomy', {}) if response else {}
            
            st.session_state.unified_taxonomy = {
                "name": full_tax_data.get('name', selected_tax.get('name', 'Taxonomy')),
                "description": full_tax_data.get('description', selected_tax.get('description', '')),
                "project_id": selected_tax.get('project_id'),
                "creation_mode": "hybrid",
                "depth": full_tax_data.get('depth', 3),
                "categories": full_tax_data.get('children', []),
                "loaded_taxonomy_id": selected_tax['id'],
                "ai_config": {
                    "generator": "grok",
                    "domain": "general",
                    "batch_size": 10,
                    "specificity_level": 1,
                    "category_limits": None,  # Liberated for wizard only
                    "selected_documents": []
                }
            }
        except Exception as e:
            st.error(f"Failed to load taxonomy details: {e}")
            return

    taxonomy = st.session_state.unified_taxonomy

    # Taxonomy Name - User can rename it
    taxonomy_name_input = st.text_input(
        "Taxonomy Name",
        value=taxonomy["name"],
        help="You can rename your taxonomy here.",
        key="wizard_taxonomy_name_input"
    )
    
    if taxonomy_name_input != taxonomy["name"]:
        taxonomy["name"] = taxonomy_name_input
        from src.compileo.features.gui.utils.taxonomy_utils import save_unified_taxonomy
        # Auto-save rename silently
        save_unified_taxonomy(taxonomy, silent=True)
        # We still set needs_refresh so that Step 5 summary is correct when user navigates
        wizard_state.needs_refresh = True
        # Fragment will auto-rerun to update its local display
    
    st.write("**Edit Categories**")
    
    # Add top-level category button
    if st.button("➕ Add Top-Level Category", type="secondary", width='stretch'):
        taxonomy["categories"].append({
            "name": "", "description": "", "confidence_threshold": 0.8, "children": [],
            "depth_limit": taxonomy["depth"] - 1
        })
        from src.compileo.features.gui.utils.taxonomy_utils import save_unified_taxonomy
        # Auto-save structural change silently
        save_unified_taxonomy(taxonomy, silent=True)
        # Set refresh flag and rerun to ensure new category is visible
        wizard_state.needs_refresh = True
        st.rerun()

    # Render collapsible taxonomy tree builder
    if taxonomy["categories"]:
        # Use a stable but unique key prefix based on the taxonomy structure
        # This ensures that when a category is removed, all remaining categories
        # get fresh widget identities, preventing Streamlit from using stale states.
        import hashlib
        tree_state = json.dumps(taxonomy["categories"], sort_keys=True)
        state_hash = hashlib.md5(tree_state.encode()).hexdigest()[:8]
        
        from src.compileo.features.gui.components.taxonomy_components import render_collapsible_category_builder
        for i, category in enumerate(taxonomy["categories"]):
            # Path MUST start with 'cat' for taxonomy_utils functions to work
            render_collapsible_category_builder(
                category, 0, taxonomy["depth"], f"cat_{i}", taxonomy,
                default_expanded=False, key_prefix=f"wiz_{state_hash}"
            )
    else:
        st.info("Your taxonomy is empty. Add categories above.")

    # Explicit Apply Changes Button
    if st.button("💾 Apply & Finish Editing", type="primary", width='stretch'):
        from src.compileo.features.gui.utils.taxonomy_utils import save_unified_taxonomy
        save_unified_taxonomy(taxonomy)
        
        wizard_state.update_step_data("taxonomy_selection", "selected_taxonomy", {
            "id": taxonomy.get("loaded_taxonomy_id"),
            "name": taxonomy["name"],
            "description": taxonomy["description"],
            "categories_count": len(taxonomy["categories"]),
            "project_id": taxonomy["project_id"]
        })
        
        if "unified_taxonomy" in st.session_state:
            del st.session_state.unified_taxonomy
        wizard_state.needs_refresh = True
        st.success("✅ Taxonomy changes applied and saved to project.")
        st.rerun()

