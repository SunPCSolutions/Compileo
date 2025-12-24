"""
Document Processing view for Compileo GUI.
Handles document upload, ingestion, and chunking.
"""

import streamlit as st
from typing import List, Dict, Any
from datetime import datetime
import os
import json
import re
import time
import pandas as pd
from ....core.logging import get_logger

logger = get_logger(__name__)

from ..services.document_api_service import get_projects, get_project_documents, delete_document
from ..services.document_processing_service import (
    parse_selected_documents,
    chunk_parsed_documents,
    get_document_parsed_files,
    get_parsed_file_content
)
from ..services.api_client import api_client
from ..components.ai_chunking_config import render_ai_assisted_configuration
from ..components.pagination_settings import render_pagination_settings
from ..state.session_state import session_state

def parse_chunk_ids(input_str: str, chunks_data: list = None) -> list[str]:
    """
    Parse chunk ID input supporting formats:
    - Single: "221"
    - Multiple: "221, 222, 224"
    - Ranges: "221-230"
    - Combined: "221-230, 235, 240-245"
    
    If chunks_data is provided, maps indices to UUIDs.
    Otherwise returns integer indices (legacy behavior, but not useful for API).
    """
    import re
    target_indices = set()  # Use set to avoid duplicates

    # Split by comma and process each part
    parts = [part.strip() for part in input_str.split(',') if part.strip()]

    for part in parts:
        if '-' in part:
            # Handle range
            try:
                start, end = [int(x.strip()) for x in part.split('-')]
                if start <= end:
                    target_indices.update(range(start, end + 1))
                else:
                    # Invalid range, skip
                    continue
            except ValueError:
                # Invalid range format, skip
                continue
        else:
            # Handle single number
            try:
                target_indices.add(int(part.strip()))
            except ValueError:
                # Invalid number, skip
                continue

    # Map indices to UUIDs if chunks_data is provided
    if chunks_data:
        chunk_uuids = []
        index_map = {}
        for chunk in chunks_data:
            if isinstance(chunk, dict):
                idx = chunk.get('chunk_index')
                uuid = chunk.get('id')
                if idx is not None and uuid:
                    index_map[int(idx)] = uuid
        
        for idx in target_indices:
            if idx in index_map:
                chunk_uuids.append(index_map[idx])
        return chunk_uuids
    
    return sorted(list(target_indices))


def validate_and_fix_json_schema(schema_json: str) -> tuple[str, bool]:
    """
    Validate JSON schema and attempt to fix common corruption issues.

    Uses codecs.decode with unicode_escape to handle over-escaped strings
    that may result from copying/pasting JSON with backslashes.

    Returns:
        tuple: (fixed_schema, was_fixed) - the schema string and whether it was modified
    """
    import codecs

    if not schema_json or not schema_json.strip():
        return schema_json, False

    # First, try parsing as-is
    try:
        json.loads(schema_json)
        return schema_json, False  # Already valid
    except json.JSONDecodeError as e:
        original_error = str(e)

        # Check if it's a backslash escaping issue
        if "Invalid \\escape" in original_error or "\\" in schema_json:
            try:
                # Use codecs.decode to handle Python-style escape sequences
                # This properly handles strings that have been "over-escaped"
                # e.g., {"key": "value\\with\\backslashes"} → {"key": "value\with\backslashes"}
                fixed_text = codecs.decode(schema_json, 'unicode_escape')

                # Test if the fix works
                json.loads(fixed_text)
                return fixed_text, True  # Successfully fixed

            except (UnicodeDecodeError, json.JSONDecodeError):
                # Fix attempt failed, try alternative approach
                try:
                    # Fallback: double the backslashes in value fields
                    def fix_backslashes(match):
                        content = match.group(1)
                        fixed_content = content.replace('\\', '\\\\')
                        return f'"value": "{fixed_content}"'

                    fixed_json = re.sub(
                        r'"value"\s*:\s*"([^"]*)"',  # Match "value": "content"
                        fix_backslashes,
                        schema_json
                    )

                    json.loads(fixed_json)
                    return fixed_json, True

                except json.JSONDecodeError:
                    pass

        # Return original if we can't fix it
        return schema_json, False


def render_document_processing():
    """Render the Document Processing page with compact tabbed layout."""
    st.markdown("## 📄 Document Processing")

    # Project Selection (compact)
    projects = get_projects()
    if projects:
        project_options = {f"{p['name']} (ID: {p['id']})": p['id'] for p in projects}
        selected_project_display = st.selectbox(
            "Project",
            options=list(project_options.keys()),
            help="Choose the project to process documents for",
            key="project_select"
        )
        selected_project_id = project_options[selected_project_display]
        session_state.selected_project_id = selected_project_id
    else:
        st.error("⚠️ No projects available. Please create a project first.")
        return

    # Create compact tabbed interface
    tab_parse, tab_chunk, tab_manage = st.tabs(["📄 Parse Documents", "✂️ Configure & Chunk Documents", "📑 View & Manage Chunks"])

    # ============================================================================
    # TAB 1: PARSING
    # ============================================================================
    with tab_parse:
        # Get existing documents for the project
        existing_documents = get_project_documents(selected_project_id)

        # Initialize pagination settings
        pages_per_split = 5  # Default value

        # Check for scraper plugin
        try:
            plugins = api_client.get("/api/v1/plugins/")
            has_scraper_plugin = any(p['id'] == 'compileo-scrapy-playwright-scraper' for p in plugins) if plugins else False
        except:
            has_scraper_plugin = False

        col1, col2 = st.columns(2)

        with col1:
            input_mode = "Upload"
            if has_scraper_plugin:
                input_mode = st.radio("Input Source", ["Upload", "Website"], horizontal=True)

            uploaded_files = None
            website_url = None
            website_depth = 1

            if input_mode == "Upload":
                # Initialize upload key if not present
                if 'processing_upload_key' not in st.session_state:
                    st.session_state.processing_upload_key = 0

                # Document Upload
                uploaded_files = st.file_uploader(
                    "Upload documents",
                    accept_multiple_files=True,
                    type=['pdf', 'docx', 'doc', 'pptx', 'ppt', 'xlsx', 'xls', 'txt', 'md', 'csv', 'json', 'xml'],
                    help="Drag and drop files here • Limit 200MB per file • PDF, Office (DOCX, PPTX, XLSX), TXT, MD, CSV, JSON, XML",
                    key=f"processing_upload_{st.session_state.processing_upload_key}"
                )

                # Pagination Settings (shown after file upload)
                if uploaded_files:
                    pages_per_split = render_pagination_settings()
                    session_state.overlap_pages = 0 # Always 0 with dynamic overlap generation
            else:
                website_url = st.text_input("Website URL", placeholder="https://example.com")
                website_depth = st.number_input("Depth", min_value=1, max_value=5, value=1, help="Crawling depth")

        with col2:
            # Parser Selection
            if input_mode == "Upload":
                # Determine supported parsers based on uploaded files (reactive filtering)
                all_parsers = ["grok", "gemini", "ollama", "openai", "pypdf", "unstructured", "huggingface", "novlm"]
                supported_parsers = all_parsers.copy()
                
                # Check uploaded files in session state
                check_files = st.session_state.get(f"processing_upload_{st.session_state.get('processing_upload_key', 0)}")
                if check_files:
                    if not isinstance(check_files, list):
                        check_files = [check_files]
                        
                    has_non_pdf = any(not f.name.lower().endswith('.pdf') for f in check_files)
                    if has_non_pdf:
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

                selected_parser = st.selectbox(
                    "Parser",
                    options=supported_parsers,
                    index=default_index,
                    help="AI model for parsing. Available models filtered based on uploaded file types.",
                    key="parser_select"
                )
            else:
                st.info("Using Scrapy-Playwright Plugin")
                selected_parser = "scrapy_playwright"

        # Document Selection
        selected_document_ids: List[int] = []

        if existing_documents:
            st.markdown("**Select documents to parse:**")

            # Compact document list with delete options
            cols = st.columns(3)
            for i, doc in enumerate(existing_documents):
                col_idx = i % 3
                with cols[col_idx]:
                    status_emoji = "✅" if doc.get('status') == 'parsed' else "📄"
                    can_select = doc.get('status') != 'parsed'

                    # Document checkbox and delete button in columns
                    doc_col1, doc_col2 = st.columns(2)
                    with doc_col1:
                        if st.checkbox(
                            f"{status_emoji} {doc['file_name']}",
                            value=False,
                            disabled=not can_select,
                            key=f"parse_doc_{doc['id']}",
                            help="Already parsed" if not can_select else "Click to parse"
                        ):
                            selected_document_ids.append(doc['id'])
                    with doc_col2:
                        if st.button("🗑️", key=f"delete_doc_{doc['id']}", help=f"Delete {doc['file_name']}"):
                            delete_document(doc['id'])

        # Parse Action
        has_input = len(selected_document_ids) > 0 or (uploaded_files is not None and len(uploaded_files) > 0) or (website_url is not None and len(website_url) > 0)
        
        if has_input:
            label = "Parse Website" if website_url else f"📄 Parse {len(selected_document_ids) + (len(uploaded_files) if uploaded_files else 0)} Documents"
            
            if st.button(label, type="primary"):
                st.info("GUI: Submitting parsing job...")
                
                if website_url:
                    # Handle website scraping directly via API since parse_selected_documents handles files/ids
                    try:
                        # We need to trigger the ingestion API with the URL
                        # parse_selected_documents logic is in services/document_processing_service.py
                        # It uploads files then calls process API.
                        # For website, we can call the process API directly with the URL as a "document" ?
                        # Or use the new ingestion endpoint logic?
                        
                        # Since we modified the ingestion module to handle URLs, we can potentially pass it.
                        # However, parse_selected_documents expects files.
                        
                        # Let's call the plugin API directly for now, OR adapt parse_selected_documents.
                        # Using plugin API directly is cleaner for the GUI flow here.
                        
                        # Note: GUI APIClient.post expects 'data' arg, which it sends as json internally
                        response = api_client.post("/api/v1/plugins/scrapy-playwright/scrape", data={
                            "url": website_url,
                            "depth": website_depth,
                            "project_id": str(selected_project_id)
                        })
                        if response and response.get("status") == "success":
                             st.success(f"Website scraping completed for {website_url}")
                             # Refresh document list after successful parsing
                             st.rerun()
                        else:
                             st.error(f"Website scraping failed: {response}")

                    except Exception as e:
                        st.error(f"Error submitting scraping job: {e}")
                else:
                    parse_selected_documents(
                        selected_project_id,
                        uploaded_files,
                        selected_parser,
                        selected_document_ids,
                        pages_per_split
                    )
                st.info("GUI: Job submitted.")
        else:
            st.info("Select documents to parse, upload new files, or enter a URL.")

    # ============================================================================
    # TAB 2: CONFIGURE & CHUNK DOCUMENTS
    # ============================================================================
    with tab_chunk:
        # Get fresh document list for this tab
        existing_documents = get_project_documents(selected_project_id)

        # In-view job monitoring is now handled within the button handlers
        # in document_processing_service.py. This redundant section is removed to avoid
        # disruptive whole-page polling reruns.

        # Get parsed documents for configuration - refresh automatically if job just completed
        docs_placeholder = st.empty()

        # Check if we just completed a job (session state flag)
        job_just_completed = hasattr(session_state, 'job_just_completed') and session_state.job_just_completed
        if job_just_completed:
            session_state.job_just_completed = False  # Clear the flag

        # Always fetch fresh documents for this tab
        existing_documents = get_project_documents(selected_project_id)
        available_parsed_docs = [doc for doc in existing_documents if doc.get('status') in ['parsed', 'chunked']]

        # Document selection for chunking (moved to top)
        st.markdown("### 📄 Select Documents to Chunk")

        # Show document count and auto-refresh status
        if available_parsed_docs:
            docs_placeholder.success(f"✅ {len(available_parsed_docs)} parsed documents available")
        else:
            docs_placeholder.warning("No parsed documents available. Parse documents first in the '📄 Parse Documents' tab.")

        selected_chunk_docs = []

        if available_parsed_docs:
            # Compact grid layout for document selection
            cols = st.columns(3)
            for i, doc in enumerate(available_parsed_docs):
                col_idx = i % 3
                with cols[col_idx]:
                    if st.checkbox(f"📄 {doc['file_name']}", key=f"chunk_doc_{doc['id']}"):
                        selected_chunk_docs.append(doc)

            # Show selection status
            if selected_chunk_docs:
                st.success(f"Selected {len(selected_chunk_docs)} documents for chunking")
            else:
                st.info("Select documents to chunk from the list above")

        # Always show chunking configuration
        st.markdown("### ⚙️ Chunking Configuration")

        # Configuration mode first
        config_mode = st.radio(
            "Configuration",
            ["Manual", "AI-Assisted"],
            help="Manual setup or AI recommendations",
            key="config_mode",
            horizontal=True
        )

        # Chunking strategy and model selection
        col1, col2 = st.columns(2)

        with col1:
            chunk_strategy = st.selectbox(
                "Chunking Strategy",
                options=["character", "token", "semantic", "delimiter", "schema"],
                help="Method to split documents into chunks",
                key="chunk_strategy"
            )

        with col2:
            # AI Model selection - for strategies that require AI OR when AI-assisted is selected
            if chunk_strategy in ["semantic"] or config_mode == "AI-Assisted":
                selected_chunker = st.selectbox(
                    "AI Model",
                    options=["grok", "gemini", "ollama", "openai"],
                    help="AI model for intelligent chunking",
                    key="chunker_model"
                )
            else:
                # For deterministic strategies without AI assistance, use a default model (won't be used)
                selected_chunker = "grok"
                st.write("")  # Empty space to maintain layout

        # Cross-file chunking is automatically enabled for all parsed documents
        use_sliding_window = True

        # Parameters based on strategy
        chunk_size = 512
        overlap = 50
        similarity_threshold = 0.7
        min_chunk_size = 100
        delimiters = ["\n\n", "\n"]
        semantic_prompt = ""
        schema_json = ""

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

        if config_mode == "AI-Assisted":
            user_instructions, user_examples = render_ai_assisted_configuration(
                available_parsed_docs, selected_project_id, selected_chunker
            )

        # ============================================================================
        # PARSED FILE SELECTION AND PREVIEW
        # ============================================================================
        selected_parsed_files = []
        if selected_chunk_docs:
            st.markdown("### 📄 Parsed File Selection (Optional)")

            # Check for documents with multiple parsed files
            docs_with_parsed_files = []
            for doc in selected_chunk_docs:
                parsed_files = get_document_parsed_files(doc['id'])
                if parsed_files and len(parsed_files) > 1:
                    docs_with_parsed_files.append((doc, parsed_files))

            if docs_with_parsed_files:
                st.info("📋 Some documents have multiple parsed files. You can select specific files to use for AI-assisted configuration, or leave unselected to process all files automatically.")

                for doc, parsed_files in docs_with_parsed_files:
                    st.markdown(f"**{doc['file_name']}**")

                    # Create options for the dropdown
                    file_options = {}
                    for i, parsed_file in enumerate(parsed_files, 1):
                        file_path = parsed_file['file_path']
                        filename = os.path.basename(file_path)
                        page_number = parsed_file.get('page_number')

                        display_name = filename
                        if page_number:
                            display_name = f"{filename} (Page {page_number})"

                        file_options[display_name] = {
                            'path': file_path,
                            'index': i-1,
                            'page_number': page_number
                        }

                    # File selector dropdown
                    selected_file_display = st.selectbox(
                        f"Select parsed file for {doc['file_name']} (optional):",
                        options=["All files (automatic)"] + list(file_options.keys()),
                        key=f"parsed_file_select_{doc['id']}"
                    )

                    if selected_file_display != "All files (automatic)":
                        selected_file_info = file_options[selected_file_display]
                        selected_file_info['document_id'] = doc['id']
                        selected_parsed_files.append(selected_file_info)

                        # Show metadata if available
                        page_number = selected_file_info.get('page_number')
                        if page_number:
                            st.metric("Page Number", page_number)

                        # Preview content
                        content = get_parsed_file_content(selected_file_info['path'])
                        if content:
                            st.markdown("**Preview:**")
                            # Show first 1000 characters as preview
                            preview_text = content[:1000] + ("..." if len(content) > 1000 else "")
                            st.text_area(
                                f"Content preview for {os.path.basename(selected_file_info['path'])}:",
                                value=preview_text,
                                height=200,
                                disabled=True,
                                key=f"preview_{doc['id']}_{selected_file_info['index']}"
                            )

                            # Copy button for individual file
                            if st.button(f"📋 Copy {os.path.basename(selected_file_info['path'])} Content",
                                        key=f"copy_{doc['id']}_{selected_file_info['index']}"):
                                # Copy to clipboard (Streamlit doesn't have direct clipboard, but we can show it)
                                st.code(content, language="markdown")
                                st.success("Content copied to display area above!")
                        else:
                            st.warning(f"Could not load content for {os.path.basename(selected_file_info['path'])}")
            else:
                st.info("📄 Selected documents will be chunked as complete files automatically.")

        # Chunk button (common for both modes)
        if selected_chunk_docs and st.button("✂️ Chunk Documents", type="primary"):
                    # Debug: Log button click
                    from datetime import datetime
                    import json
                    debug_context = {
                        "timestamp": datetime.now().isoformat(),
                        "level": "INFO",
                        "component": "document_processing_view",
                        "message": "Chunk Documents button clicked",
                        "context": {
                            "config_mode": config_mode,
                            "selected_chunk_docs_count": len(selected_chunk_docs),
                            "chunk_strategy": chunk_strategy,
                            "chunker": selected_chunker,
                            "ai_recommendations_available": "ai_recommended_strategy" in st.session_state
                        }
                    }
                    logger.debug(json.dumps(debug_context))

                    # Check for AI recommendations and use them if available
                    if "ai_recommended_strategy" in st.session_state and "ai_recommended_params" in st.session_state:
                        st.info("🤖 Using AI-recommended chunking parameters...")
                        ai_strategy = st.session_state.ai_recommended_strategy
                        ai_params = st.session_state.ai_recommended_params

                        # Override form values with AI recommendations
                        # Ensure strategy is lowercase to match backend expectations
                        chunk_strategy = ai_strategy.lower()
                        if "chunk_size" in ai_params:
                            chunk_size = ai_params["chunk_size"]
                        if "overlap" in ai_params:
                            overlap = ai_params["overlap"]
                        if "delimiters" in ai_params:
                            delimiters = ai_params["delimiters"]
                        if "semantic_prompt" in ai_params:
                            semantic_prompt = ai_params["semantic_prompt"]
                        if "json_schema" in ai_params:
                            schema_json = ai_params["json_schema"]

                        debug_context["ai_strategy_used"] = ai_strategy
                        debug_context["ai_params_used"] = ai_params
                        logger.debug(json.dumps({
                            "timestamp": datetime.now().isoformat(),
                            "level": "INFO",
                            "component": "document_processing_view",
                            "message": "AI recommendations applied",
                            "context": debug_context
                        }))

                    # Validate and fix schema JSON if using schema strategy
                    original_schema = schema_json
                    if chunk_strategy == "schema" and schema_json.strip():
                        fixed_schema, was_fixed = validate_and_fix_json_schema(schema_json)
                        if was_fixed:
                            st.info("🔧 Auto-fixed JSON schema backslash escaping issues")
                            schema_json = fixed_schema
                            st.code(f"Original: {original_schema[:100]}...")
                            st.code(f"Fixed: {schema_json[:100]}...")
                        elif original_schema.strip():
                            # Try to validate the schema to give immediate feedback
                            try:
                                json.loads(original_schema)
                                st.success("✅ JSON schema is valid")
                            except json.JSONDecodeError as e:
                                st.error(f"❌ JSON schema error: {e}")
                                st.warning("💡 Try re-pasting from AI recommendations or use double backslashes (\\\\) for regex patterns")
                                return

                    # Use selected parsed files if any were chosen, otherwise process all files automatically
                    selected_files_for_chunking = selected_parsed_files if selected_parsed_files else None

                    chunk_parsed_documents(
                        selected_project_id,
                        selected_chunk_docs,
                        selected_chunker,
                        chunk_strategy,
                        chunk_size,
                        overlap,
                        delimiters,
                        semantic_prompt,
                        schema_json,
                        selected_files_for_chunking,
                        use_sliding_window
                    )

    # ============================================================================
    # TAB 3: VIEW & MANAGE CHUNKS
    # ============================================================================
    with tab_manage:
        # 1. Select Project
        with st.spinner("Loading projects..."):
            try:
                projects_resp = api_client.get("/api/v1/projects")
                projects = projects_resp.get("projects", []) if projects_resp else []
            except Exception as e:
                st.error(f"Failed to load projects: {e}")
                return

        # Get fresh document list for this tab
        existing_documents = get_project_documents(selected_project_id)

        if not projects:
            st.info("No projects found. Please create a project first.")
            if st.button("Go to Projects"):
                session_state.current_page = "projects"
                st.rerun()
            return

        # Use session state to persist selection if possible, or default to first
        project_options = {p['name']: p['id'] for p in projects}

        # If we have a current project in session state, try to use it
        index = 0
        if session_state.current_project:
            current_proj_id = session_state.current_project.get('id')
            names = [name for name, pid in project_options.items() if pid == current_proj_id]
            if names:
                try:
                    index = list(project_options.keys()).index(names[0])
                except ValueError:
                    pass

        selected_project_name = st.selectbox("Select Project", options=list(project_options.keys()), index=index, key="tab_chunk_project_select")
        project_id = project_options[selected_project_name]

        # Update session state if changed
        if not session_state.current_project or session_state.current_project.get('id') != project_id:
            # Find the project dict
            proj = next((p for p in projects if p['id'] == project_id), None)
            if proj:
                session_state.current_project = proj

        # 2. Select Document
        with st.spinner("Loading documents..."):
            try:
                docs_resp = api_client.get("/api/v1/documents", params={"project_id": project_id})
                documents = docs_resp.get("documents", []) if docs_resp else []
            except Exception as e:
                st.error(f"Failed to load documents: {e}")
                return

        if not documents:
            st.info("No documents found in this project.")
            return

        doc_options = {f"{d['file_name']} (ID: {d['id']})": d['id'] for d in documents}
        selected_doc_label = st.selectbox("Select Document", options=list(doc_options.keys()), key="tab_chunk_doc_select")
        document_id = doc_options[selected_doc_label]

        # 3. List Chunks
        st.divider()
        st.subheader(f"Chunks for Document {document_id}")

        # Refresh button
        if st.button("🔄 Refresh Chunks", key="refresh_chunks"):
            st.rerun()

        with st.spinner("Loading chunks..."):
            try:
                response = api_client.get(f"/api/v1/chunks/document/{document_id}")
                chunks = response.get("chunks", []) if isinstance(response, dict) else response
            except Exception as e:
                st.error(f"Failed to load chunks: {e}")
                return

        if not chunks:
            st.info("No chunks found for this document.")
            st.markdown("Go to **Configure & Chunk Documents** tab to chunk this document.")
        else:
            st.success(f"Found {len(chunks)} chunks.")

            # Delete Specific Chunks
            col_title, col_example = st.columns([1, 2])
            with col_title:
                st.markdown('<p style="color: #ff6b35; font-weight: bold; margin: 0;">Delete Specific Chunks</p>', unsafe_allow_html=True)
            with col_example:
                st.markdown('<p style="color: #666; margin: 0; padding-top: 2px;">For example, single chunk: <code>4</code> ; chunk range: <code>10-22</code> ; or a combination: <code>4, 6-10, 10-22</code></p>', unsafe_allow_html=True)

            chunk_ids_input = st.text_input("Chunk IDs to delete", placeholder="e.g., 4, 10-22, 25", key="tab_chunk_ids_delete", label_visibility="collapsed")

            # Both delete buttons side by side
            delete_col1, delete_col2 = st.columns(2)
            with delete_col1:
                if st.button("🗑️ Delete Selected Chunks", type="primary", width='stretch', key="tab_delete_selected_chunks"):
                    if not chunk_ids_input.strip():
                        st.error("Please enter chunk IDs to delete.")
                        return

                    try:
                        # Parse the input to extract chunk IDs (UUIDs)
                        chunk_ids = parse_chunk_ids(chunk_ids_input, chunks)
                        if not chunk_ids:
                            st.error("No valid chunk indices found matching the current document's chunks.")
                            return

                        # Delete chunks in batch
                        response = api_client.delete("/api/v1/chunks/batch", data={"chunk_ids": chunk_ids})
                        if response and response.get("deleted_count", 0) > 0:
                            st.success(f"Successfully deleted {response['deleted_count']} chunks.")
                        else:
                            st.warning("No chunks were deleted. Please check the chunk indices.")

                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete chunks: {e}")

            with delete_col2:
                if st.button("🗑️ Delete All Chunks", type="primary", width='stretch', key="tab_delete_all_chunks"):
                    # Use a confirmation dialog pattern if Streamlit supported modal easily,
                    # but here we'll just use session state for confirmation or just do it if "Are you sure" is implied by the button label being red/primary?
                    # Actually standard pattern is click -> separate confirm button appears.
                    st.session_state.tab_show_delete_confirm = True

            if st.session_state.get("tab_show_delete_confirm"):
                st.warning("Are you sure you want to delete ALL chunks for this document? This cannot be undone.")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ Yes, Delete All", key="tab_confirm_delete_all"):
                        try:
                            api_client.delete(f"/api/v1/chunks/document/{document_id}")
                            st.success("All chunks deleted successfully.")
                            st.session_state.tab_show_delete_confirm = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete chunks: {e}")
                with c2:
                    if st.button("❌ Cancel", key="tab_cancel_delete_all"):
                        st.session_state.tab_show_delete_confirm = False
                        st.rerun()

            # Display chunks as expandable sections for better content viewing
            st.write(f"Displaying {len(chunks)} chunks:")
            
            # Sort chunks by index for better readability
            try:
                chunks.sort(key=lambda x: int(x.get("chunk_index")) if isinstance(x, dict) and x.get("chunk_index") is not None else 0)
            except:
                pass

            for item in chunks:
                content = ""
                full_content = ""
                if isinstance(item, dict):
                    # API returns 'text' field for content, not 'content' or 'content_preview'
                    full_content = item.get("text", "")
                    if not full_content:
                        full_content = "(No content available)"
                    content = full_content
                else:
                    content = str(item)

                # Get index and token count for display
                chunk_index = item.get('chunk_index') if isinstance(item, dict) else 'N/A'
                # API doesn't return token_count, so estimate from content length
                token_count = 'N/A'
                if content and content != "(No content available)":
                    token_count = f"~{len(content) // 4}"

                uuid_str = item.get('id') if isinstance(item, dict) else item

                with st.expander(f"Chunk {chunk_index} (Tokens: {token_count})"):
                    st.caption(f"UUID: {uuid_str}")
                    st.text_area(
                        "Content",
                        value=content,
                        height=200,
                        disabled=True,
                        key=f"content_{uuid_str}"
                    )

    # Job status is now displayed in the chunking tab
