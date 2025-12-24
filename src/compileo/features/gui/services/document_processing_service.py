"""
Document processing service module for Compileo GUI.
Handles document parsing and chunking workflows.
"""

from typing import List, Dict, Any, Optional
import streamlit as st
import os
import json
from datetime import datetime

from ..services.api_client import api_client
from ..services.async_job_submission_service import async_job_service
from ..services.job_monitoring_service import wait_for_upload_completion, wait_for_upload_completion_with_paths, monitor_job_synchronously
from ..state.session_state import session_state
from ...jobhandle.models import JobType, JobPriority
from ....core.logging import get_logger

logger = get_logger(__name__)


def parse_selected_documents(project_id: int, uploaded_files: List, parser: str, selected_document_ids: Optional[List[int]] = None, pages_per_split: int = 5):
    """Parse selected documents to markdown without chunking using async job submission."""
    from datetime import datetime
    try:
        # DEBUG: [DEBUG_20251003_PDF_PARSE] - Starting document parsing process
        debug_context = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "component": "document_processing_service",
            "message": "Starting document parsing process",
            "context": {
                "project_id": project_id,
                "uploaded_files_count": len(uploaded_files) if uploaded_files else 0,
                "selected_docs_count": len(selected_document_ids) if selected_document_ids else 0,
                "parser": parser
            }
        }
        logger.debug(json.dumps(debug_context))

        # Prepare document IDs for parsing
        document_ids_to_parse = []

        # Handle uploaded files
        if uploaded_files:
            # DEBUG: [DEBUG_20251003_PDF_PARSE] - Uploading files
            debug_context = {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "component": "document_processing_service",
                "message": "Uploading files",
                "context": {
                    "project_id": project_id,
                    "files_count": len(uploaded_files)
                }
            }
            logger.debug(json.dumps(debug_context))
            with st.spinner("📤 Uploading files..."):
                st.info(f"📤 Uploading {len(uploaded_files)} files first...")
                # First upload the files
                upload_response = api_client.upload_documents(project_id, uploaded_files)
                if upload_response and "uploaded_files" in upload_response:
                    document_ids_to_parse.extend([doc["id"] for doc in upload_response["uploaded_files"]])
                    st.success(f"✅ Uploaded {len(upload_response['uploaded_files'])} files")
                    # DEBUG: [DEBUG_20251003_PDF_PARSE] - Files uploaded successfully
                    debug_context = {
                        "timestamp": datetime.now().isoformat(),
                        "level": "INFO",
                        "component": "document_processing_service",
                        "message": "Files uploaded successfully",
                        "context": {
                            "project_id": project_id,
                            "uploaded_count": len(upload_response["uploaded_files"]),
                            "document_ids": document_ids_to_parse
                        }
                    }
                    logger.debug(json.dumps(debug_context))
                else:
                    # DEBUG: [DEBUG_20251003_PDF_PARSE] - File upload failed
                    logger.error(f"File upload failed for project_id: {project_id}")
                    st.error("❌ Failed to upload files")
                    return

        # Handle existing documents (only if no uploaded files)
        elif selected_document_ids:
            document_ids_to_parse.extend(selected_document_ids)
            # DEBUG: [DEBUG_20251003_PDF_PARSE] - Added existing documents
            debug_context = {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "component": "document_processing_service",
                "message": "Added existing documents to parse list",
                "context": {
                    "project_id": project_id,
                    "existing_docs": selected_document_ids
                }
            }
            logger.debug(json.dumps(debug_context))

        if not document_ids_to_parse:
            # DEBUG: [DEBUG_20251003_PDF_PARSE] - No documents to parse
            logger.warning(f"No documents to parse for project_id: {project_id}")
            st.error("No documents to parse")
            return

        # DEBUG: [DEBUG_20251003_PDF_PARSE] - Preparing job parameters
        debug_context = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "component": "document_processing_service",
            "message": "Preparing job parameters",
            "context": {
                "project_id": project_id,
                "total_docs_to_parse": len(document_ids_to_parse),
                "document_ids": document_ids_to_parse
            }
        }
        logger.debug(json.dumps(debug_context))

        # Submit async parsing job
        st.info(f"🚀 Submitting parsing job for {len(document_ids_to_parse)} documents with {parser}...")

        # Use provided pagination settings
        overlap_pages = 0  # Always 0 with dynamic overlap generation

        job_parameters = {
            "operation": "parse_documents",
            "project_id": project_id,
            "document_ids": document_ids_to_parse,
            "parser": parser,
            "pages_per_split": pages_per_split,
            "overlap_pages": overlap_pages
        }

        # DEBUG: [DEBUG_20251003_PDF_PARSE] - Submitting job to async service
        debug_context = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "component": "document_processing_service",
            "message": "Submitting job to async service",
            "context": {
                "project_id": project_id,
                "job_parameters_keys": list(job_parameters.keys())
            }
        }
        logger.debug(json.dumps(debug_context))

        job_id = async_job_service.submit_job(
            job_type=JobType.DOCUMENT_PROCESSING,
            parameters=job_parameters,
            priority=JobPriority.NORMAL
        )

        # Store job ID in session state for monitoring
        session_state.processing_job_id = job_id

        # DEBUG: [DEBUG_20251003_PDF_PARSE] - Job submitted successfully
        debug_context = {
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "component": "document_processing_service",
            "message": "Job submitted successfully",
            "context": {
                "project_id": project_id,
                "job_id": job_id,
                "document_count": len(document_ids_to_parse)
            }
        }
        logger.debug(json.dumps(debug_context))

        st.success(f"✅ Parsing job submitted successfully! Job ID: {job_id}")
        
        # Monitor job synchronously in-view (Wizard-style)
        success = monitor_job_synchronously(job_id, success_text="Document parsing completed!")
        
        if success:
            # Set flag for auto-refresh
            session_state.job_just_completed = True
            
            # Reset file uploader component by incrementing key
            if 'processing_upload_key' in st.session_state:
                st.session_state.processing_upload_key += 1
                
            st.rerun()  # Refresh to show updated document status

    except Exception as e:
        # DEBUG: [DEBUG_20251003_PDF_PARSE] - Exception during parsing
        logger.error(f"Exception during parsing job submission for project_id {project_id}: {e}", exc_info=True)
        st.error(f"Error during document parsing job submission: {e}")


def chunk_parsed_documents(project_id: int, parsed_docs: List[dict], chunker: str, chunk_strategy: str,
                            chunk_size: int, overlap: int, delimiters: List[str], semantic_prompt: str, schema_json: str,
                            selected_parsed_files: Optional[List[Dict[str, Any]]] = None,
                            sliding_window: bool = False):
    """Chunk already parsed documents using current settings with async job submission."""
    try:
        files_by_document = None
        document_ids = []

        if selected_parsed_files:
            st.info(f"🚀 Submitting chunking job for {len(selected_parsed_files)} selected parsed files...")

            files_by_document = {}
            for file_info in selected_parsed_files:
                doc_id = file_info.get('document_id')
                if doc_id is None:
                    filename = os.path.basename(file_info['path'])
                    try:
                        # Attempt to extract doc id from filename if available
                        parts = filename.split('_')
                        if parts and parts[0].isdigit():
                            doc_id = int(parts[0])
                    except Exception:
                        pass
                
                if doc_id is None:
                    st.warning(f"Could not identify document ID for file: {file_info['path']}")
                    continue

                doc_id_str = str(doc_id)
                if doc_id_str not in files_by_document:
                    files_by_document[doc_id_str] = []
                    st.info(f"📋 Grouping selected files for document {doc_id}")

                files_by_document[doc_id_str].append(file_info['path'])
            
            # Extract document IDs from keys
            document_ids = [int(did) for did in files_by_document.keys()]
            
        else:
            document_ids = [doc["id"] for doc in parsed_docs]
            st.info(f"🚀 Submitting chunking job for {len(document_ids)} documents...")

        job_parameters = {
            "operation": "chunk_documents",
            "project_id": project_id,
            "document_ids": document_ids,
            "selected_files": files_by_document,  # Will be None if all files selected
            "chunker": chunker,
            "chunk_strategy": chunk_strategy,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "delimiters": delimiters,
            "semantic_prompt": semantic_prompt,
            "schema_definition": schema_json,
            "character_chunk_size": chunk_size,
            "character_overlap": overlap,
            "skip_parsing": True,
            "sliding_window": sliding_window
        }

        job_id = async_job_service.submit_job(
            job_type=JobType.DOCUMENT_PROCESSING,
            parameters=job_parameters,
            priority=JobPriority.NORMAL
        )

        st.success(f"✅ Chunking job submitted successfully! Job ID: {job_id}")
        
        # Monitor job synchronously in-view (Wizard-style)
        success = monitor_job_synchronously(job_id, success_text="Document chunking completed!")
        
        if success:
            # Set flag for auto-refresh
            session_state.job_just_completed = True
            st.rerun()  # Refresh to show updated document status

    except Exception as e:
        st.error(f"Error during document chunking job submission: {e}")


# REMOVED: Legacy process_documents function
# All GUI document processing now goes through RQ worker via:
# - parse_selected_documents() for parsing
# - chunk_parsed_documents() for chunking
# No alternative routes or backwards compatibility maintained.




def get_project_documents(project_id: int) -> List[Dict[str, Any]]:
    """Get list of documents for a specific project from API."""
    try:
        response = api_client.get(f"/api/v1/documents?project_id={project_id}")
        if response and "documents" in response:
            return response["documents"]
        return []
    except Exception as e:
        st.error(f"Failed to load project documents: {e}")
        return []


def get_document_parsed_files(document_id: int) -> List[Dict[str, Any]]:
    """
    Get the list of parsed files for a document from the API response.

    Args:
        document_id: The document ID

    Returns:
        List of parsed file dictionaries (e.g., [{'file_path': '...', 'page_number': 1}, ...])
    """
    try:
        # Get document info which now includes parsed_files list
        response = api_client.get(f"/api/v1/documents/{document_id}")
        if response and "parsed_files" in response:
            return response["parsed_files"]
        return []

    except Exception as e:
        st.error(f"Failed to get parsed files for document {document_id}: {e}")
        return []


def get_parsed_file_content(file_path: str) -> Optional[str]:
    """
    Get the content of a parsed file.
    For JSON files with structured content, extracts main_content.
    For plain text/markdown files, returns the full content.

    Args:
        file_path: Path to the parsed file (JSON or text)

    Returns:
        File content as string or None if not found
    """
    try:
        if not os.path.exists(file_path):
            return None

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()

        # Check if this is a JSON file with structured content
        try:
            json_data = json.loads(raw_content)
            if isinstance(json_data, dict) and 'main_content' in json_data:
                # Extract main_content from JSON structure
                return json_data['main_content']
        except json.JSONDecodeError:
            # Not JSON, treat as plain text
            pass

        # Return raw content for non-JSON files or JSON without main_content
        return raw_content

    except Exception as e:
        st.error(f"Failed to read parsed file {file_path}: {e}")
        return None