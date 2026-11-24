"""
Document processing service module for Compileo GUI.
Handles document parsing and chunking workflows.
"""

import logging
from typing import List, Dict, Any, Optional
import streamlit as st
import os
import json
from datetime import datetime

from ..services.api_client import api_client
from ..services.unified_job_service import unified_job_service, JobResult
from ..state.session_state import session_state
from ...jobhandle.models import JobType, JobPriority


logger = logging.getLogger(__name__)

def _get_user_id() -> Optional[str]:
    """Get the current user ID from session state."""
    try:
        return session_state.user_id
    except:
        return None


def parse_selected_documents(
    project_id: int,
    uploaded_files: List,
    parser: str,
    selected_document_ids: Optional[List[int]] = None,
    pages_per_split: int = 5,
    page_selection: Optional[str] = None
):
    """Parse selected documents to markdown without chunking using async job submission.

    Each document is submitted as a separate job for parallel processing.
    """
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
                    logger.error(f'{{"timestamp": "{datetime.now().isoformat()}", "level": "ERROR", "component": "document_processing_service", "message": "File upload failed", "context": {{"project_id": {project_id}}}}}')
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
            logger.warning(f'{{"timestamp": "{datetime.now().isoformat()}", "level": "WARNING", "component": "document_processing_service", "message": "No documents to parse", "context": {{"project_id": {project_id}}}}}')
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

        # Submit each document as a separate job for parallel processing
        st.info(f"🚀 Submitting {len(document_ids_to_parse)} parsing jobs with {parser}...")

        user_id = _get_user_id()
        if not user_id:
            st.error("❌ You must be logged in to parse documents.")
            return

        # Use provided pagination settings
        overlap_pages = 0  # Always 0 with dynamic overlap generation

        # Submit individual jobs - one per document
        job_ids = []
        for doc_id in document_ids_to_parse:
            job_parameters = {
                "operation": "parse_documents",
                "project_id": project_id,
                "document_ids": [doc_id],  # Single document per job
                "parser": parser,
                "pages_per_split": pages_per_split,
                "overlap_pages": overlap_pages,
                "page_selection": page_selection
            }

            # Submit job
            job_id = unified_job_service._submit_job(
                job_type=JobType.DOCUMENT_PROCESSING,
                parameters=job_parameters,
                priority=JobPriority.NORMAL,
                user_id=user_id,
                start_time=datetime.now()
            )
            
            if job_id:
                job_ids.append(job_id)
                logger.info(f"Submitted parse job {job_id} for document {doc_id}")
            else:
                st.error(f"❌ Failed to submit job for document {doc_id}")

        if not job_ids:
            st.error("❌ Failed to submit any parsing jobs")
            return

        # Monitor all jobs
        st.info(f"📋 Submitted {len(job_ids)} parsing jobs. Monitoring progress...")
        
        all_success = True
        failed_jobs = []
        
        for job_id in job_ids:
            result = unified_job_service._monitor_job(
                job_id=job_id,
                start_time=datetime.now(),
                status_text=st.empty(),
                job_id_text=st.empty(),
                timeout=1800,  # 30 minutes per job
                poll_interval=3
            )
            
            if not result.success:
                all_success = False
                failed_jobs.append({"job_id": job_id, "error": result.error_message})
        
        if all_success:
            st.success(f"✅ Successfully parsed all {len(document_ids_to_parse)} document(s)!")
            # Refresh to show updated document list
            st.rerun()
        else:
            error_details = "\n".join([f"Job {j['job_id']}: {j['error']}" for j in failed_jobs])
            st.error(f"❌ {len(failed_jobs)}/{len(job_ids)} parsing jobs failed:\n{error_details}")

    except Exception as e:
        # DEBUG: [DEBUG_20251003_PDF_PARSE] - Exception during parsing
        logger.error(f'{{"timestamp": "{datetime.now().isoformat()}", "level": "ERROR", "component": "document_processing_service", "message": "Exception during parsing job submission", "context": {{"project_id": {project_id}, "error": "{str(e)}"}}}}')
        st.error(f"Error during document parsing job submission: {e}")
        logger.debug(f"GUI DEBUG: Exception during parsing job submission: {e}")
        import traceback
        logger.debug(f"GUI DEBUG: Traceback: {traceback.format_exc()}")


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

        user_id = _get_user_id()
        if not user_id:
            st.error("❌ You must be logged in to chunk documents.")
            return

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

        # Use unified job service for submission and monitoring
        result = unified_job_service.submit_and_monitor(
            job_type=JobType.DOCUMENT_PROCESSING,
            parameters=job_parameters,
            priority=JobPriority.NORMAL,
            user_id=user_id,
            title=f"Chunking {len(document_ids)} Document(s)",
            timeout=3600,  # 60 minutes
            poll_interval=3
        )

        if result.success:
            st.success(f"✅ Successfully chunked {len(document_ids)} document(s)!")
            # Refresh to show updated chunks
            st.rerun()
        else:
            st.error(f"❌ Chunking failed: {result.error_message}")

    except Exception as e:
        st.error(f"Error during document chunking: {e}")


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