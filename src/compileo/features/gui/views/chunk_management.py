"""
Chunk management view for Compileo GUI.
Allows viewing and deleting chunks for documents.
"""

import streamlit as st
import pandas as pd
import time
import re
from src.compileo.features.gui.services.api_client import api_client
from src.compileo.features.gui.state.session_state import session_state

def parse_chunk_ids(input_str: str, chunks_data: list) -> list[str]:
    """
    Parse chunk ID input supporting formats:
    - Single: "1" (chunk index)
    - Multiple: "1, 2, 4"
    - Ranges: "1-10"
    - Combined: "1-5, 8, 10-12"
    
    Resolves indices to actual chunk UUIDs.
    """
    target_indices = set()
    
    # Split by comma and process each part
    parts = [part.strip() for part in input_str.split(',') if part.strip()]

    for part in parts:
        if '-' in part:
            # Handle range
            try:
                start, end = [int(x.strip()) for x in part.split('-')]
                if start <= end:
                    target_indices.update(range(start, end + 1))
            except ValueError:
                continue
        else:
            # Handle single number
            try:
                target_indices.add(int(part.strip()))
            except ValueError:
                continue

    # Map indices to UUIDs
    # chunks_data is expected to be a list of dicts with 'id' and 'chunk_index'
    chunk_uuids = []
    
    # Create a mapping of index to ID
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

def render_chunk_management():
    """Render the chunk management view."""
    st.title("View & Manage Chunks")
    st.markdown("View and manage generated chunks for your documents.")
    # Version for cache busting
    st.write("Version: 1.1")

    # 1. Select Project
    with st.spinner("Loading projects..."):
        try:
            projects_resp = api_client.get("/api/v1/projects")
            projects = projects_resp.get("projects", []) if projects_resp else []
        except Exception as e:
            st.error(f"Failed to load projects: {e}")
            return

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

    selected_project_name = st.selectbox("Select Project", options=list(project_options.keys()), index=index)
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
    selected_doc_label = st.selectbox("Select Document", options=list(doc_options.keys()))
    document_id = doc_options[selected_doc_label]

    # 3. List Chunks
    st.divider()
    st.subheader(f"Chunks for Document {document_id}")

    # Refresh button
    if st.button("🔄 Refresh Chunks"):
        st.rerun()

    with st.spinner("Loading chunks..."):
        try:
            # API returns { "document_id": ..., "chunks": [...], "total": ... }
            response = api_client.get(f"/api/v1/chunks/document/{document_id}")
            chunks = response.get("chunks", []) if isinstance(response, dict) else response
        except Exception as e:
            st.error(f"Failed to load chunks: {e}")
            return

    if not chunks:
        st.info("No chunks found for this document.")
        st.markdown("Go to **Document Processing** to chunk this document.")
    else:
        st.success(f"Found {len(chunks)} chunks.")
        
        # Delete All Button
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🗑️ Delete All Chunks", type="primary", width='stretch'):
                # Use a confirmation dialog pattern if Streamlit supported modal easily,
                # but here we'll just use session state for confirmation or just do it if "Are you sure" is implied by the button label being red/primary?
                # Actually standard pattern is click -> separate confirm button appears.
                st.session_state.show_delete_confirm = True

        if st.session_state.get("show_delete_confirm"):
            st.warning("Are you sure you want to delete ALL chunks for this document? This cannot be undone.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Yes, Delete All"):
                    try:
                        api_client.delete(f"/api/v1/chunks/document/{document_id}")
                        st.success("All chunks deleted successfully.")
                        st.session_state.show_delete_confirm = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete chunks: {e}")
            with c2:
                if st.button("❌ Cancel"):
                    st.session_state.show_delete_confirm = False
                    st.rerun()

        # Prepare chunk list for processing
        chunk_list = chunks
        if isinstance(chunks, dict):
             chunk_list = chunks.get("chunks", [])

        # Delete Specific Chunks
        st.markdown("**Delete Specific Chunks**")
        st.markdown("Enter **Chunk Indices** (not UUIDs) to delete. Example: `1, 3-5` deletes Chunk 1, 3, 4, and 5.")
        chunk_ids_input = st.text_input("Chunk Indices to delete", placeholder="e.g., 1, 3-5", key="chunk_ids_input")

        if st.button("🗑️ Delete Selected Chunks"):
            if not chunk_ids_input.strip():
                st.error("Please enter chunk indices to delete.")
                # We don't return here to allow the rest of the UI to render, but stop processing this action
            else:
                try:
                    # Parse the input to extract chunk UUIDs based on indices
                    chunk_ids_to_delete = parse_chunk_ids(chunk_ids_input, chunk_list)
                    
                    if not chunk_ids_to_delete:
                        st.error("No valid chunk indices found matching the current document's chunks. Please check your input and available chunk indices.")
                    else:
                        # Delete chunks in batch
                        st.info(f"Deleting {len(chunk_ids_to_delete)} chunks...")
                        response = api_client.delete("/api/v1/chunks/batch", data={"chunk_ids": chunk_ids_to_delete})
                        
                        if response and isinstance(response, dict):
                            deleted = response.get("deleted_count", 0)
                            if deleted > 0:
                                st.success(f"Successfully deleted {deleted} chunks.")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.warning("Operation completed but reported 0 deleted chunks. They might have been already deleted.")
                        else:
                             st.warning("Received unexpected response from server. Check logs.")

                except Exception as e:
                    st.error(f"Failed to delete chunks: {str(e)}")
        
        # Display Chunks Table
        if chunk_list:
            data = []
            for c in chunk_list:
                content = ""
                full_content = ""
                if isinstance(c, dict):
                    # Prefer full content if available, fallback to preview
                    # API returns 'text' field for full content from ChunkLoader
                    full_content = c.get("text") or c.get("content") or c.get("content_preview", "")
                    if not full_content:
                        # Fallback for old chunks or data issues
                        full_content = "(No content available)"
                    
                    content = full_content
                else:
                    content = str(c)
                    full_content = str(c)
                
                # Get chunk index - prefer 'chunk_index' but handle potential missing keys
                chunk_index = c.get("chunk_index") if isinstance(c, dict) else "N/A"
                if chunk_index is None:
                    chunk_index = "N/A"
                
                # Calculate token count estimation if not present (approx 4 chars per token)
                token_count = c.get("token_count") if isinstance(c, dict) else "N/A"
                if (token_count is None or token_count == "N/A") and content:
                    token_count = f"~{len(content) // 4}"

                data.append({
                    "UUID": c.get("id") if isinstance(c, dict) else c,
                    "Index": chunk_index,
                    "Tokens": token_count,
                    "Content": content
                })
            
            # Sort by index
            try:
                data.sort(key=lambda x: int(x["Index"]) if isinstance(x["Index"], (int, str)) and str(x["Index"]).isdigit() else 0)
            except ValueError:
                pass # Keep original order if sorting fails
            
            # Display chunks as expandable sections
            st.write(f"Displaying {len(data)} chunks:")
            for item in data:
                # Show Index prominently
                title = f"Chunk {item['Index']} (Tokens: {item['Tokens']})"
                with st.expander(title):
                    st.caption(f"UUID: {item['UUID']}")
                    st.text_area(
                        "Content",
                        value=item['Content'],
                        height=200,
                        disabled=True,
                        key=f"content_{item['UUID']}"
                    )

