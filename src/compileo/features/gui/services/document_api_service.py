"""Document API service module for Compileo GUI."""

from typing import List, Dict, Any

from ..services.api_client import api_client
import streamlit as st


def get_projects() -> List[Dict[str, Any]]:
    """Get list of available projects from API."""
    try:
        response = api_client.get("/api/v1/projects")
        if response and "projects" in response:
            return response["projects"]
        return []
    except Exception as e:
        st.error(f"Failed to load projects: {e}")
        return []


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


def delete_document(document_id: int):
    """Delete a document."""
    try:
        response = api_client.delete(f"/api/v1/documents/{document_id}")
        if response and "message" in response:
            st.success(response["message"])
            st.rerun()
        else:
            st.error("Failed to delete document")
    except Exception as e:
        st.error(f"Failed to delete document: {e}")