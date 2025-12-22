"""
Projects page for managing Compileo projects.
"""

import os
import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional, List

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from src.compileo.features.gui.state.session_state import session_state
from src.compileo.features.gui.services.api_client import api_client, APIError


def render_bulk_selection_controls(projects: List[Dict[str, Any]]):
    """Render bulk selection controls for projects."""
    # Initialize selected projects in session state if not exists
    if "selected_projects" not in st.session_state:
        st.session_state.selected_projects = set()

    # Select All checkbox
    all_selected = len(st.session_state.selected_projects) == len(projects) and len(projects) > 0
    select_all = st.checkbox(
        "Select All",
        value=all_selected,
        key="select_all_projects",
        help="Select/deselect all projects"
    )

    # Update selection when select all changes
    if select_all and not all_selected:
        st.session_state.selected_projects = {p["id"] for p in projects}
        st.rerun()
    elif not select_all and all_selected:
        st.session_state.selected_projects.clear()
        st.rerun()

    # Delete Selected button
    selected_count = len(st.session_state.selected_projects)
    if selected_count > 0:
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button(
                f"🗑️ Delete Selected ({selected_count})",
                type="primary",
                help=f"Delete {selected_count} selected project(s)"
            ):
                st.session_state.bulk_deleting_projects = True
                st.rerun()
        with col2:
            if st.button("Clear Selection"):
                st.session_state.selected_projects.clear()
                st.rerun()

    # Bulk delete confirmation modal
    if st.session_state.get("bulk_deleting_projects", False):
        render_bulk_delete_confirmation(projects)


def render_bulk_delete_confirmation(projects: List[Dict[str, Any]]):
    """Render bulk delete confirmation modal."""
    selected_ids = st.session_state.selected_projects
    selected_projects = [p for p in projects if p["id"] in selected_ids]

    st.subheader("🗑️ Delete Multiple Projects")
    st.warning(f"Are you sure you want to delete **{len(selected_projects)}** project(s)?")
    st.error("This action cannot be undone. All associated documents and datasets will be removed.")

    # Show selected projects
    with st.expander("Selected Projects"):
        for project in selected_projects:
            st.write(f"• **{project.get('name', 'Unknown')}** (ID: {project.get('id')})")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("❌ Yes, Delete All", type="primary"):
            try:
                with st.spinner(f"Deleting {len(selected_projects)} projects..."):
                    response = api_client.delete("/api/v1/projects", data={
                        "project_ids": list(selected_ids)
                    })

                deleted_count = len(response.get("deleted", []))
                failed_count = len(response.get("failed", []))

                if failed_count == 0:
                    st.success(f"Successfully deleted {deleted_count} project(s) and all associated data!")
                else:
                    st.warning(f"Deleted {deleted_count} project(s), but {failed_count} failed. Check the errors below.")
                    for failure in response.get("failed", []):
                        st.error(f"Project {failure['id']}: {failure['error']}")

                # Clear selection and modal state
                st.session_state.selected_projects.clear()
                del st.session_state.bulk_deleting_projects
                st.rerun()

            except APIError as e:
                st.error(f"Failed to delete projects: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

    with col2:
        if st.button("⬅️ Cancel"):
            del st.session_state.bulk_deleting_projects
            st.rerun()


def render_projects_list():
    """Render the projects listing with pagination and actions."""
    st.subheader("Your Projects")

    # Search and filter controls
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        search_query = st.text_input("Search projects", placeholder="Enter project name...")

    with col2:
        sort_by = st.selectbox("Sort by", ["Newest", "Oldest", "Name A-Z", "Name Z-A"])

    with col3:
        items_per_page = st.selectbox("Items per page", [10, 20, 50], index=1)

    # Load projects
    try:
        with st.spinner("Loading projects..."):
            # Get first page to start
            response = api_client.get("/api/v1/projects", params={"per_page": items_per_page, "page": 1})
            projects = response.get("projects", [])
            total_projects = response.get("total", 0)
            total_pages = (total_projects + items_per_page - 1) // items_per_page

        if not projects:
            st.info("No projects found. Create your first project using the 'Create Project' tab.")
            return

        # Filter projects based on search
        if search_query:
            projects = [p for p in projects if search_query.lower() in p.get("name", "").lower()]

        # Sort projects
        if sort_by == "Newest":
            projects.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        elif sort_by == "Oldest":
            projects.sort(key=lambda x: x.get("created_at", ""))
        elif sort_by == "Name A-Z":
            projects.sort(key=lambda x: x.get("name", "").lower())
        elif sort_by == "Name Z-A":
            projects.sort(key=lambda x: x.get("name", "").lower(), reverse=True)

        # Bulk selection controls
        render_bulk_selection_controls(projects)

        # Display projects
        for project in projects:
            render_project_card(project)

        # Pagination info
        st.caption(f"Showing {len(projects)} of {total_projects} projects")

    except APIError as e:
        st.error(f"Failed to load projects: {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

def render_project_card(project: Dict[str, Any]):
    """Render a single project card with actions."""
    with st.container():
        # Checkbox column
        col0, col1, col2, col3 = st.columns([0.5, 3, 1, 1])

        with col0:
            # Checkbox for selection
            project_id = project['id']
            is_selected = project_id in st.session_state.get('selected_projects', set())
            if st.checkbox(
                f"Select {project.get('name', 'project')}",
                value=is_selected,
                key=f"select_{project_id}",
                label_visibility="collapsed"
            ):
                if project_id not in st.session_state.selected_projects:
                    st.session_state.selected_projects.add(project_id)
                    st.rerun()
            else:
                if project_id in st.session_state.selected_projects:
                    st.session_state.selected_projects.remove(project_id)
                    st.rerun()

        with col1:
            st.subheader(f"📁 {project.get('name', 'Unnamed Project')}")
            st.write(f"**ID:** {project.get('id')}")
            st.write(f"**Created:** {format_datetime(project.get('created_at'))}")
            st.write(f"**Documents:** {project.get('document_count', 0)}")
            st.write(f"**Datasets:** {project.get('dataset_count', 0)}")

            if project.get('description'):
                st.write(f"**Description:** {project.get('description')}")

        with col2:
            if st.button("👁️ View Details", key=f"view_{project['id']}"):
                st.session_state.selected_project = project
                st.rerun()

            if st.button("📝 Edit", key=f"edit_{project['id']}"):
                st.session_state.editing_project = project
                st.rerun()

        with col3:
            if st.button("🗑️ Delete", key=f"delete_{project['id']}"):
                st.session_state.deleting_project = project
                st.rerun()

        st.divider()

def render_create_project():
    """Render the project creation form."""
    st.subheader("Create New Project")

    with st.form("create_project_form"):
        name = st.text_input("Project Name", help="Enter a unique name for your project")
        description = st.text_area("Description (Optional)", help="Describe the purpose of this project")

        submitted = st.form_submit_button("Create Project")

        if submitted:
            if not name.strip():
                st.error("Project name is required")
                return

            try:
                with st.spinner("Creating project..."):
                    response = api_client.post("/api/v1/projects", data={
                        "name": name.strip(),
                        "description": description.strip() if description else None
                    })

                st.success(f"Project '{response['name']}' created successfully!")
                st.rerun()

            except APIError as e:
                st.error(f"Failed to create project: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

def render_project_details():
    """Render detailed view of a selected project."""
    if "selected_project" not in st.session_state:
        st.info("Select a project to view details.")
        return

    project = st.session_state.selected_project

    st.subheader(f"📁 {project.get('name', 'Unnamed Project')}")

    # Project info
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**ID:** {project.get('id')}")
        st.write(f"**Created:** {format_datetime(project.get('created_at'))}")
        st.write(f"**Status:** {project.get('status', 'Active')}")

    with col2:
        st.metric("Documents", project.get('document_count', 0))
        st.metric("Datasets", project.get('dataset_count', 0))

    if project.get('description'):
        st.write(f"**Description:** {project.get('description')}")

    # Action buttons
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📝 Edit Project"):
            st.session_state.editing_project = project
            del st.session_state.selected_project
            st.rerun()

    with col2:
        if st.button("📄 View Documents"):
            st.session_state.viewing_documents = project
            st.rerun()

    with col3:
        if st.button("📊 View Datasets"):
            st.session_state.viewing_datasets = project
            st.rerun()

    with col4:
        if st.button("⬅️ Back to List"):
            del st.session_state.selected_project
            st.rerun()

    st.divider()

    # Show documents if requested
    if "viewing_documents" in st.session_state and st.session_state.viewing_documents['id'] == project['id']:
        render_project_documents(project)

    # Show datasets if requested
    elif "viewing_datasets" in st.session_state and st.session_state.viewing_datasets['id'] == project['id']:
        render_project_datasets(project)

def render_project_documents(project: Dict[str, Any]):
    """Render documents for a project."""
    st.subheader("📄 Project Documents")

    try:
        with st.spinner("Loading documents..."):
            response = api_client.get(f"/api/v1/projects/{project['id']}/documents")
            documents = response.get("documents", [])

        if not documents:
            st.info("No documents found for this project.")
        else:
            for doc in documents:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{doc.get('file_name', 'Unknown')}**")
                        # Try to get file size from source_file_path if available
                        file_size = "N/A"
                        source_path = doc.get('source_file_path')
                        if source_path and os.path.exists(source_path):
                            try:
                                file_size = f"{os.path.getsize(source_path)} bytes"
                            except OSError:
                                file_size = "N/A"
                        st.write(f"Size: {file_size}")
                        st.write(f"Uploaded: {format_datetime(doc.get('created_at'))}")
                    with col2:
                        if st.button("👁️ View", key=f"view_doc_{doc.get('id')}"):
                            st.info("Document viewing not implemented yet")
                    st.divider()

    except APIError as e:
        st.error(f"Failed to load documents: {str(e)}")

def render_project_datasets(project: Dict[str, Any]):
    """Render datasets for a project."""
    st.subheader("📊 Project Datasets")

    try:
        with st.spinner("Loading datasets..."):
            response = api_client.get(f"/api/v1/projects/{project['id']}/datasets")
            datasets = response.get("datasets", [])

        if not datasets:
            st.info("No datasets found for this project.")
        else:
            for dataset in datasets:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{dataset.get('name', 'Unknown')}**")
                        st.write(f"Type: {dataset.get('output_type', 'Unknown')}")
                        st.write(f"Created: {format_datetime(dataset.get('created_at'))}")
                    with col2:
                        if st.button("📥 Download", key=f"download_{dataset.get('id')}"):
                            st.info("Dataset download not implemented yet")
                    st.divider()

    except APIError as e:
        st.error(f"Failed to load datasets: {str(e)}")

def render_edit_project():
    """Render the project editing form."""
    if "editing_project" not in st.session_state:
        st.info("Select a project to edit.")
        return

    project = st.session_state.editing_project

    st.subheader(f"📝 Edit Project: {project.get('name')}")

    with st.form("edit_project_form"):
        name = st.text_input("Project Name", value=project.get('name', ''))
        description = st.text_area("Description", value=project.get('description', ''))

        col1, col2 = st.columns(2)

        with col1:
            submitted = st.form_submit_button("💾 Save Changes")

        with col2:
            if st.form_submit_button("❌ Cancel"):
                del st.session_state.editing_project
                st.rerun()
                return

        if submitted:
            if not name.strip():
                st.error("Project name is required")
                return

            try:
                with st.spinner("Updating project..."):
                    response = api_client.put(f"/api/v1/projects/{project['id']}", data={
                        "name": name.strip(),
                        "description": description.strip() if description else None
                    })

                st.success("Project updated successfully!")
                del st.session_state.editing_project
                st.rerun()

            except APIError as e:
                st.error(f"Failed to update project: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

def render_delete_confirmation():
    """Render project deletion confirmation."""
    if "deleting_project" not in st.session_state:
        return

    project = st.session_state.deleting_project

    st.subheader("🗑️ Delete Project")
    st.warning(f"Are you sure you want to delete the project **{project.get('name')}**?")
    st.error("This action cannot be undone. All associated documents and datasets will be removed.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("❌ Yes, Delete", type="primary"):
            try:
                with st.spinner("Deleting project..."):
                    api_client.delete(f"/api/v1/projects/{project['id']}")

                st.success("Project deleted successfully!")
                del st.session_state.deleting_project
                st.rerun()

            except APIError as e:
                st.error(f"Failed to delete project: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

    with col2:
        if st.button("⬅️ Cancel"):
            del st.session_state.deleting_project
            st.rerun()

def render_project_statistics():
    """Render project statistics dashboard."""
    st.subheader("📊 Project Statistics")

    try:
        with st.spinner("Loading statistics..."):
            # Get all projects
            response = api_client.get("/api/v1/projects", params={"per_page": 1000})  # Get all projects
            projects = response.get("projects", [])

        if not projects:
            st.info("No projects available for statistics.")
            return

        # Calculate statistics
        total_projects = len(projects)
        total_documents = sum(p.get('document_count', 0) for p in projects)
        total_datasets = sum(p.get('dataset_count', 0) for p in projects)

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Projects", total_projects)

        with col2:
            st.metric("Total Documents", total_documents)

        with col3:
            st.metric("Total Datasets", total_datasets)

        with col4:
            avg_docs_per_project = total_documents / total_projects if total_projects > 0 else 0
            st.metric("Avg Docs/Project", f"{avg_docs_per_project:.1f}")

        st.divider()

        # Projects table
        st.subheader("Project Overview")

        if PANDAS_AVAILABLE:
            # Prepare data for table
            table_data = []
            for project in projects:
                table_data.append({
                    "Name": project.get('name', 'Unknown'),
                    "ID": project.get('id'),
                    "Documents": project.get('document_count', 0),
                    "Datasets": project.get('dataset_count', 0),
                    "Created": format_datetime(project.get('created_at'))
                })

            df = pd.DataFrame(table_data)
            st.dataframe(df, width='stretch')

            # Charts
            if len(projects) > 1:
                st.subheader("Visualizations")

                col1, col2 = st.columns(2)

                with col1:
                    st.bar_chart(df.set_index('Name')[['Documents', 'Datasets']])

                with col2:
                    # Documents vs Datasets comparison
                    comparison_data = pd.DataFrame({
                        'Metric': ['Documents', 'Datasets'],
                        'Count': [total_documents, total_datasets]
                    })
                    st.bar_chart(comparison_data.set_index('Metric'))
        else:
            # Fallback display without pandas
            st.info("Install pandas for enhanced table and chart display.")

            # Simple table display
            for project in projects:
                with st.container():
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.write(f"**{project.get('name', 'Unknown')}**")
                    with col2:
                        st.write(f"ID: {project.get('id')}")
                    with col3:
                        st.write(f"Docs: {project.get('document_count', 0)}")
                    with col4:
                        st.write(f"Datasets: {project.get('dataset_count', 0)}")
                    st.divider()

    except APIError as e:
        st.error(f"Failed to load statistics: {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

def format_datetime(dt_str: Optional[str]) -> str:
    """Format datetime string for display."""
    if not dt_str:
        return "Unknown"

    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M')
    except:
        return dt_str

# Handle different view states
def render_projects():
    """Main render function with state management."""
    # Check for different view states
    if "selected_project" in st.session_state:
        render_project_details()
    elif "editing_project" in st.session_state:
        render_edit_project()
    elif "deleting_project" in st.session_state:
        render_delete_confirmation()
    else:
        # Main projects view with tabs
        st.title("📁 Projects")
        st.markdown("Manage your Compileo projects")

        # Check if API key is set
        if not session_state.api_key:
            st.warning("Please set your API key in the sidebar to access project management features.")
            return

        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📋 Projects List", "➕ Create Project", "📊 Statistics"])

        with tab1:
            render_projects_list()

        with tab2:
            render_create_project()

        with tab3:
            render_project_statistics()