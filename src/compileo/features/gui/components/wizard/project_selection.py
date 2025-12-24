"""
Project selection step for the dataset creation wizard.
"""

import streamlit as st
from src.compileo.features.gui.state.wizard_state import wizard_state
from src.compileo.features.gui.services.api_client import api_client, APIError


def render_project_selection():
    """Render the project selection step."""
    st.header("📁 Step 1: Project Selection")

    st.markdown("Select an existing project or create a new one for your dataset.")

    # Get available projects
    try:
        projects_response = api_client.get("/api/v1/projects")
        projects = projects_response.get("projects", []) if projects_response else []
    except APIError as e:
        st.error(f"Failed to load projects: {e}")
        projects = []

    # Project selection
    if projects:
        st.subheader("Select Existing Project")
        project_options = {f"{p['name']} (ID: {p['id']})": p for p in projects}
        selected_project_key = st.selectbox(
            "Choose a project",
            options=list(project_options.keys()),
            help="Select an existing project to work with"
        )

        if selected_project_key:
            selected_project = project_options[selected_project_key]
            wizard_state.update_step_data("project_selection", "project_id", selected_project["id"])
            wizard_state.update_step_data("project_selection", "project_name", selected_project["name"])

            st.success(f"Selected project: **{selected_project['name']}**")
            st.info(f"Documents: {selected_project.get('document_count', 0)} | Datasets: {selected_project.get('dataset_count', 0)}")

    st.divider()

    # Create new project
    st.subheader("Or Create New Project")
    with st.form("create_project_form"):
        new_project_name = st.text_input("Project Name", help="Enter a unique name for your project")
        new_project_description = st.text_area("Description (optional)", height=100)

        submitted = st.form_submit_button("Create Project")
        if submitted:
            if not new_project_name.strip():
                st.error("Project name is required")
            else:
                try:
                    new_project = api_client.post("/api/v1/projects", {
                        "name": new_project_name.strip(),
                        "description": new_project_description.strip() or None
                    })

                    if new_project:
                        st.success(f"Project '{new_project['name']}' created successfully!")
                        wizard_state.update_step_data("project_selection", "project_id", new_project["id"])
                        wizard_state.update_step_data("project_selection", "project_name", new_project["name"])
                        st.rerun()
                    else:
                        st.error("Failed to create project: API returned no response")

                except APIError as e:
                    st.error(f"Failed to create project: {e}")