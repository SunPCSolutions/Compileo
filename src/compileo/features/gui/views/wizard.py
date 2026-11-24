"""
Dataset creation wizard page for the Compileo GUI.
"""

import streamlit as st
from typing import Dict, Any

from src.compileo.features.gui.state.wizard_state import wizard_state
from src.compileo.features.gui.services.api_client import api_client, APIError
from src.compileo.features.gui.components.wizard.project_selection import render_project_selection
from src.compileo.features.gui.components.wizard_steps import (
    render_document_upload,
    render_processing_config,
    render_taxonomy_selection,
    render_generation_params,
    render_review_generate
)

STEP_NAMES = [
    "Project Selection",
    "Document Upload",
    "Processing Configuration",
    "Taxonomy Selection",
    "Generation Parameters",
    "Review & Generate"
]

STEP_ICONS = [
    "📁", "📤", "⚙️", "🏷️", "🔧", "✅"
]

def render_step_indicator():
    """Render the step indicator/progress bar with clickable navigation."""
    st.markdown("### Dataset Creation Wizard")

    cols = st.columns(len(STEP_NAMES))
    for i, (name, icon) in enumerate(zip(STEP_NAMES, STEP_ICONS)):
        with cols[i]:
            if i < wizard_state.current_step:
                # Completed step - clickable
                if st.button(f"{icon} {name}", key=f"nav_step_{i}", help=f"Go to {name}", use_container_width=True):
                    wizard_state.current_step = i
                    st.rerun()
            elif i == wizard_state.current_step:
                # Current step - highlighted but not clickable
                st.info(f"{icon} **{name}**")
            else:
                # Future step - all clickable, validation happens within the step
                if st.button(f"{icon} {name}", key=f"nav_step_{i}", help=f"Go to {name}", use_container_width=True):
                    wizard_state.current_step = i
                    st.rerun()

    st.divider()

def render_navigation_buttons():
    """Render navigation buttons for the wizard."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if wizard_state.current_step > 0:
            if st.button("⬅️ Previous", width='stretch'):
                wizard_state.previous_step()
                st.rerun()

    with col2:
        # Progress indicator
        progress = (wizard_state.current_step + 1) / len(STEP_NAMES)
        st.progress(progress)
        st.caption(f"Step {wizard_state.current_step + 1} of {len(STEP_NAMES)}")

    with col3:
        if wizard_state.current_step < len(STEP_NAMES) - 1:
            # Check if current step is valid before allowing next
            if wizard_state.validate_current_step():
                if st.button("Next ➡️", width='stretch'):
                    # Auto-save current step data before proceeding
                    current_step_name = wizard_state.current_step_name
                    if current_step_name == "processing_config" and "processing_config" in st.session_state:
                        wizard_state.set_step_data("processing_config", st.session_state.processing_config)
                    elif current_step_name == "generation_params" and "generation_params" in st.session_state:
                        wizard_state.set_step_data("generation_params", st.session_state.generation_params)

                    wizard_state.mark_step_completed(wizard_state.current_step)
                    wizard_state.next_step()
                    st.rerun()
            else:
                st.button("Next ➡️", disabled=True, width='stretch')
                st.warning("Please complete the current step before proceeding.")

def render_current_step():
    """Render the current step content."""
    step_name = wizard_state.current_step_name

    if step_name == "project_selection":
        render_project_selection()
    elif step_name == "document_upload":
        render_document_upload()
    elif step_name == "processing_config":
        render_processing_config()
    elif step_name == "taxonomy_selection":
        render_taxonomy_selection()
    elif step_name == "generation_params":
        render_generation_params()
    elif step_name == "review_generate":
        render_review_generate()
    else:
        st.error(f"Unknown step: {step_name}")

def _validate_selected_project():
    """Validate that the selected project still exists. Returns True if valid, False if reset was needed."""
    project_data = wizard_state.get_step_data("project_selection")
    project_id = project_data.get("project_id")
    
    if not project_id:
        return True  # No project selected yet, that's fine
    
    try:
        import requests
        headers = {}
        if api_client.jwt_token:
            headers["Authorization"] = f"Bearer {api_client.jwt_token}"
            
        # Try to fetch the project directly with requests to handle 404 without showing st.error
        res = requests.get(f"{api_client.base_url}/api/v1/projects/{project_id}", headers=headers)
        
        if res.status_code == 404:
            # Project was deleted - reset wizard
            st.warning("⚠️ The previously selected project has been deleted. Starting fresh...")
            wizard_state.reset()
            # Clear any session state related to old project
            for key in list(st.session_state.keys()):
                if key.startswith('selected_document_ids') or key.startswith('uploaded_file_names') or key.startswith('wizard_'):
                    del st.session_state[key]
            st.rerun()
            return False
            
    except Exception:
        # Network errors - let it proceed and handle in the steps
        pass
    
    return True

def render_wizard():
    """Render the dataset creation wizard page."""
    st.title("🧙 Dataset Creation Wizard")
    st.markdown("Create high-quality datasets through our guided workflow")

    # Validate selected project still exists
    if not _validate_selected_project():
        return  # Reset was triggered, stop rendering

    # Step indicator
    render_step_indicator()

    # Current step content
    with st.container():
        render_current_step()

    st.divider()

    # Navigation
    render_navigation_buttons()

