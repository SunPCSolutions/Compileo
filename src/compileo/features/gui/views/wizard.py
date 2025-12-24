"""
Dataset creation wizard page for the Compileo GUI.
"""

import streamlit as st
from typing import Dict, Any

from src.compileo.features.gui.state.wizard_state import wizard_state
from ....core.logging import get_logger
from src.compileo.features.gui.components.wizard.project_selection import render_project_selection
from src.compileo.features.gui.components.wizard_steps import (
    render_combined_setup,
    render_edit_taxonomy_step,
    render_generation_params,
    render_review_generate
)

logger = get_logger(__name__)

STEP_NAMES = [
    "Project Selection",
    "Parse & Chunk & Taxonomy",
    "Edit Taxonomy",
    "Generation Parameters",
    "Review & Generate"
]

STEP_ICONS = [
    "📁", "⚙️", "🏗️", "🔧", "✅"
]

def render_step_indicator():
    """Render the step indicator/progress bar with clickable navigation."""
    st.markdown("### Dataset Creation Wizard")

    cols = st.columns(len(STEP_NAMES))
    for i, (name, icon) in enumerate(zip(STEP_NAMES, STEP_ICONS)):
        with cols[i]:
            if i < wizard_state.current_step:
                # Completed step - clickable
                if st.button(f"{icon} {name}", key=f"nav_step_{i}", help=f"Go to {name}", width='stretch'):
                    wizard_state.current_step = i
                    st.rerun()
            elif i == wizard_state.current_step:
                # Current step - highlighted but not clickable
                st.info(f"{icon} **{name}**")
            else:
                # Future step - all clickable, validation happens within the step
                if st.button(f"{icon} {name}", key=f"nav_step_{i}", help=f"Go to {name}", width='stretch'):
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
                    if current_step_name == "combined_processing" and "processing_config" in st.session_state:
                        # Save processing config from combined step
                        wizard_state.set_step_data("processing_config", st.session_state.processing_config)
                    elif current_step_name == "edit_taxonomy" and "unified_taxonomy" in st.session_state:
                        # Save taxonomy from unified builder
                        wizard_state.update_step_data("taxonomy_selection", "selected_taxonomy", st.session_state.unified_taxonomy)
                    elif current_step_name == "generation_params" and "generation_params" in st.session_state:
                        params = st.session_state.generation_params
                        wizard_state.set_step_data("generation_params", params)
                        
                        # Auto-save parameters to database for persistence
                        try:
                            project_id = wizard_state.get_step_data("project_selection").get("project_id")
                            if project_id:
                                from src.compileo.features.gui.services.api_client import api_client
                                dataset_params_request = {
                                    "project_id": str(project_id),
                                    "purpose": params.get("custom_purpose") or params.get("purpose") or "Wizard Generation",
                                    "audience": params.get("custom_audience") or params.get("audience") or "General",
                                    "extraction_rules": "default",
                                    "dataset_format": params.get("output_format", "jsonl"),
                                    "question_style": "factual",
                                    "answer_style": "comprehensive",
                                    "negativity_ratio": 0.1,
                                    "data_augmentation": "none",
                                    "custom_audience": params.get("custom_audience", ""),
                                    "custom_purpose": params.get("custom_purpose", ""),
                                    "complexity_level": params.get("complexity_level", "intermediate"),
                                    "domain": params.get("domain", "general")
                                }
                                api_client.post("/api/v1/datasets/parameters", data=dataset_params_request)
                                logger.debug(f"Auto-saved parameters for project {project_id}")
                        except Exception as e:
                            logger.debug(f"Failed to auto-save parameters: {e}")

                    wizard_state.mark_step_completed(wizard_state.current_step)
                    wizard_state.next_step()
                    st.rerun()
            else:
                st.button("Next ➡️", disabled=True, width='stretch')
                # Provide specific guidance based on current step
                current_step_name = wizard_state.current_step_name
                if current_step_name == "combined_processing":  # Combined setup
                    st.warning("⚙️ Please complete the setup by uploading documents and generating chunks before proceeding.")
                    st.info("💡 Use the 'Start Processing' button above to parse and chunk your documents.")
                elif current_step_name == "edit_taxonomy":
                    st.warning("🏷️ Please select or generate a taxonomy before proceeding.")
                elif current_step_name == "generation_params":
                    st.warning("🔧 Please configure dataset generation parameters before proceeding.")
                else:
                    st.warning("Please complete the current step before proceeding.")

def render_current_step():
    """Render the current step content."""
    step_name = wizard_state.current_step_name

    if step_name == "project_selection":
        render_project_selection()
    elif step_name == "combined_processing":  # Now combined setup
        render_combined_setup()
    elif step_name == "edit_taxonomy":
        render_edit_taxonomy_step()
    elif step_name == "generation_params":
        render_generation_params()
    elif step_name == "review_generate":
        render_review_generate()
    else:
        st.error(f"Unknown step: {step_name}")

def render_wizard():
    """Render the dataset creation wizard page."""
    # Check for refresh flag
    if wizard_state.needs_refresh:
        wizard_state.needs_refresh = False
        st.rerun()

    st.title("🧙 Dataset Creation Wizard")
    st.markdown("Create high-quality datasets through our guided workflow")

    # Step indicator
    render_step_indicator()

    # Current step content
    with st.container():
        render_current_step()

    st.divider()

    # Navigation
    render_navigation_buttons()

