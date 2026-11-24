"""
Wizard-specific state management for dataset creation workflow.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import streamlit as st

class WizardState:
    """Manages state for the dataset creation wizard."""

    STEPS = [
        "project_selection",
        "document_upload",
        "processing_config",
        "taxonomy_selection",
        "generation_params",
        "review_generate"
    ]

    def __init__(self):
        """Initialize wizard state."""
        self.initialize()

    def initialize(self):
        """Explicitly initialize the wizard state in the session."""
        if 'wizard' not in st.session_state:
            st.session_state.wizard = {
                'current_step': 0,
                'completed_steps': [],
                'data': {},
                'errors': {},
                'last_updated': datetime.utcnow()
            }

    @property
    def current_step(self) -> int:
        """Get current step index."""
        return st.session_state.wizard.get('current_step', 0)

    @current_step.setter
    def current_step(self, value: int):
        """Set current step index."""
        st.session_state.wizard['current_step'] = value
        st.session_state.wizard['last_updated'] = datetime.utcnow()

    @property
    def current_step_name(self) -> str:
        """Get current step name."""
        return self.STEPS[self.current_step]

    @property
    def completed_steps(self) -> List[int]:
        """Get list of completed step indices."""
        return st.session_state.wizard.get('completed_steps', [])

    def mark_step_completed(self, step_index: int):
        """Mark a step as completed."""
        completed = st.session_state.wizard.get('completed_steps', [])
        if step_index not in completed:
            completed.append(step_index)
            st.session_state.wizard['completed_steps'] = completed
            st.session_state.wizard['last_updated'] = datetime.utcnow()

    def is_step_completed(self, step_index: int) -> bool:
        """Check if a step is completed."""
        return step_index in self.completed_steps

    def can_access_step(self, step_index: int) -> bool:
        """Check if a step can be accessed (all previous steps completed)."""
        if step_index == 0:
            return True
        return all(i in self.completed_steps for i in range(step_index))

    def next_step(self):
        """Move to next step if possible."""
        if self.current_step < len(self.STEPS) - 1:
            self.current_step += 1

    def previous_step(self):
        """Move to previous step if possible."""
        if self.current_step > 0:
            self.current_step -= 1

    def go_to_step(self, step_index: int):
        """Go to specific step if accessible."""
        if 0 <= step_index < len(self.STEPS) and self.can_access_step(step_index):
            self.current_step = step_index

    def get_step_data(self, step_name: str) -> Dict[str, Any]:
        """Get data for a specific step."""
        data = st.session_state.wizard.get('data', {})
        return data.get(step_name, {})

    def set_step_data(self, step_name: str, data: Dict[str, Any]):
        """Set data for a specific step."""
        wizard_data = st.session_state.wizard.get('data', {})
        wizard_data[step_name] = data
        st.session_state.wizard['data'] = wizard_data
        st.session_state.wizard['last_updated'] = datetime.utcnow()

    def update_step_data(self, step_name: str, key: str, value: Any):
        """Update a specific key in step data."""
        step_data = self.get_step_data(step_name)
        step_data[key] = value
        self.set_step_data(step_name, step_data)

    def get_step_error(self, step_name: str) -> Optional[str]:
        """Get error for a specific step."""
        errors = st.session_state.wizard.get('errors', {})
        return errors.get(step_name)

    def set_step_error(self, step_name: str, error: Optional[str]):
        """Set error for a specific step."""
        errors = st.session_state.wizard.get('errors', {})
        if error:
            errors[step_name] = error
        elif step_name in errors:
            del errors[step_name]
        st.session_state.wizard['errors'] = errors
        st.session_state.wizard['last_updated'] = datetime.utcnow()

    def clear_step_error(self, step_name: str):
        """Clear error for a specific step."""
        self.set_step_error(step_name, None)

    def has_errors(self) -> bool:
        """Check if wizard has any errors."""
        errors = st.session_state.wizard.get('errors', {})
        return len(errors) > 0

    def get_all_data(self) -> Dict[str, Any]:
        """Get all wizard data."""
        return st.session_state.wizard.get('data', {})

    def validate_current_step(self) -> bool:
        """Validate current step data."""
        step_name = self.current_step_name

        # Basic validation logic
        if step_name == "project_selection":
            step_data = self.get_step_data(step_name)
            return 'project_id' in step_data
        elif step_name == "document_upload":
            step_data = self.get_step_data(step_name)
            # Allow progression if either uploaded files exist OR existing documents exist
            has_uploaded = len(step_data.get('uploaded_files', [])) > 0
            has_existing = len(step_data.get('existing_documents', [])) > 0
            return has_uploaded or has_existing
        elif step_name == "processing_config":
            # Check session state for processing config data since it's stored there
            # until Next is pressed
            import streamlit as st
            step_data = st.session_state.get('processing_config', {})
            # Check for all required AI model selections and chunk strategy
            required_fields = ['parsing_model', 'chunking_model', 'classification_model', 'chunk_strategy']
            has_config = all(field in step_data for field in required_fields)
            # Also check that chunking is completed
            wizard_data = self.get_step_data("processing_config")
            chunking_status = wizard_data.get("chunking_status", "not_started") if wizard_data else "not_started"
            return has_config and chunking_status == "completed"
        elif step_name == "taxonomy_selection":
            step_data = self.get_step_data(step_name)
            return 'selected_taxonomy' in step_data
        elif step_name == "generation_params":
            # Check session state for generation params data since it's stored there
            # until Next is pressed
            import streamlit as st
            step_data = st.session_state.get('generation_params', {})
            # Basic validation - generation_mode must be selected (required field)
            # Allow progression even with validation errors but show warnings
            return 'generation_mode' in step_data and step_data['generation_mode'] is not None

        return True

    def reset(self):
        """Reset wizard state."""
        st.session_state.wizard = {
            'current_step': 0,
            'completed_steps': [],
            'data': {},
            'errors': {},
            'last_updated': datetime.utcnow()
        }

# Global wizard state instance
wizard_state = WizardState()