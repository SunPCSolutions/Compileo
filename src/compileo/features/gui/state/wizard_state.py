"""
Wizard-specific state management for dataset creation workflow.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import streamlit as st
from ....core.logging import get_logger

logger = get_logger(__name__)

class WizardState:
    """Manages state for the dataset creation wizard."""

    STEPS = [
        "project_selection",
        "combined_processing",  # Parse & Chunk & Taxonomy
        "edit_taxonomy",        # Manual/Hybrid editing
        "generation_params",
        "review_generate",
        "refinement"
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
                'needs_refresh': False,
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

    @property
    def needs_refresh(self) -> bool:
        """Check if wizard needs refresh."""
        return st.session_state.wizard.get('needs_refresh', False)

    @needs_refresh.setter
    def needs_refresh(self, value: bool):
        """Set refresh flag."""
        st.session_state.wizard['needs_refresh'] = value

    def validate_current_step(self) -> bool:
        """Validate current step data."""
        step_name = self.current_step_name

        # Basic validation logic
        if step_name == "project_selection":
            step_data = self.get_step_data(step_name)
            return 'project_id' in step_data
        elif step_name == "combined_processing":  # Parse & Chunk & Taxonomy
            # Check if chunks have been generated for selected documents
            processing_data = self.get_step_data("processing_config")
            selected_document_ids = processing_data.get("selected_document_ids", [])
            chunk_job_id = processing_data.get("chunk_job_id")

            # If we have selected documents, check for chunks
            if selected_document_ids:
                try:
                    from ..services.api_client import api_client

                    # Check if chunks exist for selected documents
                    chunks_found = False
                    total_chunks = 0

                    for doc_id in selected_document_ids:
                        try:
                            chunks_response = api_client.get(f"/api/v1/chunks/document/{doc_id}")
                            if chunks_response and isinstance(chunks_response, dict):
                                doc_chunks = chunks_response.get("chunks", [])
                                if doc_chunks and len(doc_chunks) > 0:
                                    chunks_found = True
                                    total_chunks += len(doc_chunks)
                        except Exception as e:
                            # Log API failure but continue checking other documents
                            logger.debug(f"Failed to check chunks for document {doc_id}: {e}")
                            pass

                    # Allow progression if we found any chunks
                    # This handles cases where chunking completed successfully
                    if chunks_found:
                        logger.debug(f"Found {total_chunks} chunks across {len(selected_document_ids)} documents - allowing progression")
                        return True

                    # If chunking job was started but no chunks found yet, allow progression anyway
                    # This handles timing issues where chunks might be created after validation
                    if chunk_job_id:
                        logger.debug(f"Chunking job {chunk_job_id} was started - allowing progression despite no chunks found yet")
                        return True

                    # No chunks found and no chunking job - don't allow progression
                    logger.debug(f"No chunks found for {len(selected_document_ids)} documents and no chunking job started")
                    return False

                except Exception as e:
                    # If we can't check chunks at all, allow progression (fail open)
                    logger.debug(f"Exception during chunk validation: {e} - allowing progression")
                    return True
            else:
                # No documents selected - allow progression (user can go back if needed)
                return True
        elif step_name == "processing_config":  # Skip - now combined
            # Check session state for processing config data since it's stored there
            # until Next is pressed
            import streamlit as st
            step_data = st.session_state.get('processing_config', {})
            # Check for all required AI model selections and chunk strategy
            required_fields = ['parsing_model', 'chunking_model', 'classification_model', 'chunk_strategy']
            config_valid = all(field in step_data for field in required_fields)

            if not config_valid:
                return False

            # Additionally check if chunks are available for selected documents
            # This ensures users have generated chunks before proceeding
            project_data = self.get_step_data("project_selection")
            project_id = project_data.get("project_id")
            selected_document_ids = step_data.get("selected_document_ids", [])

            if project_id and selected_document_ids:
                try:
                    from ..services.api_client import api_client

                    # Check if chunks exist for selected documents
                    all_have_chunks = True
                    for doc_id in selected_document_ids:
                        chunks_response = api_client.get(f"/api/v1/chunks/document/{doc_id}")
                        if chunks_response:
                            chunks = chunks_response.get("chunks", [])
                            if not chunks:  # No chunks found for this document
                                all_have_chunks = False
                                break
                        else:
                            all_have_chunks = False
                            break

                    return all_have_chunks
                except Exception:
                    # If we can't check chunks, allow progression (fail open)
                    return True
            else:
                return False
        elif step_name == "edit_taxonomy":
            taxonomy_data = self.get_step_data("taxonomy_selection")
            return 'selected_taxonomy' in taxonomy_data
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