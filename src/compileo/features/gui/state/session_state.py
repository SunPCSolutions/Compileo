"""
Session state management for Streamlit GUI.
"""

import streamlit as st
from typing import Any, Dict, Optional
from datetime import datetime

class SessionState:
    """Wrapper class for Streamlit session state management."""

    def __init__(self):
        """Initialize session state with default values."""
        # Always ensure initialization, even if called multiple times
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize all session state variables."""
        st.session_state.initialized = True
        st.session_state.user_id = "default_user"  # Default user for single-user GUI operation
        st.session_state.api_key = None
        st.session_state.current_project = None
        st.session_state.selected_project_id = None
        st.session_state.processing_job_id = None
        st.session_state.generation_job_id = None
        st.session_state.wizard_state = {}
        st.session_state.last_activity = datetime.utcnow()
        st.session_state.notifications = []
        st.session_state.job_status_cache = {}
        st.session_state.overlap_pages = 1
        st.session_state.job_just_completed = False

    @property
    def user_id(self) -> Optional[str]:
        """Get current user ID."""
        return st.session_state.get('user_id')

    @user_id.setter
    def user_id(self, value: Optional[str]):
        """Set current user ID."""
        st.session_state.user_id = value

    @property
    def api_key(self) -> Optional[str]:
        """Get API key."""
        return st.session_state.get('api_key')

    @api_key.setter
    def api_key(self, value: Optional[str]):
        """Set API key."""
        st.session_state.api_key = value

    @property
    def current_project(self) -> Optional[Dict]:
        """Get current project."""
        return st.session_state.get('current_project')

    @current_project.setter
    def current_project(self, value: Optional[Dict]):
        """Set current project."""
        st.session_state.current_project = value

    @property
    def selected_project_id(self) -> Optional[int]:
        """Get selected project ID."""
        return st.session_state.get('selected_project_id')

    @selected_project_id.setter
    def selected_project_id(self, value: Optional[int]):
        """Set selected project ID."""
        st.session_state.selected_project_id = value

    @property
    def processing_job_id(self) -> Optional[str]:
        """Get processing job ID."""
        return st.session_state.get('processing_job_id')

    @processing_job_id.setter
    def processing_job_id(self, value: Optional[str]):
        """Set processing job ID."""
        st.session_state.processing_job_id = value

    @property
    def generation_job_id(self) -> Optional[str]:
        """Get generation job ID."""
        return st.session_state.get('generation_job_id')

    @generation_job_id.setter
    def generation_job_id(self, value: Optional[str]):
        """Set generation job ID."""
        st.session_state.generation_job_id = value

    @property
    def wizard_state(self) -> Dict:
        """Get wizard state."""
        return st.session_state.get('wizard_state', {})

    @wizard_state.setter
    def wizard_state(self, value: Dict):
        """Set wizard state."""
        st.session_state.wizard_state = value


    @property
    def overlap_pages(self) -> int:
        """Get overlap pages."""
        return st.session_state.get('overlap_pages', 1)

    @overlap_pages.setter
    def overlap_pages(self, value: int):
        """Set overlap pages."""
        st.session_state.overlap_pages = value

    @property
    def job_just_completed(self) -> bool:
        """Get job just completed flag."""
        return st.session_state.get('job_just_completed', False)

    @job_just_completed.setter
    def job_just_completed(self, value: bool):
        """Set job just completed flag."""
        st.session_state.job_just_completed = value

    @property
    def current_page(self) -> str:
        """Get current page."""
        return st.session_state.get('current_page', 'home')

    @current_page.setter
    def current_page(self, value: str):
        """Set current page."""
        st.session_state.current_page = value

    @property
    def notifications(self) -> list:
        """Get notifications."""
        return st.session_state.get('notifications', [])

    def add_notification(self, message: str, type: str = "info"):
        """Add a notification."""
        notifications = st.session_state.get('notifications', [])
        notifications.append({
            'message': message,
            'type': type,
            'timestamp': datetime.utcnow()
        })
        st.session_state.notifications = notifications

    def clear_notifications(self):
        """Clear all notifications."""
        st.session_state.notifications = []

    def cache_job_status(self, job_id: str, status: Dict, timeout_seconds: int = 30):
        """Cache job status with expiration time."""
        cache = st.session_state.get('job_status_cache', {})
        status_with_expiry = {
            **status,
            '_cached_at': datetime.utcnow(),
            '_expires_at': datetime.utcnow().timestamp() + timeout_seconds
        }
        cache[job_id] = status_with_expiry
        st.session_state.job_status_cache = cache

    def get_cached_job_status(self, job_id: str) -> Optional[Dict]:
        """Get cached job status if not expired."""
        cache = st.session_state.get('job_status_cache', {})
        if job_id not in cache:
            return None

        cached_data = cache[job_id]
        expires_at = cached_data.get('_expires_at', 0)
        current_time = datetime.utcnow().timestamp()

        if current_time > expires_at:
            # Expired, remove from cache
            del cache[job_id]
            st.session_state.job_status_cache = cache
            return None

        # Return status without cache metadata
        status_copy = cached_data.copy()
        status_copy.pop('_cached_at', None)
        status_copy.pop('_expires_at', None)
        return status_copy

    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get cached job status (legacy method)."""
        return self.get_cached_job_status(job_id)

    def set_job_status(self, job_id: str, status: Dict):
        """Cache job status (legacy method)."""
        self.cache_job_status(job_id, status)

    def add_user_job(self, user_id: str, job_info: Dict):
        """Add a job to user's active jobs list."""
        user_jobs_key = f'user_jobs_{user_id}'
        user_jobs = st.session_state.get(user_jobs_key, [])
        user_jobs.append(job_info)
        st.session_state[user_jobs_key] = user_jobs

    def get_user_jobs(self, user_id: str) -> list:
        """Get user's active jobs list."""
        user_jobs_key = f'user_jobs_{user_id}'
        return st.session_state.get(user_jobs_key, [])

    def remove_user_job(self, user_id: str, job_id: str):
        """Remove a job from user's active jobs list."""
        user_jobs_key = f'user_jobs_{user_id}'
        user_jobs = st.session_state.get(user_jobs_key, [])
        user_jobs = [job for job in user_jobs if job.get('job_id') != job_id]
        st.session_state[user_jobs_key] = user_jobs

    def clear_expired_cache(self, max_age_seconds: int = 1800):
        """Clear expired cache entries."""
        cache = st.session_state.get('job_status_cache', {})
        current_time = datetime.utcnow().timestamp()

        expired_keys = []
        for job_id, cached_data in cache.items():
            expires_at = cached_data.get('_expires_at', 0)
            if current_time > expires_at:
                expired_keys.append(job_id)

        for key in expired_keys:
            del cache[key]

        st.session_state.job_status_cache = cache

    def reset(self):
        """Reset session state to initial values."""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        self._initialize_session_state()

# Global session state instance
session_state = SessionState()