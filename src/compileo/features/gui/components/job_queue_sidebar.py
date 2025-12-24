"""
Job Queue Sidebar Component for Compileo GUI.
Provides real-time job queue monitoring and management in the sidebar.
"""

import streamlit as st
from datetime import datetime
from typing import List, Dict, Any
import time

from src.compileo.features.jobhandle.models import JobStatus, JobType
from src.compileo.features.gui.utils.job_queue_utils import get_job_queue_manager_safe


class JobQueueSidebar:
    """Sidebar component for job queue monitoring and controls."""

    def __init__(self):
        self.last_refresh = 0
        self.refresh_interval = 10  # seconds
        self.expanded_jobs = set()  # Track expanded job details

    def render(self):
        """Render the job queue sidebar component."""
        with st.sidebar:
            # Job Queue Header
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                <span style="font-size: 1.2rem;">🔄</span>
                <span style="font-weight: 600; color: var(--text-primary);">Job Queue</span>
            </div>
            """, unsafe_allow_html=True)

            # Quick Stats
            self._render_quick_stats()

            # Active Jobs List
            self._render_active_jobs()

            # Quick Actions
            self._render_quick_actions()

    def _render_quick_stats(self):
        """Render quick queue statistics."""
        try:
            job_queue_manager = get_job_queue_manager_safe()
            if not job_queue_manager:
                st.error("Job queue manager not available")
                return
            stats = job_queue_manager.get_queue_stats()

            col1, col2 = st.columns(2)
            with col1:
                pending = stats.get('pending_jobs', 0)
                running = stats.get('running_jobs', 0)

                st.metric(
                    "Active",
                    f"{running}",
                    help=f"{pending} pending jobs"
                )

            with col2:
                total = stats.get('total_jobs', 0)
                st.metric(
                    "Total",
                    f"{total}",
                    help="Total jobs in queue"
                )

        except Exception as e:
            st.error(f"Failed to load queue stats: {e}")

    def _render_active_jobs(self):
        """Render list of active jobs."""
        try:
            job_queue_manager = get_job_queue_manager_safe()
            if not job_queue_manager:
                st.error("Job queue manager not available")
                return

            # Get active jobs (pending + running)
            pending_jobs = job_queue_manager.get_jobs_by_status(JobStatus.PENDING, 5)
            running_jobs = job_queue_manager.get_jobs_by_status(JobStatus.RUNNING, 5)

            active_jobs = running_jobs + pending_jobs

            if not active_jobs:
                st.info("No active jobs")
                return

            # Sort by priority and creation time
            active_jobs.sort(key=lambda j: (j.priority.value, j.created_at), reverse=True)

            st.markdown("**Active Jobs:**")

            for job in active_jobs[:5]:  # Show top 5
                self._render_job_item(job)

            if len(active_jobs) > 5:
                st.caption(f"+{len(active_jobs) - 5} more jobs")

        except Exception as e:
            st.error(f"Failed to load active jobs: {e}")

    def _render_job_item(self, job):
        """Render a single job item in the sidebar."""
        # Status indicator
        status_icon = self._get_status_icon(job.status)
        priority_color = self._get_priority_color(job.priority)

        # Job type display
        job_type_display = job.job_type.value.replace('_', ' ').title()

        # Time display
        time_display = self._format_time(job.created_at)

        # Progress for running jobs
        progress_display = ""
        if job.status == JobStatus.RUNNING and hasattr(job, 'progress'):
            progress_display = f" ({job.progress:.0f}%)"

        # Create expandable item
        job_key = f"job_{job.job_id}"

        with st.expander(
            f"{status_icon} {job_type_display}{progress_display}",
            expanded=job_key in self.expanded_jobs
        ):
            # Job details
            st.caption(f"ID: {job.job_id[:8]}...")
            st.caption(f"Priority: {job.priority.value}")
            st.caption(f"Created: {time_display}")

            if job.status == JobStatus.RUNNING:
                if hasattr(job, 'progress'):
                    st.progress(job.progress / 100)
                if hasattr(job, 'started_at') and job.started_at:
                    elapsed = datetime.utcnow() - job.started_at
                    st.caption(f"Running: {elapsed.seconds}s")

            # Action buttons
            col1, col2 = st.columns(2)

            with col1:
                if job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
                    if st.button("❌ Cancel", key=f"cancel_sidebar_{job.job_id}", help="Cancel this job"):
                        from src.compileo.features.gui.utils.job_queue_utils import cancel_job_safe
                        if cancel_job_safe(job.job_id):
                            st.success("Job cancelled")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("Failed to cancel")

            with col2:
                if st.button("📋 Details", key=f"details_{job.job_id}", help="View full details"):
                    # This would open the full job management view
                    st.session_state.current_page = "job_management"
                    st.rerun()

    def _render_quick_actions(self):
        """Render quick action buttons."""
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📊 Manage", key="manage_jobs", help="Open full job management"):
                st.session_state.current_page = "job_management"
                st.rerun()

        with col2:
            if st.button("🔄 Refresh", key="refresh_queue", help="Refresh job queue"):
                st.rerun()

    def _get_status_icon(self, status: JobStatus) -> str:
        """Get status icon for job."""
        icons = {
            JobStatus.PENDING: "⏳",
            JobStatus.RUNNING: "⚙️",
            JobStatus.COMPLETED: "✅",
            JobStatus.FAILED: "❌",
            JobStatus.CANCELLED: "🚫",
            JobStatus.SCHEDULED: "📅"
        }
        return icons.get(status, "❓")

    def _get_priority_color(self, priority) -> str:
        """Get color for priority level."""
        colors = {
            "urgent": "#ef4444",    # red
            "high": "#f59e0b",      # orange
            "normal": "#3b82f6",    # blue
            "low": "#6b7280"        # gray
        }
        return colors.get(priority.value.lower(), "#6b7280")

    def _format_time(self, dt: datetime) -> str:
        """Format datetime for display."""
        if not dt:
            return "N/A"

        # dt is stored as UTC, so compare with UTC now to avoid timezone issues
        now = datetime.utcnow()
        diff = now - dt

        if diff.seconds < 60:
            return "Just now"
        elif diff.seconds < 3600:
            return f"{diff.seconds // 60}m ago"
        elif diff.days == 0:
            return f"{diff.seconds // 3600}h ago"
        else:
            return f"{diff.days}d ago"


# Global instance
job_queue_sidebar = JobQueueSidebar()


def render_job_queue_sidebar():
    """Main function to render the job queue sidebar."""
    job_queue_sidebar.render()