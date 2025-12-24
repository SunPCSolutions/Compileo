"""
Job Status Display Components for Compileo GUI.
Provides visual status indicators and detailed status displays for jobs.
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import time

from src.compileo.features.jobhandle.models import JobStatus, JobPriority, JobType
from src.compileo.features.gui.utils.job_queue_utils import get_job_queue_manager_safe


class JobStatusDisplay:
    """Component for displaying job status with visual indicators."""

    @staticmethod
    def render_status_badge(status: JobStatus, size: str = "normal") -> str:
        """Render a status badge with appropriate styling."""
        status_config = JobStatusDisplay._get_status_config(status)

        size_styles = {
            "small": "font-size: 0.8rem; padding: 0.2rem 0.5rem;",
            "normal": "font-size: 0.9rem; padding: 0.3rem 0.75rem;",
            "large": "font-size: 1rem; padding: 0.5rem 1rem;"
        }

        style = f"""
            background-color: {status_config['bg_color']};
            color: {status_config['text_color']};
            border: 1px solid {status_config['border_color']};
            border-radius: 0.375rem;
            font-weight: 500;
            display: inline-flex;
            align-items: center;
            gap: 0.375rem;
            {size_styles.get(size, size_styles['normal'])}
        """

        return f"""
        <span style="{style}">
            <span>{status_config['icon']}</span>
            <span>{status_config['label']}</span>
        </span>
        """

    @staticmethod
    def render_priority_indicator(priority: JobPriority) -> str:
        """Render a priority indicator."""
        priority_config = JobStatusDisplay._get_priority_config(priority)

        style = f"""
            color: {priority_config['color']};
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        """

        return f"""
        <span style="{style}" title="Priority: {priority_config['label']}">
            {priority_config['icon']} {priority_config['label']}
        </span>
        """

    @staticmethod
    def render_progress_bar(progress: float, status: JobStatus, show_percentage: bool = True) -> str:
        """Render a progress bar for job completion."""
        if not isinstance(progress, (int, float)) or progress < 0:
            progress = 0
        elif progress > 100:
            progress = 100

        # Color based on status
        if status == JobStatus.RUNNING:
            color = "#3b82f6"  # blue
        elif status == JobStatus.COMPLETED:
            color = "#10b981"  # green
        elif status == JobStatus.FAILED:
            color = "#ef4444"  # red
        else:
            color = "#6b7280"  # gray

        percentage_text = f"{progress:.1f}%" if show_percentage else ""

        return f"""
        <div style="width: 100%; background-color: #e5e7eb; border-radius: 0.375rem; height: 0.75rem; overflow: hidden;">
            <div style="width: {progress}%; background-color: {color}; height: 100%; transition: width 0.3s ease;">
            </div>
        </div>
        <div style="text-align: center; margin-top: 0.25rem; font-size: 0.8rem; color: #6b7280;">
            {percentage_text}
        </div>
        """

    @staticmethod
    def render_job_card(job, show_details: bool = False, show_actions: bool = True):
        """Render a comprehensive job card."""
        # Header with status and priority
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown(f"**{job.job_type.value.replace('_', ' ').title()}**")
            st.caption(f"ID: {job.job_id[:12]}...")

        with col2:
            st.markdown(
                JobStatusDisplay.render_status_badge(job.status, "small"),
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                JobStatusDisplay.render_priority_indicator(job.priority),
                unsafe_allow_html=True
            )

        # Progress bar for running jobs
        if job.status == JobStatus.RUNNING and hasattr(job, 'progress'):
            st.markdown(
                JobStatusDisplay.render_progress_bar(job.progress, job.status),
                unsafe_allow_html=True
            )

        # Timing information
        timing_info = JobStatusDisplay._get_timing_info(job)
        if timing_info:
            st.caption(timing_info)

        # Show details if requested
        if show_details:
            with st.expander("Details"):
                JobStatusDisplay._render_job_details(job)

        # Action buttons if requested
        if show_actions:
            JobStatusDisplay._render_job_actions(job)

    @staticmethod
    def render_job_list(jobs, title: str = "Jobs", max_display: int = 10):
        """Render a list of jobs with status indicators."""
        if not jobs:
            st.info(f"No {title.lower()} found")
            return

        st.subheader(title)

        # Sort jobs by priority and creation time
        sorted_jobs = sorted(jobs, key=lambda j: (j.priority.value, j.created_at), reverse=True)

        for job in sorted_jobs[:max_display]:
            with st.container():
                JobStatusDisplay.render_job_card(job, show_details=False, show_actions=False)
                st.markdown("---")

        if len(jobs) > max_display:
            st.caption(f"And {len(jobs) - max_display} more...")

    @staticmethod
    def _get_status_config(status: JobStatus) -> Dict[str, str]:
        """Get configuration for status display."""
        configs = {
            JobStatus.PENDING: {
                'icon': '⏳',
                'label': 'Pending',
                'bg_color': '#fef3c7',
                'text_color': '#92400e',
                'border_color': '#f59e0b'
            },
            JobStatus.RUNNING: {
                'icon': '⚙️',
                'label': 'Running',
                'bg_color': '#dbeafe',
                'text_color': '#1e40af',
                'border_color': '#3b82f6'
            },
            JobStatus.COMPLETED: {
                'icon': '✅',
                'label': 'Completed',
                'bg_color': '#d1fae5',
                'text_color': '#065f46',
                'border_color': '#10b981'
            },
            JobStatus.FAILED: {
                'icon': '❌',
                'label': 'Failed',
                'bg_color': '#fee2e2',
                'text_color': '#991b1b',
                'border_color': '#ef4444'
            },
            JobStatus.CANCELLED: {
                'icon': '🚫',
                'label': 'Cancelled',
                'bg_color': '#f3f4f6',
                'text_color': '#374151',
                'border_color': '#6b7280'
            },
            JobStatus.SCHEDULED: {
                'icon': '📅',
                'label': 'Scheduled',
                'bg_color': '#e0e7ff',
                'text_color': '#3730a3',
                'border_color': '#6366f1'
            }
        }
        return configs.get(status, {
            'icon': '❓',
            'label': 'Unknown',
            'bg_color': '#f3f4f6',
            'text_color': '#374151',
            'border_color': '#6b7280'
        })

    @staticmethod
    def _get_priority_config(priority: JobPriority) -> Dict[str, str]:
        """Get configuration for priority display."""
        configs = {
            "urgent": {'icon': '🔴', 'label': 'Urgent', 'color': '#ef4444'},
            "high": {'icon': '🟠', 'label': 'High', 'color': '#f59e0b'},
            "normal": {'icon': '🔵', 'label': 'Normal', 'color': '#3b82f6'},
            "low": {'icon': '⚪', 'label': 'Low', 'color': '#6b7280'}
        }

        priority_key = priority.value.lower() if hasattr(priority, 'value') else str(priority).lower()
        return configs.get(priority_key, configs['normal'])

    @staticmethod
    def _get_timing_info(job) -> str:
        """Get timing information for job display."""
        info_parts = []

        if job.created_at:
            created_str = job.created_at.strftime("%H:%M:%S")
            info_parts.append(f"Created: {created_str}")

        if job.started_at:
            started_str = job.started_at.strftime("%H:%M:%S")
            info_parts.append(f"Started: {started_str}")

        if job.completed_at:
            completed_str = job.completed_at.strftime("%H:%M:%S")
            info_parts.append(f"Completed: {completed_str}")

        # Calculate duration if applicable
        if job.started_at and job.completed_at:
            duration = job.completed_at - job.started_at
            info_parts.append(f"Duration: {duration.seconds}s")
        elif job.started_at and job.status == JobStatus.RUNNING:
            duration = datetime.now() - job.started_at
            info_parts.append(f"Running: {duration.seconds}s")

        return " | ".join(info_parts)

    @staticmethod
    def _render_job_details(job):
        """Render detailed job information."""
        # Basic information
        st.write(f"**Job ID:** {job.job_id}")
        st.write(f"**Type:** {job.job_type.value}")
        st.write(f"**Priority:** {job.priority.value}")
        st.write(f"**Status:** {job.status.value}")

        # Worker information
        if hasattr(job, 'worker_id') and job.worker_id:
            st.write(f"**Worker:** {job.worker_id}")

        # Parameters
        if hasattr(job, 'parameters') and job.parameters:
            st.subheader("Parameters")
            st.json(job.parameters)

        # Error information
        if hasattr(job, 'error') and job.error:
            st.error(f"**Error:** {job.error}")

        # Metrics
        if hasattr(job, 'metrics') and job.metrics:
            st.subheader("Performance Metrics")
            metrics = job.metrics
            col1, col2 = st.columns(2)

            with col1:
                if hasattr(metrics, 'cpu_usage_percent'):
                    st.metric("CPU Usage", f"{metrics.cpu_usage_percent:.1f}%")
                if hasattr(metrics, 'memory_usage_mb'):
                    st.metric("Memory", f"{metrics.memory_usage_mb:.0f} MB")

            with col2:
                if hasattr(metrics, 'execution_time_seconds'):
                    st.metric("Execution Time", f"{metrics.execution_time_seconds:.2f}s")
                if hasattr(metrics, 'items_processed'):
                    st.metric("Items Processed", metrics.items_processed)

    @staticmethod
    def _render_job_actions(job):
        """Render action buttons for job management."""
        col1, col2, col3 = st.columns(3)

        # Cancel button
        if job.status in [JobStatus.PENDING, JobStatus.RUNNING, JobStatus.SCHEDULED]:
            with col1:
                if st.button("❌ Cancel", key=f"cancel_status_display_{job.job_id}"):
                    from src.compileo.features.gui.utils.job_queue_utils import cancel_job_safe
                    if cancel_job_safe(job.job_id):
                        st.success("Job cancelled successfully")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("Failed to cancel job")

        # Restart button for failed jobs
        if job.status == JobStatus.FAILED:
            with col2:
                if st.button("🔄 Restart", key=f"restart_{job.job_id}"):
                    from src.compileo.features.gui.utils.job_queue_utils import perform_job_operation_safe
                    if perform_job_operation_safe('restart_job', job.job_id):
                        st.success("Job restarted successfully")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("Failed to restart job")

        # Details button
        with col3:
            if st.button("📋 Details", key=f"details_{job.job_id}"):
                st.session_state.current_page = "job_management"
                st.rerun()


# Convenience functions for easy use
def render_status_badge(status: JobStatus, size: str = "normal") -> str:
    """Convenience function to render status badge."""
    return JobStatusDisplay.render_status_badge(status, size)


def render_priority_indicator(priority: JobPriority) -> str:
    """Convenience function to render priority indicator."""
    return JobStatusDisplay.render_priority_indicator(priority)


def render_progress_bar(progress: float, status: JobStatus, show_percentage: bool = True) -> str:
    """Convenience function to render progress bar."""
    return JobStatusDisplay.render_progress_bar(progress, status, show_percentage)


def render_job_card(job, show_details: bool = False, show_actions: bool = True):
    """Convenience function to render job card."""
    JobStatusDisplay.render_job_card(job, show_details, show_actions)


def render_job_list(jobs, title: str = "Jobs", max_display: int = 10):
    """Convenience function to render job list."""
    JobStatusDisplay.render_job_list(jobs, title, max_display)