"""
Job Management GUI View for Compileo.
Provides comprehensive job queue monitoring, management, and analytics interface.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import time
import json

from src.compileo.features.gui.services.job_monitoring_service import monitor_job_synchronously
from .enhanced_job_queue import perform_health_check, cleanup_old_jobs
from .models import JobStatus, JobPriority, JobType


class JobManagementView:
    """Streamlit view for comprehensive job queue management."""

    def __init__(self):
        self.refresh_interval = 5  # seconds
        self.last_refresh = 0

    def render(self):
        """Render the main job management interface."""
        st.title("🔄 Job Queue Management")
        st.markdown("Monitor, manage, and analyze background extraction jobs")

        # Manual refresh button
        if st.button("🔄 Refresh Dashboard", type="secondary", width='stretch'):
            st.rerun()

        # Create tabs for different management sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "🎯 Active Jobs", "📋 Job History",
            "⚙️ Management", "📈 Analytics"
        ])

        with tab1:
            self.render_overview_tab()

        with tab2:
            self.render_active_jobs_tab()

        with tab3:
            self.render_job_history_tab()

        with tab4:
            self.render_management_tab()

        with tab5:
            self.render_analytics_tab()

    def render_overview_tab(self):
        """Render the overview dashboard."""
        st.header("Queue Overview")

        # Health check
        health_status = perform_health_check()

        # Status indicator
        status_color = {
            'healthy': '🟢',
            'warning': '🟡',
            'critical': '🔴'
        }.get(health_status['status'], '⚪')

        st.subheader(f"{status_color} System Health: {health_status['status'].title()}")

        if health_status['issues']:
            with st.expander("⚠️ Issues Detected"):
                for issue in health_status['issues']:
                    st.warning(issue)

        # Queue statistics
        stats = health_status['queue_stats']

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Pending Jobs",
                stats.get('pending_jobs', 0),
                help="Jobs waiting to be processed"
            )

        with col2:
            st.metric(
                "Running Jobs",
                stats.get('running_jobs', 0),
                help="Jobs currently being processed"
            )

        with col3:
            st.metric(
                "Scheduled Jobs",
                stats.get('scheduled_jobs', 0),
                help="Jobs scheduled for future execution"
            )

        with col4:
            st.metric(
                "Total Jobs",
                stats.get('total_jobs', 0),
                help="Total jobs in the system"
            )

        # Resource usage
        st.subheader("Resource Usage")

        col1, col2, col3 = st.columns(3)

        with col1:
            cpu_usage = stats.get('cpu_usage_percent', 0)
            st.metric(
                "CPU Usage",
                f"{cpu_usage:.1f}%",
                help="Current CPU utilization"
            )
            st.progress(min(cpu_usage / 100, 1.0))

        with col2:
            memory_usage = stats.get('memory_usage_mb', 0)
            st.metric(
                "Memory Usage",
                f"{memory_usage:.0f} MB",
                help="Current memory utilization"
            )
            # Assuming 1GB limit for progress bar
            st.progress(min(memory_usage / 1024, 1.0))

        with col3:
            cache_size = stats.get('cache_size', 0)
            st.metric(
                "Cache Size",
                cache_size,
                help="Number of cached results"
            )

        # Recent activity
        st.subheader("Recent Activity")
        self.render_recent_activity()

    def render_recent_activity(self):
        """Render recent job activity."""
        # Get recent jobs (last 10)
        recent_jobs = []

        # This would need to be implemented to get recent jobs from the queue
        # For now, show placeholder
        if not recent_jobs:
            st.info("No recent job activity to display")
            return

        # Display recent jobs in a table
        df = pd.DataFrame(recent_jobs)
        st.dataframe(df, width='stretch')

    def render_active_jobs_tab(self):
        """Render active jobs management."""
        st.header("Active Jobs Management")

        # Get active jobs using safe accessor
        from src.compileo.features.gui.utils.job_queue_utils import get_job_queue_manager_safe
        job_queue_manager = get_job_queue_manager_safe()
        if not job_queue_manager:
            st.error("Job queue manager is currently unavailable. Please try refreshing the page.")
            return

        pending_jobs = job_queue_manager.get_jobs_by_status(JobStatus.PENDING, 100)
        running_jobs = job_queue_manager.get_jobs_by_status(JobStatus.RUNNING, 100)
        scheduled_jobs = job_queue_manager.queue.scheduler.scheduled_jobs.values() if hasattr(job_queue_manager.queue, 'scheduler') else []

        # Job type filter
        job_types = [e.value for e in JobType] + ["all"]
        selected_type = st.selectbox("Filter by Job Type", job_types, index=len(job_types)-1)

        # Priority filter
        priorities = [e.value for e in JobPriority] + ["all"]
        selected_priority = st.selectbox("Filter by Priority", priorities, index=len(priorities)-1)

        # Pending jobs
        if pending_jobs:
            st.subheader("⏳ Pending Jobs")
            self.render_job_table(pending_jobs, "pending", selected_type, selected_priority)

        # Running jobs
        if running_jobs:
            st.subheader("⚙️ Running Jobs")
            self.render_job_table(running_jobs, "running", selected_type, selected_priority)

        # Scheduled jobs
        if scheduled_jobs:
            st.subheader("📅 Scheduled Jobs")
            scheduled_list = list(scheduled_jobs)
            self.render_job_table(scheduled_list, "scheduled", selected_type, selected_priority)

        if not pending_jobs and not running_jobs and not scheduled_jobs:
            st.info("No active jobs found")

    def render_job_table(self, jobs: List, status: str, type_filter: str, priority_filter: str):
        """Render a table of jobs with management actions."""
        # Filter jobs
        filtered_jobs = []
        for job in jobs:
            if type_filter != "all" and job.job_type != type_filter:
                continue
            if priority_filter != "all" and job.priority.value != priority_filter:
                continue
            filtered_jobs.append(job)

        if not filtered_jobs:
            st.info(f"No {status} jobs match the current filters")
            return

        # Create data for display
        job_data = []
        for job in filtered_jobs:
            job_data.append({
                'Job ID': job.job_id[:8] + "...",
                'Type': job.job_type.value if hasattr(job.job_type, 'value') else str(job.job_type),
                'Priority': job.priority.value,
                'Progress': f"{job.progress:.1f}%" if hasattr(job, 'progress') else "N/A",
                'Created': job.created_at.strftime("%H:%M:%S") if job.created_at else "N/A",
                'Started': job.started_at.strftime("%H:%M:%S") if job.started_at else "N/A",
                'Worker': job.worker_id or "N/A",
                'Actions': job.job_id  # Store full ID for actions
            })

        df = pd.DataFrame(job_data)

        # Display table
        st.dataframe(
            df.drop('Actions', axis=1),
            width='stretch',
            column_config={
                "Progress": st.column_config.ProgressColumn(
                    "Progress",
                    help="Job completion progress",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
            }
        )

        # Action buttons for each job
        st.subheader("Job Actions")

        # Group actions by job
        cols = st.columns(min(len(filtered_jobs), 4))
        for i, job in enumerate(filtered_jobs):
            with cols[i % len(cols)]:
                with st.expander(f"Job {job.job_id[:8]}..."):
                    col1, col2 = st.columns(2)

                    with col1:
                        if status in ["pending", "running", "scheduled"]:
                            if st.button("❌ Cancel", key=f"cancel_{status}_{job.job_id}"):
                                from src.compileo.features.gui.utils.job_queue_utils import cancel_job_safe
                                if cancel_job_safe(job.job_id):
                                    st.success("Job cancelled successfully")
                                    st.rerun()
                                else:
                                    st.error("Failed to cancel job")

                        if status == "failed":
                            if st.button("🔄 Restart", key=f"restart_{status}_{job.job_id}"):
                                from src.compileo.features.gui.utils.job_queue_utils import perform_job_operation_safe
                                if perform_job_operation_safe('restart_job', job.job_id):
                                    st.success("Job restarted successfully")
                                    st.rerun()
                                else:
                                    st.error("Failed to restart job")

                    with col2:
                        if st.button("📋 Details", key=f"details_{status}_{job.job_id}"):
                            self.show_job_details(job)

    def show_job_details(self, job):
        """Show detailed information about a job."""
        st.subheader(f"Job Details: {job.job_id}")

        # Basic info
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Type:** {job.job_type}")
            st.write(f"**Priority:** {job.priority.value}")
            st.write(f"**Status:** {job.status.value}")

        with col2:
            st.write(f"**Created:** {job.created_at}")
            st.write(f"**Started:** {job.started_at or 'Not started'}")
            st.write(f"**Completed:** {job.completed_at or 'Not completed'}")

        # Progress and metrics
        if hasattr(job, 'progress'):
            st.progress(job.progress / 100)
            st.write(f"**Progress:** {job.progress:.1f}%")

        if hasattr(job, 'metrics'):
            st.subheader("Performance Metrics")
            metrics_col1, metrics_col2 = st.columns(2)

            with metrics_col1:
                st.write(f"**CPU Usage:** {job.metrics.cpu_usage_percent:.1f}%")
                st.write(f"**Memory Usage:** {job.metrics.memory_usage_mb:.0f} MB")
                st.write(f"**Execution Time:** {job.metrics.execution_time_seconds:.2f}s")

            with metrics_col2:
                st.write(f"**API Calls:** {job.metrics.api_calls_made}")
                st.write(f"**Items Processed:** {job.metrics.items_processed}")
                st.write(f"**Errors:** {job.metrics.errors_count}")

        # Parameters
        if hasattr(job, 'parameters') and job.parameters:
            with st.expander("Job Parameters"):
                st.json(job.parameters)

        # Error information
        if hasattr(job, 'error') and job.error:
            st.error(f"**Error:** {job.error}")

        # Dependencies
        if hasattr(job, 'dependencies') and job.dependencies:
            with st.expander("Dependencies"):
                for dep in job.dependencies:
                    st.write(f"- {dep.job_id} ({dep.dependency_type})")

        # Tags
        if hasattr(job, 'tags') and job.tags:
            st.write(f"**Tags:** {', '.join(job.tags)}")

    def render_job_history_tab(self):
        """Render job history and completed jobs."""
        st.header("Job History")

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            status_filter = st.multiselect(
                "Status",
                [e.value for e in JobStatus],
                default=[JobStatus.COMPLETED.value, JobStatus.FAILED.value]
            )

        with col2:
            type_filter = st.selectbox("Job Type", ["all"] + [e.value for e in JobType])

        with col3:
            days_back = st.selectbox("Time Range", [1, 7, 30, 90], index=1)

        # Get historical jobs (this would need to be implemented)
        # For now, show placeholder
        st.info("Job history feature requires database integration for completed jobs")

        # Sample data for demonstration removed as per investigation
        sample_history = []

        if sample_history:
            df = pd.DataFrame(sample_history)
            st.dataframe(df, width='stretch')

    def render_management_tab(self):
        """Render job management controls."""
        st.header("Job Management Controls")

        # Bulk operations
        st.subheader("Bulk Operations")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🧹 Cleanup Old Jobs", help="Remove jobs older than 30 days"):
                cleaned = cleanup_old_jobs(30)
                st.success(f"Cleaned up {cleaned} old jobs")

        with col2:
            if st.button("🔄 Clear Cache", help="Clear result cache"):
                from src.compileo.features.gui.utils.job_queue_utils import perform_job_operation_safe
                # Access the queue's result_cache through the manager
                manager = perform_job_operation_safe('__getattribute__', 'queue')
                if manager and hasattr(manager, 'result_cache'):
                    manager.result_cache.clear()
                    st.success("Cache cleared successfully")

        with col3:
            if st.button("📊 Health Check", help="Run system health check"):
                health = perform_health_check()
                if health['status'] == 'healthy':
                    st.success("System is healthy")
                else:
                    st.warning(f"System status: {health['status']}")

        # Queue configuration
        st.subheader("Queue Configuration")

        col1, col2 = st.columns(2)

        with col1:
            from src.compileo.features.gui.utils.job_queue_utils import get_job_queue_stats_safe
            stats = get_job_queue_stats_safe()
            st.write("**Queue Type:**", stats.get('queue_type', 'unknown'))
            st.write("**Worker Count:**", "Auto-scaled")  # Would need to implement worker tracking

        with col2:
            st.write("**Cache TTL:**", "1 hour")  # Would need to make configurable
            st.write("**Max Retries:**", "3")  # Would need to make configurable

        # Manual job submission
        st.subheader("Manual Job Submission")

        with st.expander("Submit New Job"):
            col1, col2 = st.columns(2)

            with col1:
                job_type = st.selectbox("Job Type", [e.value for e in JobType])
                priority = st.selectbox("Priority", [e.value for e in JobPriority])

            with col2:
                schedule_time = st.checkbox("Schedule for later")
                scheduled_at = None
                if schedule_time:
                    scheduled_at = st.time_input("Schedule time")

            parameters = st.text_area("Parameters (JSON)", height=100, placeholder='{"key": "value"}')

            if st.button("Submit Job"):
                try:
                    params = json.loads(parameters) if parameters.strip() else {}

                    schedule = None
                    if schedule_time and scheduled_at:
                        # Combine with today's date
                        now = datetime.now()
                        schedule_datetime = datetime.combine(now.date(), scheduled_at)
                        if schedule_datetime < now:
                            schedule_datetime += timedelta(days=1)  # Tomorrow
                        schedule = type('Schedule', (), {'scheduled_time': schedule_datetime})()

                    from src.compileo.features.gui.utils.job_queue_utils import submit_job_safe
                    job_id = submit_job_safe(
                        job_type,
                        params,
                        priority=JobPriority(priority),
                        schedule=schedule
                    )
                    if not job_id:
                        st.error("Failed to submit job - job queue manager not available.")

                    st.success(f"Job submitted successfully: {job_id}")
                    
                    # Monitor the submitted job synchronously
                    monitor_job_synchronously(job_id)

                except json.JSONDecodeError:
                    st.error("Invalid JSON in parameters")
                except Exception as e:
                    st.error(f"Failed to submit job: {e}")

    def render_analytics_tab(self):
        """Render job analytics and reporting."""
        st.header("Job Analytics & Reporting")

        # Time range selector
        time_range = st.selectbox(
            "Time Range",
            ["Last Hour", "Last 24 Hours", "Last 7 Days", "Last 30 Days"],
            index=1
        )

        # Sample analytics data (would need real implementation)
        analytics_data = {
            'job_counts': {
                'extraction': 45,
                'batch_extraction': 12,
                'taxonomy_processing': 8,
                'cleanup': 3
            },
            'success_rates': {
                'extraction': 92.5,
                'batch_extraction': 88.2,
                'taxonomy_processing': 95.1,
                'cleanup': 100.0
            },
            'avg_durations': {
                'extraction': 125.5,  # seconds
                'batch_extraction': 450.2,
                'taxonomy_processing': 89.3,
                'cleanup': 15.8
            }
        }

        # Job type distribution
        st.subheader("Job Type Distribution")

        job_types = list(analytics_data['job_counts'].keys())
        job_counts = list(analytics_data['job_counts'].values())

        fig = px.pie(
            values=job_counts,
            names=job_types,
            title="Jobs by Type"
        )
        st.plotly_chart(fig, width='stretch')

        # Success rates
        st.subheader("Success Rates by Job Type")

        success_df = pd.DataFrame({
            'Job Type': list(analytics_data['success_rates'].keys()),
            'Success Rate (%)': list(analytics_data['success_rates'].values())
        })

        fig = px.bar(
            success_df,
            x='Job Type',
            y='Success Rate (%)',
            title="Success Rates",
            color='Success Rate (%)',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, width='stretch')

        # Performance metrics
        st.subheader("Average Job Duration")

        duration_df = pd.DataFrame({
            'Job Type': list(analytics_data['avg_durations'].keys()),
            'Duration (seconds)': list(analytics_data['avg_durations'].values())
        })

        fig = px.bar(
            duration_df,
            x='Job Type',
            y='Duration (seconds)',
            title="Average Job Duration",
            color='Duration (seconds)',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, width='stretch')

        # System performance over time (placeholder)
        st.subheader("System Performance Trends")

        # Generate sample time series data
        dates = pd.date_range(end=datetime.now(), periods=24, freq='h')
        cpu_usage = [65 + 10 * (i % 3 - 1) + (i % 7) for i in range(24)]
        memory_usage = [512 + 50 * (i % 4 - 2) + (i % 5) * 10 for i in range(24)]
        job_count = [2 + (i % 6) for i in range(24)]

        perf_df = pd.DataFrame({
            'Time': dates,
            'CPU Usage (%)': cpu_usage,
            'Memory Usage (MB)': memory_usage,
            'Active Jobs': job_count
        })

        # Multi-line chart
        fig = px.line(
            perf_df,
            x='Time',
            y=['CPU Usage (%)', 'Memory Usage (MB)', 'Active Jobs'],
            title="System Performance Over Time"
        )
        st.plotly_chart(fig, width='stretch')

        # Export options
        st.subheader("Export Analytics")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📊 Export to CSV"):
                # Would implement CSV export
                st.info("CSV export feature coming soon")

        with col2:
            if st.button("📈 Generate Report"):
                # Would implement PDF/HTML report generation
                st.info("Report generation feature coming soon")


def render_job_management():
    """Main function to render the job management view."""
    view = JobManagementView()
    view.render()


if __name__ == "__main__":
    render_job_management()