"""
Real-time job monitoring service for Compileo GUI.
Provides event-driven job status polling using long polling for efficient real-time updates.
"""

import threading
import time
import logging
from typing import Dict, Any, Optional, Callable, List
import streamlit as st

from .api_client import api_client

logger = logging.getLogger(__name__)

class RealtimeJobMonitor:
    """Real-time job status monitoring with long polling."""

    def __init__(self):
        self._monitored_jobs: Dict[str, Dict[str, Any]] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start_monitoring(self, job_id: str, callback: Optional[Callable] = None, poll_interval: int = 2):
        """
        Start monitoring a job for real-time status updates.

        Args:
            job_id: The job ID to monitor
            callback: Optional callback function to call when status changes
            poll_interval: Interval between polls in seconds
        """
        if job_id in self._monitored_jobs:
            logger.warning(f"Job {job_id} is already being monitored")
            return

        self._monitored_jobs[job_id] = {
            'last_status': None,
            'last_progress': 0.0,
            'last_updated': None,
            'poll_interval': poll_interval,
            'callback': callback
        }

        if callback:
            if job_id not in self._callbacks:
                self._callbacks[job_id] = []
            self._callbacks[job_id].append(callback)

        logger.info(f"Started monitoring job {job_id}")

        # Start monitoring thread if not already running
        if not self._running:
            self._start_monitoring_thread()

    def stop_monitoring(self, job_id: str):
        """Stop monitoring a specific job."""
        if job_id in self._monitored_jobs:
            del self._monitored_jobs[job_id]
            if job_id in self._callbacks:
                del self._callbacks[job_id]
            logger.info(f"Stopped monitoring job {job_id}")

    def stop_all_monitoring(self):
        """Stop monitoring all jobs."""
        self._monitored_jobs.clear()
        self._callbacks.clear()
        self._stop_event.set()
        logger.info("Stopped monitoring all jobs")

    def is_monitoring(self, job_id: str) -> bool:
        """Check if a job is being monitored."""
        return job_id in self._monitored_jobs

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the last known status of a monitored job."""
        if job_id not in self._monitored_jobs:
            return None
        return self._monitored_jobs[job_id]

    def _start_monitoring_thread(self):
        """Start the background monitoring thread."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_jobs, daemon=True)
        self._monitor_thread.start()
        logger.info("Started real-time job monitoring thread")

    def _monitor_jobs(self):
        """Main monitoring loop that polls for job status updates."""
        while not self._stop_event.is_set():
            try:
                jobs_to_remove = []

                for job_id, job_info in list(self._monitored_jobs.items()):
                    try:
                        # Use long polling endpoint for efficient status checking
                        status_response = self._poll_job_status(job_id, job_info)

                        if status_response:
                            self._handle_status_update(job_id, status_response, job_info)

                            # Check if job is completed/failed/cancelled
                            status = status_response.get('status', '')
                            if status in ['completed', 'failed', 'cancelled']:
                                jobs_to_remove.append(job_id)

                    except Exception as e:
                        logger.error(f"Error monitoring job {job_id}: {e}")
                        # Continue monitoring other jobs

                # Remove completed jobs
                for job_id in jobs_to_remove:
                    self.stop_monitoring(job_id)

                # Sleep before next polling cycle
                time.sleep(0.5)  # Small delay between job checks

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Wait longer on error

        self._running = False
        logger.info("Real-time job monitoring thread stopped")

    def _poll_job_status(self, job_id: str, job_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Poll job status using long polling endpoint."""
        try:
            # Use the long polling endpoint with current known status
            current_status = job_info.get('last_status')
            timeout = min(job_info.get('poll_interval', 2), 30)  # Max 30 seconds

            params = {'timeout': timeout}
            if current_status:
                params['current_status'] = current_status

            response = api_client.get(f"/api/v1/jobs/status/{job_id}/poll", params=params)
            return response

        except Exception as e:
            logger.error(f"Failed to poll job {job_id} status: {e}")
            # Fallback to regular status endpoint
            try:
                return api_client.get(f"/api/v1/jobs/status/{job_id}")
            except Exception as e2:
                logger.error(f"Fallback status check also failed for job {job_id}: {e2}")
                return None

    def _handle_status_update(self, job_id: str, status_response: Dict[str, Any], job_info: Dict[str, Any]):
        """Handle job status update."""
        current_status = status_response.get('status')
        current_progress = status_response.get('progress', 0.0)
        current_updated = status_response.get('updated_at')

        # Check if status actually changed
        last_status = job_info.get('last_status')
        last_progress = job_info.get('last_progress', 0.0)
        last_updated = job_info.get('last_updated')

        status_changed = current_status != last_status
        progress_changed = abs(current_progress - last_progress) > 0.01  # Small threshold
        updated_changed = current_updated != last_updated

        if status_changed or progress_changed or updated_changed:
            # Update stored status
            job_info['last_status'] = current_status
            job_info['last_progress'] = current_progress
            job_info['last_updated'] = current_updated

            # Call callbacks
            self._call_callbacks(job_id, status_response)

            logger.debug(f"Job {job_id} status update: {current_status} ({current_progress:.1f}%)")

    def _call_callbacks(self, job_id: str, status_response: Dict[str, Any]):
        """Call registered callbacks for job status updates."""
        if job_id in self._callbacks:
            for callback in self._callbacks[job_id]:
                try:
                    callback(job_id, status_response)
                except Exception as e:
                    logger.error(f"Error in callback for job {job_id}: {e}")

    def add_callback(self, job_id: str, callback: Callable):
        """Add a callback function for job status updates."""
        if job_id not in self._callbacks:
            self._callbacks[job_id] = []
        self._callbacks[job_id].append(callback)

    def remove_callback(self, job_id: str, callback: Callable):
        """Remove a callback function."""
        if job_id in self._callbacks:
            try:
                self._callbacks[job_id].remove(callback)
                if not self._callbacks[job_id]:
                    del self._callbacks[job_id]
            except ValueError:
                pass  # Callback not found

# Global instance
realtime_job_monitor = RealtimeJobMonitor()

def monitor_job_status(job_id: str, callback: Optional[Callable] = None, poll_interval: int = 2):
    """
    Convenience function to start monitoring a job.

    Args:
        job_id: The job ID to monitor
        callback: Optional callback function(status_response) called on updates
        poll_interval: Polling interval in seconds
    """
    realtime_job_monitor.start_monitoring(job_id, callback, poll_interval)

def stop_monitoring_job(job_id: str):
    """Stop monitoring a specific job."""
    realtime_job_monitor.stop_monitoring(job_id)

def get_monitored_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Get the last known status of a monitored job."""
    return realtime_job_monitor.get_job_status(job_id)

def display_realtime_job_status(job_id: str, show_progress: bool = True, show_steps: bool = True):
    """
    Display real-time job status in Streamlit with automatic updates.

    Args:
        job_id: The job ID to display
        show_progress: Whether to show progress bar
        show_steps: Whether to show current step information
    """
    # Create a placeholder for the status display
    status_placeholder = st.empty()

    def update_display(job_id_inner: str, status_response: Dict[str, Any]):
        """Callback to update the display when status changes."""
        with status_placeholder.container():
            _render_job_status(status_response, show_progress, show_steps)

    # Start monitoring if not already
    if not realtime_job_monitor.is_monitoring(job_id):
        monitor_job_status(job_id, update_display)

    # Get current status and display immediately
    current_status = get_monitored_job_status(job_id)
    if current_status:
        with status_placeholder.container():
            _render_job_status(current_status, show_progress, show_steps)
    else:
        # Fallback to API call
        try:
            status_response = api_client.get(f"/api/v1/jobs/status/{job_id}")
            if status_response:
                with status_placeholder.container():
                    _render_job_status(status_response, show_progress, show_steps)
                # Start monitoring
                monitor_job_status(job_id, update_display)
        except Exception as e:
            st.error(f"Failed to get job status: {e}")

def _render_job_status(status_response: Dict[str, Any], show_progress: bool, show_steps: bool):
    """Render job status display."""
    status = status_response.get('status', 'unknown')
    progress = status_response.get('progress', 0)
    current_step = status_response.get('current_step', '')

    # Status indicator
    if status == 'running':
        st.markdown("🔄 **Job Running**")
    elif status == 'completed':
        st.markdown("✅ **Job Completed**")
    elif status == 'failed':
        st.markdown("❌ **Job Failed**")
    elif status == 'pending':
        st.markdown("⏳ **Job Pending**")
    elif status == 'cancelled':
        st.markdown("🚫 **Job Cancelled**")
    else:
        st.markdown(f"❓ **Job {status.title()}**")

    # Progress bar
    if show_progress and status == 'running':
        st.progress(progress / 100)
        st.text(f"Progress: {progress:.1f}%")

    # Current step
    if show_steps and current_step and status == 'running':
        st.text(f"Current Step: {current_step}")

    # Error display
    error = status_response.get('error')
    if error and status == 'failed':
        st.error(f"Error: {error}")

    # Result display
    result = status_response.get('result')
    if result and status == 'completed':
        with st.expander("📋 Job Result"):
            st.json(result)