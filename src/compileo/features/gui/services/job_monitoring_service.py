"""
Job monitoring service module for Compileo GUI.
Handles job status tracking and display for upload and processing operations.
Now uses real-time polling for efficient status updates.
"""

import time
from typing import List, Dict, Any, Optional
import streamlit as st

from ..services.api_client import api_client
from ..services.realtime_job_monitor import display_realtime_job_status, monitor_job_status, stop_monitoring_job
from ..state.session_state import session_state


def wait_for_upload_completion(job_id: str, timeout: int = 60) -> List[int]:
    """Wait for upload job completion and return document IDs."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            status_response = api_client.get_upload_status(job_id)
            if status_response:
                status = status_response.get("status", "unknown")
                if status == "completed":
                    # Extract document IDs from processed files
                    processed_files = status_response.get("processed_files", [])
                    return [doc["id"] for doc in processed_files if "id" in doc]
                elif status == "failed":
                    st.error("Upload job failed")
                    return []

            time.sleep(2)  # Wait 2 seconds before checking again

        except Exception as e:
            st.error(f"Error checking upload status: {e}")
            return []

    st.error("Upload timeout")
    return []


def wait_for_upload_completion_with_paths(job_id: str, timeout: int = 60) -> List[Dict[str, Any]]:
    """Wait for upload job completion and return document info with file paths."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            status_response = api_client.get_upload_status(job_id)
            if status_response:
                status = status_response.get("status", "unknown")
                if status == "completed":
                    # Extract document info from processed files
                    processed_files = status_response.get("processed_files", [])
                    result = []
                    for doc in processed_files:
                        if "id" in doc and "file_name" in doc and "source_file_path" in doc:
                            result.append({
                                "id": doc["id"],
                                "file_name": doc["file_name"],
                                "file_path": doc["source_file_path"]
                            })
                    return result
                elif status == "failed":
                    st.error("Upload job failed")
                    return []

            time.sleep(2)  # Wait 2 seconds before checking again

        except Exception as e:
            st.error(f"Error checking upload status: {e}")
            return []

    st.error("Upload timeout")
    return []


def display_processing_status(job_id: str):
    """Display real-time processing status and results with enhanced UI."""
    try:
        # Use the new real-time monitoring system
        display_realtime_job_status(job_id, show_progress=True, show_steps=True)

        # Check if job is completed to clear session state
        # This is a simple check - the real-time monitor handles the updates
        try:
            status_response = api_client.get(f"/api/v1/jobs/status/{job_id}")
            if status_response:
                status = status_response.get("status", "unknown")
                if status in ["completed", "failed", "cancelled"]:
                    # Clear the job ID from session state
                    session_state.processing_job_id = None
        except Exception:
            # Ignore errors in status check for cleanup
            pass

    except Exception as e:
        st.markdown(f'<div class="status-error">❌ Error setting up real-time monitoring: {e}</div>', unsafe_allow_html=True)
        # Fallback to basic status display
        try:
            status_response = api_client.get(f"/api/v1/jobs/status/{job_id}")
            if status_response:
                _display_basic_status(status_response)
        except Exception as e2:
            st.markdown(f'<div class="status-error">❌ Error checking processing status: {e2}</div>', unsafe_allow_html=True)

def _display_basic_status(status_response: Dict[str, Any]):
    """Fallback basic status display without real-time updates."""
    status = status_response.get("status", "unknown")
    progress = status_response.get("progress", 0)
    current_step = status_response.get("current_step", "Unknown")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">📊 Processing Status</div>', unsafe_allow_html=True)

    # Progress bar with percentage
    st.markdown(f'<div class="progress-label">Progress: {progress}%</div>', unsafe_allow_html=True)
    st.progress(progress / 100)

    if status == "running":
        st.markdown(f'<div class="status-info">ℹ️ Current Step: {current_step}</div>', unsafe_allow_html=True)
    elif status == "completed":
        st.markdown('<div class="status-success">✅ Processing completed successfully!</div>', unsafe_allow_html=True)
        # Display results
        result = status_response.get("result", {})
        if result:
            with st.expander("📋 Processing Results"):
                st.json(result)
    elif status == "failed":
        st.markdown('<div class="status-error">❌ Processing failed</div>', unsafe_allow_html=True)
        error = status_response.get("error")
        if error:
            st.markdown(f'<div class="status-error">❌ Error: {error}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def monitor_job_synchronously(job_id: str, success_text: str = "✅ Job completed successfully!", endpoint_type: str = "general") -> bool:
    """
    Monitor a job synchronously using a local placeholder and loop.
    Prevents disruptive whole-page polling.
    """
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    max_wait = 7200  # 2 hours max
    wait_count = 0
    interval = 2 # Poll every 2 seconds
    
    while wait_count < max_wait:
        try:
            # Route to correct endpoint based on type
            if endpoint_type == "dataset":
                status_response = api_client.get(f"/api/v1/datasets/generate/{job_id}/status")
            else:
                status_response = api_client.get(f"/api/v1/jobs/status/{job_id}")
                
            if status_response:
                status = status_response.get("status", "unknown")
                progress = status_response.get("progress", 0)
                current_step = status_response.get("current_step", "Processing...")
                
                with status_placeholder.container():
                    if status == "running":
                        st.info(f"🔄 **{current_step}**")
                        progress_placeholder.progress(progress / 100)
                    elif status == "pending":
                        st.warning("⏳ **Job Pending...**")
                        progress_placeholder.progress(0.05)
                    elif status == "completed":
                        st.success(success_text)
                        progress_placeholder.empty()
                        return True
                    elif status == "failed":
                        st.error(f"❌ **Job Failed**")
                        error = status_response.get("error")
                        if error:
                            st.error(f"Error Details: {error}")
                        return False
                    elif status == "cancelled":
                        st.warning("🚫 **Job Cancelled**")
                        return False
            
        except Exception as e:
            # Don't fail on transient API errors, just wait
            pass
            
        time.sleep(interval)
        wait_count += interval
        
    st.error("⏰ Job monitoring timed out after 2 hours.")
    return False