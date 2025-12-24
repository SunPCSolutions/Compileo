"""
Job monitoring service module for Compileo GUI.
Handles job status tracking and display for upload and processing operations.
Now uses real-time polling for efficient status updates.
"""

import time
from typing import List, Dict, Any, Optional
import streamlit as st

from ..services.api_client import api_client
from ..state.session_state import session_state


def monitor_job_synchronously(job_id: str, success_text: str = "Processing completed!", timeout: int = 600, poll_interval: int = 2, endpoint_type: str = "general") -> bool:
    """
    Monitors a job using placeholders and blocks the current script run until terminal state.
    Provides accurate status reporting without relying on whole-page polling reruns.
    
    Args:
        job_id: The ID of the job to monitor
        success_text: Text to show upon successful completion
        timeout: Maximum time to wait in seconds
        poll_interval: Interval between status checks in seconds
        endpoint_type: Type of job endpoint ("general" or "dataset")
        
    Returns:
        bool: True if job completed successfully, False otherwise
    """
    # Create placeholders for dynamic updates
    status_placeholder = st.empty()
    details_placeholder = st.empty()
    
    start_time = time.time()
    last_status = None
    last_step = None
    
    # Use a container for better layout within placeholders
    while time.time() - start_time < timeout:
        try:
            # Use specific dataset status endpoint if requested
            if endpoint_type == "dataset":
                 endpoint = f"/api/v1/datasets/generate/{job_id}/status"
            else:
                 endpoint = f"/api/v1/jobs/status/{job_id}"
            
            status_response = api_client.get(endpoint)
            
            if not status_response:
                status_placeholder.warning(f"⚠️ No response from status endpoint for job {job_id}")
                time.sleep(poll_interval)
                continue
                
            status = status_response.get("status", "unknown")
            # Attempt to extract current step from various possible fields
            current_step = (
                status_response.get("current_step") or
                status_response.get("metadata", {}).get("progress_message") or
                status_response.get("metadata", {}).get("current_step") or
                "Processing..."
            )
            
            # Update status display if changed
            if status != last_status or current_step != last_step:
                with status_placeholder.container():
                    if status == "running":
                        st.info(f"🔄 **Status: Running** | {current_step}")
                    elif status == "pending":
                        st.info(f"⏳ **Status: Pending**")
                    elif status == "completed":
                        st.success(f"✅ {success_text}")
                        # Show result if available
                        result = status_response.get("result")
                        if result:
                            with details_placeholder.expander("📊 View Results"):
                                st.json(result)
                        return True
                    elif status == "failed":
                        error_msg = status_response.get("error", "Unknown error")
                        st.error(f"❌ **Job Failed**: {error_msg}")
                        return False
                    elif status == "cancelled":
                        st.warning("🚫 **Job Cancelled**")
                        return False
                    else:
                        st.info(f"❓ **Status: {status.title()}** | {current_step}")
                
                last_status = status
                last_step = current_step

        except Exception as e:
            # Don't fail the monitoring loop on a single transient error
            status_placeholder.warning(f"⚠️ Status check error: {e}")
            
        time.sleep(poll_interval)
        
    status_placeholder.error(f"❌ **Monitoring Timeout**: Job {job_id} did not finish within {timeout} seconds.")
    return False


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

