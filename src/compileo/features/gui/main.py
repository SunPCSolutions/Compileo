"""
Main Streamlit application entry point for Compileo GUI.
"""

import sys
import os

# Ensure the project root is in the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import requests

from src.compileo.features.gui.config import config
from src.compileo.features.gui.state.session_state import session_state
from src.compileo.features.gui.state.wizard_state import wizard_state
from src.compileo.features.gui.services.api_client import api_client
from src.compileo.features.gui.utils.settings_storage import settings_storage
from src.compileo.features.gui.views.home import render_home
from src.compileo.features.gui.views.projects import render_projects
from src.compileo.features.gui.views.wizard import render_wizard
from src.compileo.features.gui.views.document_processing import render_document_processing
from src.compileo.features.gui.views.chunk_management import render_chunk_management
from src.compileo.features.gui.views.core_dataset_generation import render_core_dataset_generation
from src.compileo.features.gui.views.taxonomy import render_taxonomy
from src.compileo.features.gui.views.extraction_unified import render_extraction_unified
from src.compileo.features.gui.views.quality import render_quality
from src.compileo.features.gui.views.benchmarking import render_benchmarking
from src.compileo.features.gui.views.settings import render_settings
from src.compileo.features.jobhandle.job_management import render_job_management
from src.compileo.features.gui.components.job_queue_sidebar import render_job_queue_sidebar

def setup_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title=config.app_title,
        page_icon=config.app_icon,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items=None  # Disable menu items to prevent connection errors
    )

    # Enhanced CSS for modern look and better styling
    st.markdown("""
    <style>
    /* Modern color palette */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --success-color: #10b981;
        --error-color: #ef4444;
        --warning-color: #f59e0b;
        --background-light: #f8fafc;
        --card-background: #ffffff;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --border-color: #e2e8f0;
    }

    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        text-align: center;
    }

    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 2rem;
        background: var(--card-background);
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 2rem;
    }

    .logo-section {
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .logo-text {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--primary-color);
    }

    .settings-button {
        background: var(--primary-color);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        cursor: pointer;
        font-size: 0.9rem;
        transition: all 0.2s ease;
    }

    .settings-button:hover {
        background: var(--secondary-color);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }

    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--primary-color);
    }

    /* Card styling */
    .card {
        background: var(--card-background);
        border: 1px solid var(--border-color);
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        transition: box-shadow 0.2s ease;
    }

    .card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    .card-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Status styling with icons */
    .status-success {
        color: var(--success-color);
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .status-error {
        color: var(--error-color);
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .status-warning {
        color: var(--warning-color);
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .status-info {
        color: var(--primary-color);
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Button enhancements */
    .stButton button {
        border-radius: 0.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    /* Sidebar styling - High specificity to override Streamlit defaults */
    [data-testid="stSidebar"] .sidebar-section {
        margin-bottom: 0.75rem !important;
    }

    /* Reduce spacing for bottom sections */
    [data-testid="stSidebar"] .sidebar-section:last-child {
        margin-bottom: 0.25rem !important;
    }

    [data-testid="stSidebar"] .sidebar-section-header {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        margin-bottom: 0.25rem !important;
    }

    [data-testid="stSidebar"] .sidebar-divider {
        border-top: 1px solid var(--border-color) !important;
        margin: 0.25rem 0 !important;
    }

    /* Tighter spacing for settings section */
    [data-testid="stSidebar"] .settings-section {
        margin-bottom: 0.25rem !important;
    }

    /* Reduce button spacing within sections - very specific selectors */
    [data-testid="stSidebar"] .sidebar-section .stButton {
        margin-bottom: 0.125rem !important;
    }

    [data-testid="stSidebar"] .sidebar-section .stButton:last-child {
        margin-bottom: 0 !important;
    }

    /* Override Streamlit button defaults in sidebar */
    [data-testid="stSidebar"] .stButton > button {
        margin-bottom: 0.125rem !important;
    }

    [data-testid="stSidebar"] .stButton:last-child > button {
        margin-bottom: 0 !important;
    }

    .nav-button {
        background: none;
        border: none;
        width: 100%;
        text-align: left;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .nav-button:hover {
        background: var(--background-light);
    }

    .nav-button.active {
        background: var(--primary-color);
        color: white;
    }

    /* Progress bars */
    .progress-container {
        margin: 1rem 0;
    }

    .progress-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
    }

    /* Form styling */
    .stSelectbox, .stTextInput, .stNumberInput, .stSlider {
        margin-bottom: 1rem;
    }

    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: var(--text-primary);
        color: white;
        text-align: center;
        border-radius: 0.375rem;
        padding: 0.5rem;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.8rem;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .header-container {
            padding: 1rem;
            flex-direction: column;
            gap: 1rem;
        }

        .main-header {
            font-size: 2rem;
        }

        .card {
            padding: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def shutdown_compileo():
    """Shutdown the Compileo API server and all workers."""
    try:
        # Get API URL from settings
        saved_settings = settings_storage.load_settings()
        api_url = saved_settings.get("api_base_url", config.api_base_url).rstrip('/')

        # Call shutdown endpoint
        response = requests.post(f"{api_url}/shutdown", timeout=30)

        if response.status_code == 200:
            st.success("✅ Shutdown initiated successfully!")
            st.info("The API server and workers will shutdown shortly.")
            st.info("You may need to restart the services manually.")
        else:
            st.error(f"❌ Shutdown failed: HTTP {response.status_code}")
            st.text(response.text[:500])  # Show first 500 chars of error

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Could not connect to API server: {e}")
        st.info("Make sure the Compileo API server is running and accessible.")
    except Exception as e:
        st.error(f"❌ Unexpected error during shutdown: {e}")


def render_header():
    """Render the top header with logo."""
    st.markdown("""
    <div class="header-container">
        <div class="logo-section">
            <div style="font-size: 2rem;">🔬</div>
            <div class="logo-text">Compileo</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render the sidebar with grouped navigation sections."""
    with st.sidebar:
        # Compileo title above Wizard section
        st.markdown('<div style="text-align: center; font-size: 1.5rem; font-weight: 700; color: var(--primary-color); margin-bottom: 1rem;">Compileo</div>', unsafe_allow_html=True)

        # Wizard section (moved to top)
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-header">🧙 Wizard</div>', unsafe_allow_html=True)
        if st.button("Dataset Generation Wizard", key="nav_wizard",
                     type="secondary" if session_state.current_page != "wizard" else "primary"):
            session_state.current_page = "wizard"
            st.query_params["page"] = "wizard"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Divider
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # Workflow section (consolidated)
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-header">⚙️ Workflow</div>', unsafe_allow_html=True)

        # Home as first workflow item
        if st.button("🏠 Home", key="nav_home",
                     type="secondary" if session_state.current_page != "home" else "primary"):
            session_state.current_page = "home"
            st.query_params["page"] = "home"
            st.rerun()

        workflow_pages = [
            ("📁 Projects", "projects"),
            ("📄 Document Processing", "document_processing"),
            ("🏷️ Taxonomy", "taxonomy"),
            ("🔍 Extraction", "extraction_unified"),
            ("📊 Dataset Generation", "core_dataset_generation")
        ]

        for display_name, page_key in workflow_pages:
            if st.button(display_name, key=f"nav_{page_key}",
                         type="secondary" if session_state.current_page != page_key else "primary"):
                session_state.current_page = page_key
                st.query_params["page"] = page_key
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Divider
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # Analysis section
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-header">📊 Analysis</div>', unsafe_allow_html=True)

        analysis_pages = [
            ("📊 Quality Metrics", "quality"),
            ("📈 Benchmarking", "benchmarking")
        ]

        for display_name, page_key in analysis_pages:
            if st.button(display_name, key=f"nav_{page_key}",
                           type="secondary" if session_state.current_page != page_key else "primary"):
                session_state.current_page = page_key
                st.query_params["page"] = page_key
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Divider
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # Job Queue section
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-header">🔄 Job Queue</div>', unsafe_allow_html=True)
        if st.button("📋 Job Management", key="nav_job_management",
                       type="secondary" if session_state.current_page != "job_management" else "primary"):
            session_state.current_page = "job_management"
            st.query_params["page"] = "job_management"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Divider
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # Settings and System Control (combined section for tighter spacing)
        st.markdown('<div class="sidebar-section settings-section">', unsafe_allow_html=True)
        if st.button("⚙️ Settings", key="nav_settings",
                       type="secondary" if session_state.current_page != "settings" else "primary"):
            session_state.current_page = "settings"
            st.query_params["page"] = "settings"
            st.rerun()

        # Shutdown button with consistent spacing
        st.markdown('<div style="margin: 0.125rem 0 0 0 !important;">', unsafe_allow_html=True)
        if st.button("🛑 Shutdown", key="shutdown_button", type="secondary",
                     help="Gracefully shutdown the Compileo API server and all workers"):
            shutdown_compileo()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Current project info
        if session_state.current_project:
            st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-section-header">Current Project</div>', unsafe_allow_html=True)
            st.markdown(f"**{session_state.current_project['name']}**")
            st.markdown(f"ID: {session_state.current_project['id']}")
            if st.button("Clear Project", key="clear_project"):
                session_state.current_project = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # Notifications
        notifications = session_state.notifications
        if notifications:
            st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-section-header">Notifications</div>', unsafe_allow_html=True)
            for i, notification in enumerate(notifications[-3:]):  # Show last 3
                icon = "❌" if notification['type'] == 'error' else "⚠️" if notification['type'] == 'warning' else "ℹ️"
                st.markdown(f"{icon} {notification['message']}")

            if st.button("Clear Notifications", key="clear_notifications"):
                session_state.clear_notifications()
            st.markdown('</div>', unsafe_allow_html=True)

        # Job Queue Sidebar
        render_job_queue_sidebar()

    return session_state.current_page

def main():
    """Main application entry point."""
    setup_page()

    # Initialize job queue manager for GUI using safe utilities
    from src.compileo.features.gui.utils.job_queue_utils import initialize_job_queue_manager_safe

    if not initialize_job_queue_manager_safe(
        global_max_jobs=10,
        auto_start_worker=False  # Don't auto-start worker - let dedicated process handle this
    ):
        st.error("Failed to initialize job queue manager. Some features may be unavailable.")

    # Initialize session state
    # Load saved API key into session state if not already set
    if not session_state.api_key:
        saved_settings = settings_storage.load_settings()
        saved_api_key = saved_settings.get("api_key")
        if saved_api_key:
            session_state.api_key = saved_api_key
            # Also update the API client with the loaded key
            api_client.set_api_key(saved_api_key)

    # Apply API base URL: prefer environment variable over saved settings
    import os
    env_api_url = os.getenv("API_BASE_URL")
    if env_api_url:
        # Environment variable takes precedence
        api_client.update_settings(base_url=env_api_url, api_key=api_client.api_key or "")
    else:
        # Fall back to saved settings
        saved_settings = settings_storage.load_settings()
        saved_api_url = saved_settings.get("api_base_url")
        if saved_api_url:
            api_client.update_settings(base_url=saved_api_url, api_key=api_client.api_key or "")

    wizard_state.initialize()  # Ensure wizard state is initialized

    # Define valid pages
    valid_pages = {
        "home", "projects", "wizard", "document_processing", "chunk_management", "taxonomy",
        "extraction_unified", "core_dataset_generation", "quality", "benchmarking",
        "job_management", "settings"
    }

    # Get current page from query parameters - this takes precedence on page load
    query_params = st.query_params

    # Get page parameter - handle different possible return types from st.query_params
    if "page" in query_params:
        page_param = query_params["page"]
        if isinstance(page_param, list) and page_param:
            url_page = page_param[0]
        elif isinstance(page_param, str):
            url_page = page_param
        else:
            url_page = str(page_param) if page_param else None
    else:
        url_page = None

    # Determine the current page: URL param takes precedence, then session state, then default
    if url_page and url_page in valid_pages:
        current_page = url_page
        session_state.current_page = current_page  # Sync session state
    elif 'current_page' in st.session_state and st.session_state['current_page'] in valid_pages:
        current_page = st.session_state['current_page']
    else:
        current_page = "home"
        session_state.current_page = current_page

    # Render sidebar (buttons handle URL updates directly)
    selected_page = render_sidebar()

    # Render main content based on current page
    try:
        if current_page == "home":
            render_home()
        elif current_page == "projects":
            render_projects()
        elif current_page == "wizard":
            render_wizard()
        elif current_page == "document_processing":
            render_document_processing()
        elif current_page == "chunk_management":
            render_chunk_management()
        elif current_page == "core_dataset_generation":
            render_core_dataset_generation()
        elif current_page == "taxonomy":
            render_taxonomy()
        elif current_page == "extraction_unified":
            render_extraction_unified()
        elif current_page == "quality":
            render_quality()
        elif current_page == "benchmarking":
            render_benchmarking()
        elif current_page == "job_management":
            render_job_management()
        elif current_page == "settings":
            render_settings()
        else:
            st.error(f"Unknown page: {current_page}")
            # Reset to home if unknown
            session_state.current_page = "home"
            st.rerun()
    except Exception as e:
        st.error(f"Error rendering page {current_page}: {e}")
        st.code(str(e))
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()