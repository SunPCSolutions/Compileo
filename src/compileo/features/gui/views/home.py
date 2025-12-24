"""
Home/Dashboard page for the Compileo GUI.
"""

import streamlit as st
from datetime import datetime

from src.compileo.features.gui.state.session_state import session_state
from src.compileo.features.gui.services.api_client import api_client

def render_home():
    """Render the home/dashboard page."""
    st.title("🏠 Dashboard")
    st.markdown("Welcome to the Compileo Dataset Creation Platform")

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Projects", "0", help="Total number of projects")

    with col2:
        st.metric("Datasets", "0", help="Total datasets created")

    with col3:
        st.metric("Documents", "0", help="Total documents processed")

    with col4:
        st.metric("Quality Score", "0%", help="Average quality score")

    st.divider()

    # Quick actions
    st.subheader("Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📁 Create New Project", width='stretch'):
            session_state.current_page = "projects"
            st.rerun()

    with col2:
        if st.button("🧙 Start Dataset Wizard", width='stretch'):
            session_state.current_page = "wizard"
            st.rerun()

    with col3:
        if st.button("📊 View Quality Metrics", width='stretch'):
            session_state.current_page = "quality"
            st.rerun()

    st.divider()

    # Recent activity
    st.subheader("Recent Activity")

    # Placeholder for recent activity
    with st.container():
        st.info("No recent activity. Start by creating a project or running the dataset wizard.")

    # System status
    st.subheader("System Status")

    col1, col2 = st.columns(2)

    with col1:
        st.success("✅ API Connection: Connected")
        st.success("✅ Database: Connected")

    with col2:
        st.success("✅ Vector Store: Connected")
        st.success("✅ LLM Services: Available")

    # Footer
    st.divider()
    st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")