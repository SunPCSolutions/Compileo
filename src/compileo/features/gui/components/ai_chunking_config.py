"""
AI Chunking Configuration Component
Provides AI-assisted chunking configuration UI with document analysis and recommendations.
"""

import streamlit as st
from typing import List, Dict, Any, Optional
import json
from ....core.logging import get_logger

logger = get_logger(__name__)

from .ai_chunking_analyzer import (
    analyze_document_for_chunking,
    display_ai_recommendations,
    apply_ai_recommendations
)
from .content_processor import extract_document_pages, smart_sample_content
from ..services.api_client import api_client
from ..services.document_api_service import get_project_documents
def render_ai_assisted_configuration(available_parsed_docs: List[Dict[str, Any]], selected_project_id: int, selected_chunker: str) -> tuple[str, List[str]]:
    """
    Render the AI-assisted chunking configuration UI.

    This complex function handles:
    - Document selection and content loading
    - Content preview with pagination
    - Example collection (copy buttons)
    - Goal description input
    - AI analysis and recommendations
    - Complex state management with session_state

    Args:
        available_parsed_docs: List of available parsed documents
        selected_project_id: ID of the selected project
        selected_chunker: Selected chunking model (gemini, grok, ollama)
    """
    st.markdown("### 🤖 AI-Assisted Chunking Configuration")
    st.markdown("Get intelligent recommendations for optimal chunking strategies based on your document content and goals.")

    # Initialize session state for this component
    if 'ai_chunking_state' not in st.session_state:
        st.session_state.ai_chunking_state = {
            'selected_doc_id': None,
            'available_parsed_files': [],
            'selected_parsed_file': None,
            'document_content': None,
            'current_page': 0,
            'pages': [],
            'goal_description': '',
            'analysis_results': None,
            'show_analysis': False,
            'loading_content': False,
            'analysis_in_progress': False
        }

    # Initialize global examples list that persists across chunks
    if 'ai_chunking_examples' not in st.session_state:
        st.session_state.ai_chunking_examples = []

    state = st.session_state.ai_chunking_state

    # Document & File Selection Section
    st.markdown("#### 📄 Document & File Selection")

    # Single row: Document selection, parsed file selection, and refresh button
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        if available_parsed_docs:
            doc_options = ["Select a document..."] + [f"{doc['file_name']} ({doc['id']})" for doc in available_parsed_docs]
            selected_doc_display = st.selectbox(
                "Choose document:",
                options=doc_options,
                key="ai_chunking_doc_selector",
                help="Select a parsed document to analyze for chunking recommendations"
            )

            if selected_doc_display != "Select a document..." and selected_doc_display:
                selected_doc_id = None
                for doc in available_parsed_docs:
                    if f"{doc['file_name']} ({doc['id']})" == selected_doc_display:
                        selected_doc_id = doc['id']
                        break

                if selected_doc_id != state['selected_doc_id']:
                    state['selected_doc_id'] = selected_doc_id
                    state['available_parsed_files'] = []
                    state['selected_parsed_file'] = None
                    state['document_content'] = None
                    state['current_page'] = 0
                    state['pages'] = []
                    # Keep examples when switching documents
                    state['analysis_results'] = None
                    state['show_analysis'] = False
                    st.rerun()
        else:
            st.warning("⚠️ No parsed documents available. Please parse some documents first.")
            return "", []

    with col2:
        # Load available parsed files if document selected
        if state['selected_doc_id'] and not state['available_parsed_files']:
            with st.spinner("Loading..."):
                try:
                    # Get list of available parsed files from API
                    doc_response = api_client.get(f"/api/v1/documents/{state['selected_doc_id']}/content")
                    if doc_response and 'parsed_files' in doc_response:
                        state['available_parsed_files'] = doc_response['parsed_files']
                    else:
                        state['available_parsed_files'] = []
                except Exception as e:
                    state['available_parsed_files'] = []

        # Parsed file selection
        if state['available_parsed_files']:
            file_options = ["Select a parsed file..."] + state['available_parsed_files']
            selected_file = st.selectbox(
                "Choose parsed file chunk:",
                options=file_options,
                key="parsed_file_selector",
                help="Select a specific parsed file chunk to analyze for chunking recommendations"
            )

            # Handle parsed file selection change
            # Handle parsed file selection change
            if selected_file != "Select a parsed file...":
                if selected_file != state.get('selected_parsed_file'):
                    logger.debug(f"Changing parsed file from {state.get('selected_parsed_file')} to {selected_file}")
                    state['selected_parsed_file'] = selected_file
                    state['document_content'] = None
                    state['current_page'] = 0
                    state['pages'] = []
                    # Keep examples when switching parsed files
                    state['analysis_results'] = None
                    state['show_analysis'] = False
                    st.rerun()
            elif state.get('selected_parsed_file'):  # User selected "Select a parsed file..." option
                logger.debug("Clearing parsed file selection")
                state['selected_parsed_file'] = None
                state['document_content'] = None
                state['current_page'] = 0
                state['pages'] = []
                # Keep examples when clearing selection
                state['analysis_results'] = None
                state['show_analysis'] = False
                st.rerun()
        else:
            # Show disabled selector when no document selected
            st.selectbox(
                "Choose parsed file chunk:",
                options=["Select a document first..."],
                disabled=True,
                key="parsed_file_selector_disabled"
            )

    with col3:
        if st.button("🔄 Refresh", help="Reload available documents"):
            st.cache_data.clear()
            st.rerun()

    # Load content for selected parsed file
    if state['selected_parsed_file'] and not state['document_content']:
        with st.spinner(f"Loading content for {state['selected_parsed_file']}..."):
            try:
                # Get content for specific parsed file
                doc_response = api_client.get(f"/api/v1/documents/{state['selected_doc_id']}/content", params={'parsed_file': state['selected_parsed_file']})
                if doc_response and 'content' in doc_response:
                    content = doc_response['content']
                    state['document_content'] = content
                    state['pages'] = extract_document_pages(content, chunk_size=10000)
                    st.success(f"✅ File loaded successfully ({len(content):,} characters, {len(state['pages'])} pages)")
                else:
                    st.error("❌ Failed to load file content")
            except Exception as e:
                st.error(f"❌ Error loading file content: {str(e)}")

    # Content Preview Section
    if state['document_content'] and state['pages']:
        st.markdown(f"#### 👀 Content Preview - {state['selected_parsed_file']}")

        # Pagination controls
        col_prev, col_page, col_next = st.columns([1, 2, 1])

        with col_prev:
            if st.button("⬅️ Previous", disabled=state['current_page'] == 0):
                state['current_page'] = max(0, state['current_page'] - 1)
                st.rerun()

        with col_page:
            total_pages = len(state['pages'])
            page_options = [f"Page {i+1} of {total_pages}" for i in range(total_pages)]
            selected_page_display = st.selectbox(
                "Navigate pages:",
                options=page_options,
                index=state['current_page'],
                key="page_selector"
            )
            if selected_page_display:
                page_num = int(selected_page_display.split()[1]) - 1
                if page_num != state['current_page']:
                    state['current_page'] = page_num
                    st.rerun()

        with col_next:
            if st.button("Next ➡️", disabled=state['current_page'] >= len(state['pages']) - 1):
                state['current_page'] = min(len(state['pages']) - 1, state['current_page'] + 1)
                st.rerun()

        # Display current page content
        if state['current_page'] < len(state['pages']):
            current_content = state['pages'][state['current_page']]

            # Content display
            st.markdown(f"**Page {state['current_page'] + 1} Content ({state['selected_parsed_file']}):**\n*Read the content below to identify text for examples.*")

            # Simple content display (not disabled so it's clearly readable)
            st.text_area(
                "Content:",
                value=current_content,
                height=300,
                disabled=False,
                key=f"content_display_{state['current_page']}_{state['selected_parsed_file']}"
            )

            # Text input for examples
            st.markdown("### 📝 Add Text Examples")
            st.markdown("Copy text from the content above and paste it here, or type your own examples.")

            # Text input area
            example_text_key = f"example_input_{state['current_page']}_{state['selected_parsed_file']}"
            example_text = st.text_area(
                "Enter example text:",
                value="",
                height=100,
                key=example_text_key,
                placeholder="Paste or type text here to add as an example...",
                help="Enter the text you want to add as an example for AI analysis."
            )

            # Buttons
            col_add, col_copy_page, col_clear = st.columns([1, 1, 1])

            def add_example():
                text_to_add = st.session_state.get(example_text_key, "").strip()
                if text_to_add and text_to_add not in st.session_state.ai_chunking_examples:
                    st.session_state.ai_chunking_examples.append(text_to_add)
                    # Mark that text was added
                    st.session_state[f"text_added_{state['current_page']}_{state['selected_parsed_file']}"] = True
                    # Clear the input after adding
                    st.session_state[example_text_key] = ""
                else:
                    st.session_state[f"text_added_{state['current_page']}_{state['selected_parsed_file']}"] = False

            def copy_page_to_examples():
                if current_content not in st.session_state.ai_chunking_examples:
                    st.session_state.ai_chunking_examples.append(current_content)

            def clear_all_examples():
                st.session_state.ai_chunking_examples = []

            with col_add:
                # Check if text was just added by the callback
                text_was_added = st.session_state.get(f"text_added_{state['current_page']}_{state['selected_parsed_file']}", False)

                if st.button("➕ Add to Examples", key=f"add_example_{state['current_page']}_{state['selected_parsed_file']}", on_click=add_example):
                    if text_was_added:
                        st.success(f"✅ Added to examples!")
                        # Clear the flag
                        st.session_state[f"text_added_{state['current_page']}_{state['selected_parsed_file']}"] = False
                    else:
                        st.warning("⚠️ No text to add. Please enter text above.")

            with col_copy_page:
                if st.button(f"📄 Copy Entire Page", key=f"copy_page_{state['current_page']}_{state['selected_parsed_file']}", on_click=copy_page_to_examples):
                    st.success(f"✅ Entire page added to examples!")
                elif current_content in st.session_state.ai_chunking_examples:
                    st.info("ℹ️ This page is already in your examples.")

            with col_clear:
                if st.button("🗑️ Clear All Examples", key="clear_examples", on_click=clear_all_examples):
                    st.success("✅ All examples cleared!")

    # Examples Collection Section
    if st.session_state.ai_chunking_examples:
        st.markdown("#### 📚 Collected Examples")
        st.markdown(f"**{len(st.session_state.ai_chunking_examples)} example(s) collected for AI analysis:**")

        for i, example in enumerate(st.session_state.ai_chunking_examples):
            with st.expander(f"Example {i+1} ({len(example)} chars)"):
                st.text_area(
                    f"Example {i+1}:",
                    value=example,
                    height=150,
                    key=f"example_{i}",
                    disabled=True
                )
                if st.button(f"Remove Example {i+1}", key=f"remove_example_{i}"):
                    st.session_state.ai_chunking_examples.pop(i)
                    st.success(f"✅ Example {i+1} removed!")
                    st.rerun()

    # Goal Description Input
    st.markdown("#### 🎯 Chunking Goals")
    state['goal_description'] = st.text_area(
        "Describe your chunking goals:",
        value=state['goal_description'],
        height=100,
        placeholder="Describe what you want to achieve with chunking. For example:\n- Maintain semantic coherence within chunks\n- Keep related concepts together\n- Break at natural section boundaries\n- Optimize for question-answering tasks",
        help="Provide detailed instructions about how you want the document to be chunked. Be specific about what constitutes meaningful chunks for your use case."
    )

    # AI Analysis Section
    if state['document_content'] and state['goal_description'].strip():
        st.markdown("#### 🧠 AI Analysis & Recommendations")

        # Prepare analysis input
        sample_content = smart_sample_content(state['document_content'], st.session_state.ai_chunking_examples)
        user_instructions = state['goal_description']

        # Add examples to instructions if available
        if st.session_state.ai_chunking_examples:
            examples_text = "\n\n=== USER-PROVIDED EXAMPLES ===\n" + "\n---\n".join(st.session_state.ai_chunking_examples)
            user_instructions += examples_text

        # Analysis button
        if st.button("🔍 Analyze & Recommend Chunking Strategy", type="primary"):
            state['analysis_in_progress'] = True
            state['show_analysis'] = False
            st.rerun()

        # Perform analysis
        if state['analysis_in_progress'] and not state['show_analysis']:
            with st.spinner("🤖 AI is analyzing your document and requirements..."):
                recommendations = analyze_document_for_chunking(
                    sample_content=sample_content,
                    user_instructions=user_instructions,
                    selected_chunker=selected_chunker
                )

                if recommendations:
                    state['analysis_results'] = recommendations
                    state['show_analysis'] = True
                    st.success("✅ Analysis complete! Review recommendations below.")
                else:
                    st.error("❌ Analysis failed. Please check your API configuration and try again.")

            state['analysis_in_progress'] = False

        # Display results
        if state['show_analysis'] and state['analysis_results']:
            display_ai_recommendations(state['analysis_results'])

    elif not state['document_content']:
        st.info("ℹ️ Please select a document to begin AI-assisted configuration.")

    elif not state['goal_description'].strip():
        st.info("ℹ️ Please describe your chunking goals to enable AI analysis.")

    # Reset button
    st.markdown("---")
    if st.button("🔄 Reset Configuration", help="Clear all selections and start over"):
        state.clear()
        st.session_state.ai_chunking_state = {
            'selected_doc_id': None,
            'available_parsed_files': [],
            'selected_parsed_file': None,
            'document_content': None,
            'current_page': 0,
            'pages': [],
            'goal_description': '',
            'analysis_results': None,
            'show_analysis': False,
            'loading_content': False,
            'analysis_in_progress': False
        }
        st.success("✅ Configuration reset!")
        st.rerun()

    # Return current state for calling code
    return state['goal_description'], st.session_state.ai_chunking_examples
from ..state.session_state import session_state