"""
Selectable Text Display Component

A custom Streamlit component that allows users to select text in a content preview area
with automatic header highlighting and real-time selection feedback.
"""

import os
import streamlit as st
import streamlit.components.v1 as components
from typing import Optional
import re


def _get_component_path():
    """Get the path to the component's frontend files."""
    return os.path.join(os.path.dirname(__file__), "selectable_text_display_frontend")


def _detect_headers(content: str) -> list:
    """
    Detect potential headers in the content.

    Headers are detected as:
    - Lines starting with # (markdown headers)
    - Lines that are ALL CAPS
    - Lines that are mostly uppercase with some punctuation
    """
    headers = []
    lines = content.split('\n')

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Markdown headers
        if line.startswith('#'):
            headers.append(i)
            continue

        # ALL CAPS lines (but not too short)
        if len(line) > 3 and line.isupper():
            headers.append(i)
            continue

        # Lines that are mostly uppercase (80%+ uppercase letters)
        alpha_chars = [c for c in line if c.isalpha()]
        if alpha_chars and len(alpha_chars) > 3:
            uppercase_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if uppercase_ratio > 0.8:
                headers.append(i)

    return headers


def selectable_text_display(
    content: str,
    height: int = 300,
    key: Optional[str] = None
) -> str:
    """
    Display content with selectable text and header highlighting.

    Args:
        content: The text content to display
        height: Height of the display area in pixels
        key: Unique key for the component

    Returns:
        Currently selected text, or empty string if nothing selected
    """
    # Generate unique key if not provided
    if key is None:
        import hashlib
        key = hashlib.md5(content.encode()).hexdigest()[:8]

    # Detect headers for highlighting
    header_lines = _detect_headers(content)

    # Prepare content with line numbers for header detection
    lines = content.split('\n')
    content_with_markers = '\n'.join(
        f"<!-- HEADER -->{line}" if i in header_lines else line
        for i, line in enumerate(lines)
    )

    # Component parameters
    selected_text_key = f"selected_text_area_{key}"
    _selectable_text_display_impl(
        content=content_with_markers,
        height=height,
        key=key,
        selected_text_key=selected_text_key
    )

    # Create a text area for manual text entry (after the preview)
    selected_text = st.text_area(
        "📋 Selected Text (paste your selection here):",
        value="",
        height=100,
        key=selected_text_key,
        help="Paste text you selected from the preview above, or type your own text to add to examples.",
        label_visibility="visible",
        placeholder="Paste selected text here (Ctrl+V), or type your own text..."
    )

    return selected_text or ""


def _selectable_text_display_impl(content: str, height: int, key: str, selected_text_key: str) -> Optional[str]:
    """
    Internal implementation of the selectable text display component.
    """
    try:
        # Check if we're in a Streamlit environment
        if not hasattr(st, 'components'):
            raise ImportError("Streamlit components not available")

        # Process content for header highlighting
        processed_content = content.replace('<!-- HEADER -->', '<span class="header-line">').replace('\n', '</span>\n')

        # Define the HTML template with embedded JavaScript
        html_template = f"""
        <div class="selection-instructions">
            <strong>📝 How to select text:</strong> Highlight any text in the blue box below, then copy (Ctrl+C) and paste into the text area below.
        </div>
        <div id="text-display" class="text-display" tabindex="0">
            {processed_content}
        </div>
        <div id="selected-text" class="selected-text" style="display: none;">
            <strong>Selected Text:</strong><br>
            <span id="selection-content"></span>
        </div>
        <div id="selection-info" class="selection-info"></div>

        <script>
            (function() {{
                const textDisplay = document.getElementById('text-display');
                const selectedTextDiv = document.getElementById('selected-text');
                const selectionContent = document.getElementById('selection-content');
                const selectionInfo = document.getElementById('selection-info');

                let currentSelection = '';

                // Function to update selection display
                function updateSelectionDisplay() {{
                    const selection = window.getSelection();
                    const selectedText = selection.toString().trim();

                    if (selectedText && selectedText !== currentSelection) {{
                        currentSelection = selectedText;
                        selectionContent.textContent = selectedText;
                        selectedTextDiv.style.display = 'block';
                        selectionInfo.textContent = selectedText.length + ' characters selected';

                        // Update the text area with selected text
                        // Try multiple selectors to find the text area
                        let textArea = document.querySelector('textarea[id*="{selected_text_key}"]');
                        if (!textArea) {{
                            textArea = document.querySelector('textarea[data-testid*="{selected_text_key}"]');
                        }}
                        if (!textArea) {{
                            // Try to find any textarea that might be the selected text area
                            const textareas = document.querySelectorAll('textarea');
                            for (let ta of textareas) {{
                                if (ta.id && ta.id.includes('selected_text_area')) {{
                                    textArea = ta;
                                    break;
                                }}
                            }}
                        }}
                        if (textArea) {{
                            textArea.value = selectedText;
                            // Trigger input event to notify Streamlit
                            textArea.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            console.log('Updated text area with selected text:', selectedText);
                        }} else {{
                            console.log('Could not find text area for selected text');
                        }}
                    }} else if (!selectedText) {{
                        selectedTextDiv.style.display = 'none';
                        selectionInfo.textContent = '';
                        currentSelection = '';

                        // Clear the text area
                        let textArea = document.querySelector('textarea[id*="{selected_text_key}"]');
                        if (!textArea) {{
                            textArea = document.querySelector('textarea[data-testid*="{selected_text_key}"]');
                        }}
                        if (!textArea) {{
                            const textareas = document.querySelectorAll('textarea');
                            for (let ta of textareas) {{
                                if (ta.id && ta.id.includes('selected_text_area')) {{
                                    textArea = ta;
                                    break;
                                }}
                            }}
                        }}
                        if (textArea) {{
                            textArea.value = '';
                            textArea.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            console.log('Cleared text area');
                        }} else {{
                            console.log('Could not find text area to clear');
                        }}
                    }}
                }}

                // Listen for selection changes
                document.addEventListener('selectionchange', updateSelectionDisplay);

                // Also listen for mouse up (for better cross-browser support)
                textDisplay.addEventListener('mouseup', () => {{
                    setTimeout(updateSelectionDisplay, 10);
                }});

                // Keyboard support
                textDisplay.addEventListener('keyup', (e) => {{
                    if (e.key === 'Escape') {{
                        window.getSelection().removeAllRanges();
                        updateSelectionDisplay();
                    }}
                }});

                // Handle initial load and process headers
                window.addEventListener('load', () => {{
                    // Ensure header markers are processed
                    const content = textDisplay.innerHTML;
                    textDisplay.innerHTML = content.replace(
                        /<!-- HEADER -->(.*?)(?=\n|$)/g,
                        '<span class="header-line">$1</span>'
                    );
                }});
            }})();
        </script>

        <style>
            .text-display {{
                font-family: 'Courier New', monospace;
                font-size: 14px;
                line-height: 1.4;
                white-space: pre-wrap;
                word-wrap: break-word;
                overflow-y: auto;
                border: 2px solid #007bff;
                border-radius: 4px;
                padding: 12px;
                background-color: #f8f9ff;
                height: {height - 24}px;
                user-select: text;
                cursor: text;
                box-shadow: 0 2px 4px rgba(0,123,255,0.1);
            }}

            .text-display::selection {{
                background-color: #007bff;
                color: white;
            }}

            .selection-instructions {{
                background-color: #e3f2fd;
                border: 1px solid #2196f3;
                border-radius: 4px;
                padding: 8px;
                margin-bottom: 8px;
                font-size: 12px;
                color: #1565c0;
            }}

            .header-line {{
                font-weight: bold;
                color: #2c3e50;
                background-color: #ecf0f1;
                padding: 2px 4px;
                margin: 2px 0;
                border-left: 3px solid #3498db;
                display: block;
            }}

            .selected-text {{
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                padding: 8px;
                margin-top: 8px;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                max-height: 100px;
                overflow-y: auto;
            }}

            .selection-info {{
                font-size: 12px;
                color: #666;
                margin-top: 4px;
            }}
        </style>
        """

        # Use Streamlit components v1
        components.html(
            html_template,
            height=height + 50,  # Extra height for selection display
            scrolling=False
        )

        # Return selected text from session state
        return st.session_state.get(selected_text_key, "")

    except Exception as e:
        # Fallback: return error message and use basic text area
        st.error(f"Custom text display component failed: {str(e)}")
        st.warning("Falling back to basic text area. Text selection features are not available.")

        # Fallback: basic text area (enabled for reading)
        st.text_area(
            "Content Preview (fallback mode):",
            value=content.replace('<!-- HEADER -->', ''),
            height=height,
            key=f"fallback_{key}",
            help="Custom component failed. Using basic text area - selection features not available.",
            disabled=False
        )

        return ""  # No selection in fallback mode


# Export the main function
__all__ = ['selectable_text_display']