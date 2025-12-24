import fitz  # PyMuPDF
import requests
import json
import os
import re
from typing import Optional, List, Tuple
from PIL import Image
import base64
import io
from .image_utils import convert_pdf_to_optimized_images
from ....core.logging import get_logger

logger = get_logger(__name__)

def _clean_response(response: str) -> str:
    """
    Clean unwanted commentary and analysis from responses.
    """
    # For now, return raw response to avoid processing issues
    return response.strip()

def _normalize_markdown_spacing(content: str) -> str:
    """
    Aggressively normalize spacing in markdown for consistent downstream processing.
    """
    import re

    # Remove excessive newlines (more than 3)
    content = re.sub(r'\n{4,}', '\n\n\n', content)

    # Normalize paragraph breaks to single newline
    # But preserve double newlines for sections
    lines = content.split('\n')
    normalized_lines = []
    prev_empty = False

    for line in lines:
        line = line.rstrip()  # Remove trailing spaces
        is_empty = len(line.strip()) == 0
        is_heading = line.strip().startswith('#')
        is_list = line.strip().startswith(('- ', '* ', '1. ', '    - '))

        if is_empty:
            if not prev_empty or is_heading:
                normalized_lines.append('')
            prev_empty = True
        else:
            # Normalize horizontal whitespace (except in tables and code)
            if not ('|' in line and line.count('|') > 2):  # Not a table row
                line = re.sub(r' {2,}', ' ', line)  # Multiple spaces to single
            normalized_lines.append(line)
            prev_empty = False

    # Join and clean up
    content = '\n'.join(normalized_lines)

    # Ensure section breaks are double newlines
    content = re.sub(r'(?<!\n)\n(?=#)', '\n\n', content)

    # Ensure list indentation is consistent (4 spaces)
    content = re.sub(r'^(\s*)-', '    -', content, flags=re.MULTILINE)
    content = re.sub(r'^(\s*)\*', '    *', content, flags=re.MULTILINE)

    return content.strip()

def query_grok_vlm(prompt, image=None, images: Optional[List[Image.Image]] = None, model: Optional[str] = None):
    """
    Sends a prompt and image(s) to the Grok VLM and returns the response.
    """
    from ....core.settings import backend_settings
    api_key = backend_settings.get_setting("grok_api_key")
    if not api_key:
        raise ValueError("Grok API key not configured in settings.")

    api_url = "https://api.x.ai/v1/chat/completions"

    content_blocks = []
    
    # Handle single image or list of images
    img_list = images if images else ([image] if image else [])
    
    for img in img_list:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        content_blocks.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_base64}",
                "detail": "high"
            }
        })
    
    # Add prompt text
    content_blocks.append({"type": "text", "text": prompt})

    # Use vision-capable model
    if model is None:
        model = "grok-4"  # Vision-capable model

    request_data = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": content_blocks
        }],
        "max_tokens": 32000,
        "temperature": 0.1
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(api_url, json=request_data, headers=headers)
        response.raise_for_status()

        response_json = response.json()
        markdown_content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Clean the response
        if markdown_content:
            # Remove common prompt artifacts
            prompt_artifacts = [
                "You are an expert document structure analyst and markdown conversion specialist",
                "CHAIN OF THOUGHT PROCESSING:",
                "CRITICAL STRUCTURAL REQUIREMENTS:",
                "CONTENT PRESERVATION RULES:",
                "OUTPUT QUALITY ASSURANCE:",
                "<document>",
                "</document>",
                "Convert the text within <document> tags",
                "Return only the markdown content",
                "Convert the above document to clean, structured markdown"
            ]

            for artifact in prompt_artifacts:
                if artifact in markdown_content:
                    parts = markdown_content.split(artifact, 1)
                    if len(parts) > 1:
                        markdown_content = parts[1].strip()

            # Also clean up any leading/trailing instruction text
            lines = markdown_content.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if not any(instruction in line.upper() for instruction in [
                    'PRESERVE', 'CONVERT', 'RETURN ONLY', 'EXTRACTED TEXT', 'CRITICAL', 'IMPORTANT:',
                    'CONTENT TO CONVERT', 'OUTPUT INSTRUCTION', 'YOU ARE', '<DOCUMENT>', '</DOCUMENT>',
                    'CHAIN OF THOUGHT', 'STRUCTURAL REQUIREMENTS', 'QUALITY ASSURANCE', 'ANALYSIS:',
                    'PROCESSING:', 'ASSURANCE:', 'HIERARCHY', 'CONSISTENT'
                ]):
                    cleaned_lines.append(line)

            markdown_content = '\n'.join(cleaned_lines).strip()

        if not markdown_content:
            error_msg = f"Error: Grok API returned empty response for image"
            logger.error(error_msg)
            return error_msg

        return markdown_content

    except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
        error_msg = f"Error: Grok API call failed: {str(e)}"
        logger.error(error_msg)
        return error_msg

def analyze_document_structure(pdf_path: str, api_key: Optional[str] = None, model: Optional[str] = None) -> str:
    """
    Analyzes the document structure to create a visual style guide for heading levels.
    Used for the first pass of the Two-Pass VLM Strategy.
    """
    if api_key is None:
        api_key = os.environ.get("GROK_API_KEY")
        if not api_key:
            raise ValueError("GROK_API_KEY environment variable not set or api_key parameter not provided.")

    try:
        # Convert PDF to images (Use DPI=300 as requested)
        images = convert_pdf_to_optimized_images(pdf_path, dpi=300, max_dimension=1024)
        
    except Exception as e:
        logger.error(f"Structure analysis failed: {e}")
        return ""

    if images:
        prompt = """Analyze the visual layout of this entire document (all provided pages) to create a Style Guide for markdown conversion.
Identify the specific visual cues used for the hierarchy across the whole document:

1. Main Title (if present): Describe font size, weight (bold?), alignment.
2. Level 1 Headings (H1): Describe visual appearance (size relative to body, bold?).
3. Level 2 Headings (H2): Describe visual appearance.
4. Body Text: Describe standard appearance.

Output a comprehensive 'Style Guide' describing these rules to help a future AI model correctly identify headings consistently across all pages.
Do not extract content."""

        try:
            logger.debug(f"Analyzing document structure ({len(images)} pages)...")
            # Send all images for analysis
            response = query_grok_vlm(prompt, images=images, model=model)
            if response and "Error" not in response:
                return response
        except Exception as e:
            logger.error(f"Error during structure analysis: {e}")
            
    return ""

def parse_pdf_with_grok(
    pdf_path: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    overlap_text: Optional[str] = None,
    style_guide: Optional[str] = None
) -> str:
    """
    Parses a PDF file by converting to images and using Grok vision API to convert to clean markdown.
    Handles large files by chunking them into manageable pieces.
    Includes overlap_text for context when processing split files.
    """
    if api_key is None:
        api_key = os.environ.get("GROK_API_KEY")
        if not api_key:
            raise ValueError("GROK_API_KEY environment variable not set or api_key parameter not provided.")

    # Convert PDF to optimized images
    try:
        images = convert_pdf_to_optimized_images(pdf_path, dpi=300, max_dimension=1024)
    except Exception as e:
        raise ValueError(f"Failed to process PDF file: {e}")

    full_markdown = ""

    for i, image in enumerate(images):
        logger.debug(f"Processing page {i+1}/{len(images)} with Grok...")
        
        # Inject style guide if available
        style_instruction = ""
        if style_guide:
            style_instruction = f"""
STYLE GUIDE CONTEXT:
The document uses the following visual hierarchy:
{style_guide}

INSTRUCTION: Use these specific visual cues to determine Heading Levels (#, ##, ###).
"""

        # Simplified comprehensive document processing prompt
        prompt = f"""You are an expert document analyst. Convert the document to clean markdown:
{style_instruction}
CRITICAL RULES:
- Extract ALL text exactly as it appears - do not add, remove, or modify any content
- Do not add any external content, references, citations, or markup not present in the original
- REMOVE ALL icons and symbols (like 🩺, 💉, 📋, 📁, etc.) from the output. Do NOT include them in headings or text.
- Do not add new emojis, icons, or symbols not present in the original document
- Do not wrap any part of the output in markdown code blocks (```) unless the original contains them
- Do not repeat text or duplicate sections unless they appear that way in the original
- Preserve exact line breaks and spacing patterns

FORMATTING:
- STRICTLY differentiate heading levels based on the Style Guide. Do not make everything a Level 1 (#) heading. Use ##, ###, and #### for lower levels.
- Recognize titles that span multiple lines (e.g., long titles broken across lines for formatting) and merge them into one continuous heading line without inserting newlines, extra # symbols, or splitting them
- Do not insert # in the middle of what appears to be a single title entity
- Lists: - Bullet or 1. Numbered
- Bold: **bold**, Italic: *italic*
- Tables: | Col1 | Col2 |\n|------|------|\n| Data | Data |
- Equations: $LaTeX$
- Images/Diagrams: Describe comprehensively (elements, labels, relationships)
- Charts: Describe data, axes, legends, trends

SPATIAL PRESERVATION (AGGRESSIVE NORMALIZATION):
- Use EXACTLY single \n for paragraph breaks
- Use EXACTLY double \n\n for section breaks
- Use EXACTLY triple \n\n\n for page/chapter breaks
- Eliminate ALL horizontal whitespace (tabs, multiple spaces) except in code blocks
- Normalize indentation to 4 spaces for lists
- Never more than 3 consecutive newlines
- Standardize spacing: paragraphs = \n, sections = \n\n, pages = \n\n\n

Output ONLY the converted markdown - nothing else, no additions, no modifications."""

        # Add page context to help with ordering
        if len(images) > 1:
            prompt = f"Page {i+1} of {len(images)}: {prompt}"

        try:
            markdown = query_grok_vlm(prompt, image=image, model=model)
            # Clean up the response - remove commentary and duplicate content
            if markdown and len(markdown.strip()) > 0:
                markdown = _clean_response(markdown)
                # Normalize spacing for consistent downstream processing
                markdown = _normalize_markdown_spacing(markdown)

                if markdown.strip():  # Only add if there's actual content after cleaning
                    full_markdown += f"<!-- Page {i+1} -->\n{markdown}\n\n"
                else:
                    full_markdown += f"<!-- Page {i+1} - No content extracted -->\n\n"
            else:
                full_markdown += f"<!-- Page {i+1} - No content extracted -->\n\n"

        except Exception as e:
            logger.error(f"Error processing page {i+1}: {e}")
            full_markdown += f"<!-- Page {i+1} Error: {e} -->\n\n"

    return full_markdown.strip()