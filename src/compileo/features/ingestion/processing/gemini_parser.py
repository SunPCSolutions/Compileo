"""
Gemini document parser for Compileo.
Real implementation with Google Gemini API calls using google.genai package.
"""

import os
import logging
import json
from typing import Optional

logger = logging.getLogger(__name__)

# Import google.genai for new API
try:
    from google import genai
except ImportError:
    genai = None
    logger.warning("google.genai package not available, falling back to REST API")

def analyze_document_structure(file_path: str, api_key: Optional[str] = None, model: Optional[str] = None) -> str:
    """
    Analyzes the document structure to create a visual style guide for heading levels.
    """
    try:
        # Get API key
        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                return ""

        if genai is None:
            logger.warning("google.genai package not available for structure analysis")
            return ""

        # For PDFs, convert all pages to images for analysis
        if file_path.lower().endswith('.pdf'):
            from .image_utils import convert_pdf_to_optimized_images
            import io
            from google.genai import types

            # Convert all pages with higher DPI for structure analysis
            images = convert_pdf_to_optimized_images(file_path, dpi=300, max_dimension=1024)

            if not images:
                return ""

            # Prepare content for Gemini using google.genai
            client = genai.Client(api_key=api_key)

            prompt = """Analyze the visual layout of this entire document (all provided pages) to create a Style Guide for markdown conversion.
Identify the specific visual cues used for the hierarchy across the whole document:

1. Main Title (if present): Describe font size, weight (bold?), alignment.
2. Level 1 Headings (H1): Describe visual appearance (size relative to body, bold?).
3. Level 2 Headings (H2): Describe visual appearance.
4. Body Text: Describe standard appearance.

Output a comprehensive 'Style Guide' describing these rules to help a future AI model correctly identify headings consistently across all pages.
Do not extract content."""

            # Create contents with images and prompt
            contents = []
            for img in images:
                contents.append(img)
            contents.append(prompt)

            # Call Gemini using google.genai
            response = client.models.generate_content(
                model=model or "gemini-1.5-flash",
                contents=contents,
                config={"temperature": 0.1}
            )

            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text

    except Exception as e:
        logger.warning(f"Gemini structure analysis failed: {e}")
        return ""

def parse_document_with_gemini(
    file_path: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    overlap_text: Optional[str] = None,
    style_guide: Optional[str] = None
) -> str:
    """
    Parse document using Google Gemini AI using google.genai package.

    Uses google.genai client for consistent API access across the application.
    """
    try:
        logger.info(f"Parsing document with Gemini: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get API key
        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set or api_key parameter not provided.")

        if genai is None:
            raise ValueError("google.genai package not available. Cannot use Gemini for parsing.")

        # Initialize Gemini client
        from google.genai import types
        client = genai.Client(api_key=api_key)

        # Inject style guide if available
        style_instruction = ""
        if style_guide:
            style_instruction = f"""
STYLE GUIDE CONTEXT:
The document uses the following visual hierarchy:
{style_guide}

INSTRUCTION: Use these specific visual cues to determine Heading Levels (#, ##, ###).
"""

        prompt = f"""You are an expert document analyst. Convert the document to clean markdown:
{style_instruction}
CRITICAL RULES:
- Transcribe the document content accurately
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


SPATIAL PRESERVATION:
- Exact newlines: Single \n → single, double \n\n → double
- Preserve heading spacing exactly (e.g., "# Heading\n\nText" keeps one empty line, "# Heading\nText" keeps none)


Output ONLY the converted markdown - nothing else, no additions, no modifications."""

        # Handle file input (PDF vs Text) and create contents list
        if file_path.lower().endswith('.pdf'):
            file_size = os.path.getsize(file_path)

            # Check for large file size (>40MB) to fallback to images
            if file_size > 40 * 1024 * 1024:
                logger.warning(f"PDF file {os.path.basename(file_path)} is large ({file_size/1024/1024:.2f}MB). Converting to images for inline processing.")

                from .image_utils import convert_pdf_to_optimized_images
                import io

                # Convert to optimized images (balanced quality/size)
                images = convert_pdf_to_optimized_images(file_path, dpi=200, max_dimension=1536)

                # Create contents with images and prompt
                contents = []
                for img in images:
                    contents.append(img)
                contents.append(prompt)  # Add prompt at the end

                logger.info(f"Sending request to Gemini API for file: {os.path.basename(file_path)} (Mode: Images)")

            else:
                # Small enough PDF - send directly as bytes
                with open(file_path, 'rb') as f:
                    file_content = f.read()

                # Create contents with PDF bytes and prompt
                contents = [
                    types.Part.from_bytes(data=file_content, mime_type='application/pdf'),
                    prompt
                ]

                logger.info(f"Sending request to Gemini API for file: {os.path.basename(file_path)} (Mode: PDF)")

        else:
            # Text file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_content = f.read()

            # Create contents with text file content and prompt
            contents = [file_content, prompt]

            logger.info(f"Sending request to Gemini API for file: {os.path.basename(file_path)} (Mode: Text)")

        # Call Gemini using google.genai
        response = client.models.generate_content(
            model=model or "gemini-1.5-flash",
            contents=contents,
            config={"temperature": 0.1}
        )

        # Check for finish reasons
        if response.candidates and response.candidates[0].finish_reason:
            finish_reason = response.candidates[0].finish_reason
            if finish_reason == "RECITATION":
                return f"Error: Gemini refused to process this document due to Copyright/Recitation check ({finish_reason})."

        # Extract text safely
        if response.candidates and response.candidates[0].content.parts:
            markdown_content = response.candidates[0].content.parts[0].text
        else:
            markdown_content = ""

        # Clean the response
        if markdown_content:
            # Remove common prompt artifacts that might appear in the response
            prompt_artifacts = [
                "You are an expert document conversion specialist",
                "CRITICAL REQUIREMENTS:",
                "STRUCTURAL CONVERSION RULES:",
                "OUTPUT FORMAT:",
                "<document>",
                "</document>",
                "Convert the provided PDF document",
                "Return only the converted markdown content",
                "Return ONLY the converted markdown content"
            ]

            for artifact in prompt_artifacts:
                if artifact in markdown_content:
                    # Remove everything before and including the artifact
                    parts = markdown_content.split(artifact, 1)
                    if len(parts) > 1:
                        markdown_content = parts[1].strip()

            # Clean up instruction text and metadata
            lines = markdown_content.split('\n')
            cleaned_lines = []
            skip_next_lines = 0

            for line in lines:
                line = line.strip()

                # Skip lines that contain instructions or metadata
                if any(instruction in line.upper() for instruction in [
                    'PRESERVE', 'CONVERT', 'RETURN ONLY', 'EXTRACTED TEXT', 'CRITICAL',
                    'IMPORTANT:', 'CONTENT TO CONVERT', 'OUTPUT INSTRUCTION', 'YOU ARE',
                    '<DOCUMENT>', '</DOCUMENT>', 'STRUCTURAL CONVERSION', 'MARKDOWN CONTENT'
                ]):
                    skip_next_lines = 2  # Skip this line and next few lines
                    continue

                if skip_next_lines > 0:
                    skip_next_lines -= 1
                    continue

                # Skip empty lines at the beginning
                if not cleaned_lines and not line:
                    continue

                cleaned_lines.append(line)

            markdown_content = '\n'.join(cleaned_lines).strip()

            # Additional cleanup for common artifacts
            markdown_content = markdown_content.replace('```markdown', '').replace('```', '')

            # Remove any remaining instruction blocks
            while '##' in markdown_content and any(keyword in markdown_content.split('##', 1)[0].upper() for keyword in ['REQUIREMENTS', 'RULES', 'FORMAT']):
                parts = markdown_content.split('##', 1)
                if len(parts) > 1:
                    # Find the next heading or content
                    next_heading = parts[1].find('\n#')
                    if next_heading != -1:
                        markdown_content = parts[1][next_heading:]
                    else:
                        markdown_content = parts[1]
                else:
                    break

        if not markdown_content:
            error_msg = f"Error: Gemini API returned empty response for {os.path.basename(file_path)}"
            logger.error(error_msg)
            return error_msg

        logger.info(f"Document parsed successfully with Gemini: {len(markdown_content)} characters")
        return markdown_content

    except Exception as e:
        error_msg = f"Error: Gemini API call failed for {os.path.basename(file_path)}: {str(e)}"
        logger.error(error_msg)
        return error_msg