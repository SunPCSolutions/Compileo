"""
OpenAI document parser for Compileo.
Implementation with OpenAI API (ChatGPT) calls, supporting GPT-4o Vision for multimodal parsing.
"""

import os
import logging
import base64
import json
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def analyze_document_structure(file_path: str, api_key: Optional[str] = None, model: Optional[str] = None) -> str:
    """
    Analyzes the document structure to create a visual style guide for heading levels.
    """
    try:
        from openai import OpenAI
        from .image_utils import convert_pdf_to_optimized_images
        import io
        
        # Get API key (logic copied from parse function)
        if api_key is None:
            from ....core.settings import backend_settings
            api_key = backend_settings.get_openai_api_key() or os.environ.get("OPENAI_API_KEY")
            if not api_key: return ""

        client = OpenAI(api_key=api_key)
        
        if model is None:
            from ....core.settings import backend_settings
            model = backend_settings.get_parsing_openai_model() or "gpt-4o"

        if file_path.lower().endswith('.pdf'):
            # Convert ALL pages for analysis
            images = convert_pdf_to_optimized_images(file_path, dpi=300, max_dimension=1024)
            if not images: return ""
            
            content_parts = []
            content_parts.append({"type": "text", "text": """Analyze the visual layout of this entire document (all provided pages) to create a Style Guide for markdown conversion.
Identify the specific visual cues used for the hierarchy across the whole document:

1. Main Title (if present): Describe font size, weight (bold?), alignment.
2. Level 1 Headings (H1): Describe visual appearance (size relative to body, bold?).
3. Level 2 Headings (H2): Describe visual appearance.
4. Body Text: Describe standard appearance.

Output a comprehensive 'Style Guide' describing these rules to help a future AI model correctly identify headings consistently across all pages.
Do not extract content."""})
            
            for img in images:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}", "detail": "low"}
                })

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content_parts}]
            )
            return response.choices[0].message.content or ""
            
    except Exception as e:
        logger.warning(f"OpenAI structure analysis failed: {e}")
        return ""

def parse_document_with_openai(
    file_path: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    overlap_text: Optional[str] = None,
    style_guide: Optional[str] = None
) -> str:
    """
    Parse document using OpenAI API (ChatGPT).
    Supports PDF (via conversion to images or text extraction) and images directly using Vision capabilities.

    Args:
        file_path: Path to the document file
        api_key: OpenAI API key
        model: Model to use (defaults to gpt-4o)

    Returns:
        str: Parsed document content as markdown
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("Error: 'openai' package not installed.")
        return "Error: 'openai' package not installed."

    try:
        logger.info(f"Parsing document with OpenAI: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get API key
        if api_key is None:
            # Fallback to environment variable or settings
            from ....core.settings import backend_settings
            api_key = backend_settings.get_openai_api_key()
            
            if not api_key:
                api_key = os.environ.get("OPENAI_API_KEY")
                
            if not api_key:
                raise ValueError("OpenAI API key not provided and not found in settings or environment.")

        client = OpenAI(api_key=api_key)
        
        # Use provided model or get from settings (default to gpt-4o)
        if model is None:
            from ....core.settings import backend_settings
            model = backend_settings.get_parsing_openai_model() or "gpt-4o"

        file_ext = os.path.splitext(file_path)[1].lower()

        # Handle different file types
        if file_ext == '.pdf':
            return _parse_pdf_with_openai(client, file_path, model, style_guide)
        elif file_ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp']:
            return _parse_image_with_openai(client, file_path, model)
        elif file_ext in ['.txt', '.md', '.csv', '.json', '.xml']:
            return _parse_text_with_openai(client, file_path, model)
        else:
            # Try parsing as text for unknown extensions, might fail if binary
            try:
                return _parse_text_with_openai(client, file_path, model)
            except Exception:
                raise ValueError(f"Unsupported file type: {file_ext}")

    except Exception as e:
        logger.error(f"Error parsing document with OpenAI: {e}")
        return f"Error parsing document: {str(e)}"


def _parse_pdf_with_openai(client, pdf_path: str, model: str, style_guide: Optional[str] = None) -> str:
    """
    Parses a PDF by converting pages to images and processing with OpenAI Vision.
    """
    from .image_utils import convert_pdf_to_optimized_images
    
    try:
        # Convert PDF to images
        images = convert_pdf_to_optimized_images(pdf_path, dpi=300)
    except Exception as e:
        raise ValueError(f"Failed to process PDF file: {e}")

    full_markdown = ""

    for i, image in enumerate(images):
        logger.info(f"Processing page {i+1}/{len(images)}...")
        
        # Convert image to base64
        import io
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Inject style guide
        style_instruction = ""
        if style_guide:
            style_instruction = f"\nSTYLE GUIDE CONTEXT:\n{style_guide}\nINSTRUCTION: Use these visual cues for headings (#, ##).\n"

        prompt = f"""You are an expert document analyst. Convert the document page to clean markdown.
{style_instruction}
CRITICAL RULES:
- Extract ALL text exactly as it appears.
- Do not add any external content or commentary.
- REMOVE ALL icons and symbols (like 🩺, 💉, 📋, 📁, etc.) from the output. Do NOT include them in headings or text.
- STRICTLY differentiate heading levels based on the Style Guide. Do not make everything a Level 1 (#) heading. Use ##, ###, and #### for lower levels.
- Preserve exact line breaks and spacing patterns where meaningful.
- Describe images/diagrams comprehensively.
- Output ONLY the converted markdown."""

        # Add page context
        if len(images) > 1:
            prompt = f"Page {i+1} of {len(images)}: {prompt}"

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ]
            )
            
            markdown = response.choices[0].message.content
            
            if markdown:
                # Clean up markdown formatting artifacts
                markdown = markdown.replace('```markdown', '').replace('```', '').strip()
                full_markdown += f"<!-- Page {i+1} -->\n{markdown}\n\n"
            else:
                full_markdown += f"<!-- Page {i+1} - No content extracted -->\n\n"

        except Exception as e:
            logger.error(f"Error processing page {i+1}: {e}")
            full_markdown += f"<!-- Page {i+1} Error: {e} -->\n\n"

    return full_markdown.strip()


def _parse_image_with_openai(client, image_path: str, model: str) -> str:
    """
    Parses an image file using OpenAI Vision.
    """
    try:
        with open(image_path, "rb") as image_file:
            img_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        raise ValueError(f"Failed to open image file: {e}")

    prompt = """You are an expert document analyst. Convert the document image to clean markdown.

CRITICAL RULES:
- Extract ALL text exactly as it appears.
- Do not add any external content or commentary.
- Preserve exact line breaks and spacing patterns where meaningful.
- Describe images/diagrams comprehensively.
- Output ONLY the converted markdown."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}", # Assuming jpeg/png compatible base64
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
        )
        
        markdown = response.choices[0].message.content
        if markdown:
            return markdown.replace('```markdown', '').replace('```', '').strip()
        return ""

    except Exception as e:
        raise Exception(f"Failed to process image with OpenAI: {e}")


def _parse_text_with_openai(client, text_path: str, model: str) -> str:
    """
    Parses a text file using OpenAI (useful for formatting or summarization if needed, 
    otherwise could just read the file).
    Here we mostly read the file but could use the model to "clean" or "format" it to markdown.
    For now, standard text reading is often sufficient and cheaper, but let's allow the model 
    to format it if it's unstructured text.
    """
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        raise ValueError(f"Failed to read text file: {e}")

    # For simple text files, we might just return the content. 
    # But if we want to "parse" it (e.g. convert unstructured text to markdown), we can use the model.
    # Given the cost, for now let's return raw content unless it's very messy.
    # To follow the pattern of other parsers that "convert to markdown":
    return content