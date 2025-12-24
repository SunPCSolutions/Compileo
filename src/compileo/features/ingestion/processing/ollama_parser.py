import requests
import base64
from PIL import Image
import io
import os
import time
from typing import Optional, List, Literal
from pydantic import BaseModel
from .image_utils import convert_pdf_to_optimized_images
from ....core.logging import get_logger

logger = get_logger(__name__)

class DiagramElement(BaseModel):
    name: str
    description: str
    position: Optional[str] = None
    connections: Optional[List[str]] = None

class DiagramDescription(BaseModel):
    summary: str
    elements: List[DiagramElement]
    text_content: Optional[str] = None
    ascii_art: Optional[str] = None

def query_ollama_vlm(prompt, image=None, images: Optional[List[Image.Image]] = None, model: Optional[str] = None, options: Optional[dict] = None):
    """
    Sends a prompt and image(s) to the Ollama VLM and returns the response.

    Args:
        prompt: Text prompt for the model
        image: Single PIL Image object (legacy/simple use)
        images: List of PIL Image objects (for multi-image analysis)
        model: Ollama model name (optional)
        options: Dict of generation options to override defaults (optional)
    """
    from ....core.settings import backend_settings
    api_url = f"{backend_settings.get_ollama_base_url()}/api/generate"

    # Define default options from settings
    defaults = {
        "temperature": backend_settings.get_parsing_ollama_temperature(),
        "repeat_penalty": backend_settings.get_parsing_ollama_repeat_penalty(),
        "top_p": backend_settings.get_parsing_ollama_top_p(),
        "top_k": backend_settings.get_parsing_ollama_top_k(),
        "num_predict": backend_settings.get_parsing_ollama_num_predict(),
        "num_ctx": backend_settings.get_parsing_ollama_num_ctx()
    }
    
    seed = backend_settings.get_parsing_ollama_seed()
    if seed is not None:
        defaults["seed"] = seed

    # Merge provided options with defaults, filtering out None values
    merged_options = {k: v for k, v in {**defaults, **(options or {})}.items() if v is not None}

    # Convert images to base64
    base64_images = []
    
    if images:
        img_list = images
    elif image:
        img_list = [image]
    else:
        img_list = []
        
    for img in img_list:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        base64_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

    # Use model from settings if not provided
    if model is None:
        model = backend_settings.get_parsing_ollama_model()

    request_data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "images": base64_images,
        "options": merged_options
    }

    try:
        logger.debug("Sending request to Ollama...")
        start_time = time.time()

        response = requests.post(api_url, json=request_data)

        elapsed = time.time() - start_time
        logger.debug(f"Ollama response received in {elapsed:.1f}s")

        if response.status_code != 200:
            error_msg = f"Ollama API Error {response.status_code}: {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)

        result = response.json()
        if "response" not in result:
            raise Exception(f"Invalid Ollama response format: {result}")

        response_text = result["response"]
        logger.debug(f"Response length: {len(response_text)} characters")
        if response_text:
            logger.debug(f"Response preview: {response_text[:500]}{'...' if len(response_text) > 500 else ''}")
        else:
            logger.debug("Response is empty")

        return response_text

    except requests.exceptions.ConnectionError:
        raise Exception("Cannot connect to Ollama server. Make sure Ollama is running.")
    except Exception as e:
        logger.error(f"Ollama request failed: {e}")
        raise

def _clean_nanonets_response(response: str) -> str:
    """
    Clean unwanted commentary and analysis from Nanonets OCR responses.
    """
    # For now, return raw response to avoid processing issues
    return response.strip()

def analyze_document_structure(file_path: str, model: Optional[str] = None) -> str:
    """
    Analyzes the document structure to create a visual style guide for heading levels.
    """
    try:
        if file_path.lower().endswith('.pdf'):
            images = convert_pdf_to_optimized_images(file_path, dpi=300, max_dimension=1024)
            if not images: return ""
            
            prompt = """Analyze the visual layout of this entire document (all provided pages) to create a Style Guide for markdown conversion.
Identify the specific visual cues used for the hierarchy across the whole document:

1. Main Title (if present): Describe font size, weight (bold?), alignment.
2. Level 1 Headings (H1): Describe visual appearance (size relative to body, bold?).
3. Level 2 Headings (H2): Describe visual appearance.
4. Body Text: Describe standard appearance.

Output a comprehensive 'Style Guide' describing these rules to help a future AI model correctly identify headings consistently across all pages.
Do not extract content."""

            # Ollama models might struggle with too many images at once
            # We'll try sending all, but handle failures gracefully
            try:
                logger.debug("Analyzing document structure with Ollama...")
                return query_ollama_vlm(prompt, images=images, model=model)
            except Exception as e:
                logger.warning(f"Ollama batch analysis failed, trying first page only: {e}")
                return query_ollama_vlm(prompt, image=images[0], model=model)
                
    except Exception as e:
        logger.error(f"Ollama structure analysis failed: {e}")
        return ""
    return ""

def parse_document_with_ollama(
    file_path: str,
    model: Optional[str] = None,
    options: Optional[dict] = None,
    overlap_text: Optional[str] = None,
    style_guide: Optional[str] = None
) -> str:
    """
    Parses a document file using Ollama VLM. Supports PDFs, images, and text files.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_ext = os.path.splitext(file_path)[1].lower()

    # Handle different file types
    if file_ext == '.pdf':
        return _parse_pdf_with_ollama(file_path, model, options, style_guide)
    elif file_ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp']:
        return _parse_image_with_ollama(file_path, model, options)
    elif file_ext in ['.txt', '.md']:
        return _parse_text_with_ollama(file_path, model)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}. Ollama parser supports PDF, image, and text files.")

def _parse_pdf_with_ollama(pdf_path: str, model: Optional[str] = None, options: Optional[dict] = None, style_guide: Optional[str] = None) -> str:
    """
    Parses a PDF by converting pages to images and processing with Ollama VLM.
    """
    try:
        images = convert_pdf_to_optimized_images(pdf_path, dpi=300)
    except Exception as e:
        raise ValueError(f"Failed to process PDF file: {e}")

    full_markdown = ""

    for i, image in enumerate(images):
        logger.debug(f"Processing page {i+1}/{len(images)}...")
        
        # Inject style guide
        style_instruction = ""
        if style_guide:
            style_instruction = f"\nSTYLE GUIDE CONTEXT (STRICTLY FOLLOW THIS):\n{style_guide}\n\nINSTRUCTION: Use the visual cues described above to determine Heading Levels (#, ##, ###). If text matches the visual description of a Header, FORMAT IT AS A HEADER.\n"

        # Simplified comprehensive document processing prompt
        prompt = f"""You are an expert document analyst. Convert the document to clean markdown:
{style_instruction}
    CRITICAL RULES:
    - ALWAYS use markdown headings (#, ##, ###) for titles and section headers based on the Style Guide
    - Extract ALL text exactly as it appears - do not add, remove, or modify any content
    - Do not add any external content, references, citations, or markup not present in the original
    - REMOVE ALL icons and symbols (like 🩺, 💉, 📋, 📁, etc.) from the output. Do NOT include them in headings or text.
    - Do not add new emojis, icons, or symbols not present in the original document
    - Do not wrap any part of the output in markdown code blocks (```) unless the original contains them
    - Do not repeat text or duplicate sections unless they appear that way in the original
    - Preserve exact line breaks and spacing patterns
    
    FORMATTING:
    - STRICTLY enforce heading levels based on the Style Guide.
    - Main Titles -> # Heading
    - Section Headers -> ## Heading
    - Subsections -> ### Heading
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

    OUTPUT FORMAT EXAMPLE:
    # Main Title
    
    ## Section 1
    Body text...
    
    ## Section 2
    Body text...

    Output ONLY the converted markdown - nothing else, no additions, no modifications."""

        # Add page context to help with ordering
        if len(images) > 1:
            prompt = f"{prompt}\n\n[End of Instruction] Start processing Page {i+1} of {len(images)}:"

        try:
            # Use kwargs for model and options to avoid positional argument mismatch
            markdown = query_ollama_vlm(prompt, image=image, model=model, options=options)
            # Clean up the response - remove commentary and duplicate content
            if markdown and len(markdown.strip()) > 0:
                markdown = _clean_nanonets_response(markdown)

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

def _parse_image_with_ollama(image_path: str, model: Optional[str] = None, options: Optional[dict] = None) -> str:
    """
    Parses an image file using Ollama VLM.
    """
    try:
        image = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Failed to open image file: {e}")

    # Simplified comprehensive document processing prompt
    prompt = """You are an expert document analyst. Convert the document to clean markdown:

CRITICAL RULES:
- Extract ALL text exactly as it appears - do not add, remove, or modify any content
- Do not add any external content, references, citations, or markup not present in the original
- ICON CONVERSION: If document contains icons/symbols (💉, 📋, etc.) preceding section titles, replace them with appropriate markdown headings (# for main, ## for subsections, ### for minor) based on visual prominence and hierarchy
- Do not add new emojis, icons, or symbols not present in the original document
- Do not wrap any part of the output in markdown code blocks (```) unless the original contains them
- Do not repeat text or duplicate sections unless they appear that way in the original
- Preserve exact line breaks and spacing patterns

FORMATTING:
- Headings: # Main, ## Section, ### Sub, #### Minor
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

    try:
        return query_ollama_vlm(prompt, image=image, model=model, options=options)
    except Exception as e:
        raise Exception(f"Failed to process image with Ollama: {e}")

def _parse_text_with_ollama(text_path: str, model: Optional[str] = None) -> str:
    """
    Parses a text file. For text files, we could use Ollama's text models,
    but since this is a VLM parser, we'll read the text directly for now.
    """
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        raise ValueError(f"Failed to read text file: {e}")

    # For text files, return the content as-is since Ollama VLM is designed for images
    # In the future, this could be enhanced to use text-only Ollama models
    return content

# Backward compatibility alias
def parse_pdf_with_ollama(pdf_path, model: Optional[str] = None, options: Optional[dict] = None):
    """
    Legacy function for backward compatibility. Use parse_document_with_ollama instead.
    """
    return parse_document_with_ollama(pdf_path, model, options)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Parse a PDF with an Ollama VLM.")
    parser.add_argument("pdf_path", help="The path to the PDF file.")
    args = parser.parse_args()

    parse_pdf_with_ollama(args.pdf_path)
    
    logger.info(f"\nParsing complete. Output saved to parsed_output.md")