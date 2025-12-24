import torch
from PIL import Image
from ....core.logging import get_logger

logger = get_logger(__name__)
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, AutoConfig, PretrainedConfig, BitsAndBytesConfig
import os
from typing import Optional, List
from .image_utils import convert_pdf_to_optimized_images

# Define the model name from HuggingFace Hub
MODEL_NAME = "nanonets/Nanonets-OCR2-3B"

# Determine base path for the cache directory
# In Docker, use absolute path to ensure reliability regardless of working directory
if os.path.exists("/app"):
    CACHE_DIR = "/app/src/compileo/features/ingestion/hf_models"
else:
    # Local environment fallback
    CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hf_models"))

# Ensure the cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

def load_model():
    """
    Loads the Hugging Face model, processor, and tokenizer with GPU optimization when available.
    Based on official Nanonets-OCR-s documentation.
    """
    logger.debug(f"Loading HuggingFace model: {MODEL_NAME}")
    logger.debug(f"Cache directory: {os.path.abspath(CACHE_DIR)}")

    # ALWAYS fetch HUGGING_FACE_HUB_TOKEN from database via BackendSettings
    logger.debug("Attempting to load HUGGING_FACE_HUB_TOKEN exclusively from database...")
    try:
        # Clear any existing environment variable to ensure DB is the only source
        if 'HUGGING_FACE_HUB_TOKEN' in os.environ:
            del os.environ['HUGGING_FACE_HUB_TOKEN']
            
        from src.compileo.core.settings import BackendSettings
        token = BackendSettings.get_huggingface_hub_token()
        if token:
            os.environ['HUGGING_FACE_HUB_TOKEN'] = token
            logger.debug(f"SUCCESS: HUGGING_FACE_HUB_TOKEN loaded from database (length: {len(token)})")
        else:
            logger.warning("WARNING: HUGGING_FACE_HUB_TOKEN not found in database settings table")
    except Exception as e:
        logger.error(f"CRITICAL ERROR: Failed to load HUGGING_FACE_HUB_TOKEN from database: {e}", exc_info=True)

    # Final verification log
    final_token_status = bool(os.environ.get('HUGGING_FACE_HUB_TOKEN'))
    logger.debug(f"FINAL HUB TOKEN STATUS: {'SET' if final_token_status else 'MISSING'}")

    try:
        logger.debug(f"Attempting to load model from HuggingFace Hub...")

        # Load model with performance optimizations
        # Re-enabling flash_attention_2 after manual model copy
        logger.debug(f"Calling AutoModelForImageTextToText.from_pretrained for {MODEL_NAME}...")
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            device_map="auto",  # Let transformers auto-manage GPU distribution
            torch_dtype=torch.float16,  # Force float16 for stability
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        )
        logger.debug("from_pretrained() call returned successfully")
        logger.debug(f"Model loaded with flash attention 2 and float16 precision (optimal for this model)")
        model.eval()  # Set to evaluation mode
        logger.debug(f"Model loaded successfully with fixed config on device: {next(model.parameters()).device}")

    except Exception as e:
        logger.warning(f"Failed to load model with basic config: {e}")
        logger.debug(f"Falling back to CPU loading...")
        try:
            # Fallback to CPU-only loading
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL_NAME,
                cache_dir=CACHE_DIR,
                device_map=None,
                dtype=torch.float32,
                trust_remote_code=True
            )
            model.eval()
            logger.debug(f"Model loaded successfully on CPU")
        except Exception as fallback_e:
            logger.error(f"CPU loading also failed: {fallback_e}")
            raise

    try:
        processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, use_fast=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        logger.debug(f"Processor (fast) and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load processor/tokenizer: {e}")
        raise

    logger.debug("HuggingFace model, processor, and tokenizer loaded successfully")
    return model, processor, tokenizer

def parse_image_with_hf_vlm(image: Image.Image, model, processor, style_guide: Optional[str] = None):
    """
    Parses an image with the Hugging Face VLM using the official Nanonets-OCR-s API.
    Based on the official HuggingFace documentation.
    """
    try:
        # Inject style guide
        style_instruction = ""
        if style_guide:
            style_instruction = f"\nSTYLE GUIDE CONTEXT (STRICTLY FOLLOW THIS):\n{style_guide}\n\nINSTRUCTION: Use the visual cues described above to determine Heading Levels (#, ##, ###). If text matches the visual description of a Header, FORMAT IT AS A HEADER.\n"

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

        # Create messages in the format expected by the model
        # Prepend instruction marker to prompt
        final_prompt = f"{prompt}\n\n[End of Instruction] Start processing document:"

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": final_prompt},
            ]},
        ]

        # Apply chat template and prepare inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)

        # Generate response with optimized parameters for better GPU utilization
        # Note: temperature parameter not supported by this model, using do_sample=False for deterministic output
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64000,  # official example is (15000)
            do_sample=False,  # Deterministic generation (temperature not supported by this model)
            repetition_penalty=1.2,  # Better results for complex tables
            use_cache=True  # Enable KV cache for better performance
        )

        # Extract generated tokens (excluding input)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]

        # Decode the response
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return output_text[0] if output_text else ""

    except Exception as e:
        error_msg = f"Error processing image: {e}"
        logger.error(f"[DEBUG_20251005_HuggingFaceParserBug] - {error_msg}", exc_info=True)
        return error_msg

def analyze_document_structure(pdf_path: str, api_key: Optional[str] = None, preloaded_model=None, preloaded_processor=None) -> str:
    """
    Analyzes the document structure to create a visual style guide for heading levels.
    """
    if not os.path.exists(pdf_path):
        return ""

    # Set API key if provided (needed for model loading if not preloaded)
    if api_key:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = api_key

    try:
        # Load model if needed
        if preloaded_model is not None and preloaded_processor is not None:
            model, processor = preloaded_model, preloaded_processor
        else:
            try:
                model, processor, _ = load_model()
            except Exception as e:
                logger.error(f"Failed to load model during structure analysis: {e}")
                return ""

        # Convert all pages with higher DPI for structure analysis
        images = convert_pdf_to_optimized_images(pdf_path, dpi=300)
        if not images: return ""

        prompt = """Analyze the visual layout of this entire document (all provided pages) to create a Style Guide for markdown conversion.
Identify the specific visual cues used for the hierarchy across the whole document:

1. Main Title (if present): Describe font size, weight (bold?), alignment.
2. Level 1 Headings (H1): Describe visual appearance (size relative to body, bold?).
3. Level 2 Headings (H2): Describe visual appearance.
4. Body Text: Describe standard appearance.

Output a comprehensive 'Style Guide' describing these rules to help a future AI model correctly identify headings consistently across all pages.
Do not extract content."""

        messages = [
            {"role": "user", "content": [
                *[{"type": "image", "image": img} for img in images],
                {"type": "text", "text": prompt}
            ]}
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=False
        )
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        return output_text[0] if output_text else ""

    except Exception as e:
        logger.error(f"HuggingFace structure analysis failed: {e}")
        return ""

def parse_pdf_with_vlm(
    pdf_path: str,
    api_key: Optional[str] = None,
    preloaded_model=None,
    preloaded_processor=None,
    overlap_text: Optional[str] = None,
    style_guide: Optional[str] = None
) -> str:
    """
    Parses a PDF using the Hugging Face VLM.

    Args:
        pdf_path: Path to the PDF file
        api_key: HuggingFace API key (optional, will use environment variable if not provided)
        preloaded_model: Pre-loaded model to reuse (optional)
        preloaded_processor: Pre-loaded processor to reuse (optional)
        style_guide: Optional style guide for context-aware parsing
    """
    logger.debug(f"[DEBUG_20251005_HuggingFaceParserBug] - Starting HuggingFace PDF parsing for: {pdf_path}")
    logger.debug(f"[DEBUG_20251005_HuggingFaceParserBug] - API key provided: {bool(api_key)}")
    logger.debug(f"[DEBUG_20251005_HuggingFaceParserBug] - HUGGING_FACE_HUB_TOKEN in env: {bool(os.environ.get('HUGGING_FACE_HUB_TOKEN'))}")
    logger.debug(f"[DEBUG_20251005_HuggingFaceParserBug] - Preloaded model provided: {preloaded_model is not None}")

    if not os.path.exists(pdf_path):
        error_msg = f"Error: File not found at: {pdf_path}"
        logger.error(error_msg)
        return error_msg

    # Set API key if provided
    if api_key:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = api_key
        logger.debug("Set HUGGING_FACE_HUB_TOKEN from parameter")
    elif not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        error_msg = "Error: HuggingFace API key not found. Please set HUGGING_FACE_HUB_TOKEN environment variable or provide api_key parameter."
        logger.error(error_msg)
        return error_msg

    logger.debug(f"API key configured, token present: {bool(os.environ.get('HUGGING_FACE_HUB_TOKEN'))}")

    try:
        # Use optimized image conversion for VLM processing (300dpi for better OCR quality)
        images = convert_pdf_to_optimized_images(pdf_path, dpi=300)
    except Exception as e:
        error_msg = f"Error converting PDF to optimized images: {e}"
        logger.error(f"[DEBUG_20251005_HuggingFaceParserBug] - {error_msg}")
        return error_msg

    full_markdown = ""

    # Use preloaded model if provided, otherwise load new one
    if preloaded_model is not None and preloaded_processor is not None:
        logger.debug("[DEBUG_20251005_HuggingFaceParserBug] - Using preloaded model and processor")
        model, processor = preloaded_model, preloaded_processor
    else:
        try:
            logger.debug("[DEBUG_20251005_HuggingFaceParserBug] - About to load model")
            model, processor, tokenizer = load_model()
            logger.debug("[DEBUG_20251005_HuggingFaceParserBug] - load_model() returned successfully")
            logger.debug(f"[DEBUG_20251005_HuggingFaceParserBug] - Model type: {type(model)}, Processor type: {type(processor)}, Tokenizer type: {type(tokenizer)}")
        except Exception as e:
            error_msg = f"Error loading model: {e}"
            logger.error(f"[DEBUG_20251005_HuggingFaceParserBug] - {error_msg}", exc_info=True)
            return error_msg

    logger.debug(f"[DEBUG_20251005_HuggingFaceParserBug] - Model ready, processing {len(images)} pages...")

    # Inject style guide into prompt logic inside parse_image_with_hf_vlm
    # We need to pass style_guide to parse_image_with_hf_vlm or modify the function
    # Since parse_image_with_hf_vlm is defined above, let's redefine it or pass it as an argument
    # To avoid changing the signature of parse_image_with_hf_vlm globally if used elsewhere,
    # we can modify the prompt construction here if we inline the logic, or update parse_image_with_hf_vlm
    
    # Updating parse_image_with_hf_vlm signature would require updating all calls.
    # Let's check where it's used. It's only used in this function loop.
    
    for i, image in enumerate(images):
        logger.debug(f"[DEBUG_20251005_HuggingFaceParserBug] - Processing page {i+1}/{len(images)}...")
        try:
            # We will call a modified version or update the prompt before calling
            # Wait, parse_image_with_hf_vlm has hardcoded prompt. We need to update it to accept prompt or style_guide.
            # I will modify parse_image_with_hf_vlm signature in a separate edit block below.
            markdown = parse_image_with_hf_vlm(image, model, processor, style_guide)
            logger.debug(f"[DEBUG_20251005_HuggingFaceParserBug] - Page {i+1} processed, markdown length: {len(markdown)}")
            full_markdown += markdown + "\n\n"
        except Exception as e:
            error_msg = f"Error processing page {i+1}: {e}"
            logger.error(f"[DEBUG_20251005_HuggingFaceParserBug] - {error_msg}", exc_info=True)
            full_markdown += error_msg + "\n\n"

    logger.debug(f"[DEBUG_20251005_HuggingFaceParserBug] - HuggingFace parsing complete, total markdown length: {len(full_markdown)}")
    return full_markdown

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Parse a PDF with a Hugging Face VLM.")
    parser.add_argument("pdf_path", help="The path to the PDF file.")
    args = parser.parse_args()

    markdown_output = parse_pdf_with_vlm(args.pdf_path)
    
    output_filename = "parsed_output.md"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(markdown_output)
        
    logger.info(f"\nParsing complete. Output saved to {output_filename}")