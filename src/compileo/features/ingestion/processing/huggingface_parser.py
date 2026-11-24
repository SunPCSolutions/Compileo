import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, AutoConfig, PretrainedConfig, BitsAndBytesConfig
import os
from typing import Optional, List
import io
from .image_utils import convert_pdf_to_optimized_images
from ....core.logging import get_logger

logger = get_logger(__name__)

# Define the model name from HuggingFace Hub
MODEL_NAME = "nanonets/Nanonets-OCR2-3B"

# Determine base path for the cache directory
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
    """
    logger.debug(f"Loading HuggingFace model: {MODEL_NAME}")
    logger.debug(f"Cache directory: {os.path.abspath(CACHE_DIR)}")

    # DIAGNOSTIC: Check CUDA availability before loading
    logger.debug("=== GPU DIAGNOSTIC START ===")
    logger.debug(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.debug(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
        logger.debug(f"torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
        logger.debug(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
    else:
        logger.debug("CUDA is NOT available! Model will use CPU.")
    logger.debug("=== GPU DIAGNOSTIC END ===")

    # Fetch HUGGING_FACE_HUB_TOKEN from environment (set by job executor)
    logger.debug("Checking for HUGGING_FACE_HUB_TOKEN in environment...")
    token = os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if token:
        logger.debug(f"SUCCESS: HUGGING_FACE_HUB_TOKEN found in environment")
    else:
        logger.debug("HUGGING_FACE_HUB_TOKEN not found in environment - HuggingFace downloads may fail if token is required")

    try:
        logger.debug(f"Attempting to load model with GPU optimization...")
        
        # Determine the best device mapping based on CUDA availability
        if torch.cuda.is_available():
            # Use explicit CUDA device mapping instead of "auto" which can fail silently
            device_map = "cuda"
            logger.debug(f"Using explicit device_map='cuda' for GPU acceleration")
        else:
            device_map = "auto"
            logger.debug("CUDA not available, using device_map='auto'")

        # Try different attention implementations in order of preference
        # flash_attention_2 requires specific GPU hardware (Ampere, Ada, newer)
        attention_options = ["flash_attention_2", "sdpa", "eager"]
        
        model = None
        last_error = None
        successful_attn = None
        
        for attn_impl in attention_options:
            try:
                logger.debug(f"Trying attn_implementation='{attn_impl}'...")
                
                # Load model with performance optimizations
                model = AutoModelForImageTextToText.from_pretrained(
                    MODEL_NAME,
                    cache_dir=CACHE_DIR,
                    device_map=device_map,
                    dtype="float16",  # Use string to avoid deprecation warning
                    attn_implementation=attn_impl,
                    trust_remote_code=True
                )
                logger.debug(f"Successfully loaded with attn_implementation='{attn_impl}'")
                successful_attn = attn_impl
                break
            except Exception as attn_error:
                logger.debug(f"attn_implementation='{attn_impl}' failed: {attn_error}")
                last_error = attn_error
                continue
        
        if model is None:
            raise Exception(f"All attention implementations failed. Last error: {last_error}")
        logger.debug(f"Model loaded with {successful_attn} and float16 precision")
        model.eval()  # Set to evaluation mode
        
        # DIAGNOSTIC: Verify actual device after loading
        actual_device = next(model.parameters()).device
        logger.debug(f"Model loaded successfully on device: {actual_device}")
        
        # DIAGNOSTIC: Log warning if model is on CPU despite GPU being available
        if actual_device.type == 'cpu' and torch.cuda.is_available():
            logger.warning(f"Model loaded on CPU despite GPU being available! This is a performance issue.")
            logger.debug(f"Possible cause: device_map='auto' failed to detect GPU. Try setting device_map='cuda' explicitly.")

    except Exception as e:
        logger.debug(f"Failed to load model with GPU config: {e}")
        
        # Try fallback with explicit CUDA device mapping
        if torch.cuda.is_available():
            logger.debug("Attempting fallback with explicit device_map='cuda'...")
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    MODEL_NAME,
                    cache_dir=CACHE_DIR,
                    device_map="cuda",  # Explicit CUDA device
                    dtype="float16",  # Use string to avoid deprecation warning
                    attn_implementation="flash_attention_2",
                    trust_remote_code=True
                )
                model.eval()
                actual_device = next(model.parameters()).device
                logger.debug(f"Fallback GPU load succeeded on device: {actual_device}")
            except Exception as cuda_fallback_e:
                logger.debug(f"Explicit CUDA fallback failed: {cuda_fallback_e}")
                logger.debug("Falling back to CPU...")
                try:
                    # Final fallback to CPU-only loading
                    model = AutoModelForImageTextToText.from_pretrained(
                        MODEL_NAME,
                        cache_dir=CACHE_DIR,
                        device_map=None,
                        dtype="float32",  # Use string to avoid deprecation warning
                        trust_remote_code=True
                    )
                    model.eval()
                    logger.debug(f"Model loaded successfully on CPU (final fallback)")
                except Exception as fallback_e:
                    logger.error(f"CPU loading also failed: {fallback_e}")
                    raise
        else:
            # No CUDA available, fall back to CPU
            logger.debug("No CUDA available, falling back to CPU...")
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    MODEL_NAME,
                    cache_dir=CACHE_DIR,
                    device_map=None,
                    dtype="float32",  # Use string to avoid deprecation warning
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

    logger.info("HuggingFace model, processor, and tokenizer loaded successfully")
    return model, processor, tokenizer

def parse_image_with_hf_vlm(image: Image.Image, model, processor, style_guide: Optional[str] = None):
    """
    Parses an image with the Hugging Face VLM.
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

        # Generate response
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64000,
            do_sample=False,
            repetition_penalty=1.2,
            use_cache=True
        )

        # Extract generated tokens
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]

        # Decode the response
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return output_text[0] if output_text else ""

    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        return f"Error processing image: {e}"

def analyze_document_structure(pdf_path: str, api_key: Optional[str] = None, preloaded_model=None, preloaded_processor=None) -> str:
    """
    Analyzes the document structure to create a visual style guide for heading levels.
    """
    if not os.path.exists(pdf_path):
        return ""

    try:
        # Load model if needed
        if preloaded_model is not None and preloaded_processor is not None:
            model, processor = preloaded_model, preloaded_processor
        else:
            try:
                model, processor, _ = load_model()
            except Exception:
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
        logger.warning(f"HuggingFace structure analysis failed: {e}")
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
    """
    logger.info(f"Starting HuggingFace PDF parsing for: {pdf_path}")

    if not os.path.exists(pdf_path):
        return f"Error: File not found at: {pdf_path}"

    try:
        # Use optimized image conversion
        images = convert_pdf_to_optimized_images(pdf_path, dpi=300)
    except Exception as e:
        return f"Error converting PDF to optimized images: {e}"

    full_markdown = ""

    # Use preloaded model if provided, otherwise load new one
    if preloaded_model is not None and preloaded_processor is not None:
        model, processor = preloaded_model, preloaded_processor
    else:
        try:
            model, processor, tokenizer = load_model()
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            return f"Error loading model: {e}"

    logger.info(f"Model ready, processing {len(images)} pages...")
    
    for i, image in enumerate(images):
        logger.debug(f"Processing page {i+1}/{len(images)}...")
        try:
            markdown = parse_image_with_hf_vlm(image, model, processor, style_guide)
            full_markdown += markdown + "\n\n"
        except Exception as e:
            logger.error(f"Error processing page {i+1}: {e}", exc_info=True)
            full_markdown += f"Error processing page {i+1}: {e}\n\n"

    logger.info(f"HuggingFace parsing complete, total markdown length: {len(full_markdown)}")
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
