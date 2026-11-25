import os
import re
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
import ollama
from google import genai
from .schema import ChunkingStrategy, LLMPromptStrategy, DelimiterStrategy, CharacterStrategy, TokenStrategy, SchemaStrategy, ManifestData
from typing import List, Optional, Dict, Any

from ...core.openai_compat import openai_sampling_options

# Import Grok API client for chunking
try:
    from ..taxonomy.api_client import GrokAPIClient
except ImportError:
    GrokAPIClient = None

# Maximum content size to prevent memory issues (10MB)
MAX_CONTENT_SIZE = 10 * 1024 * 1024

logger = logging.getLogger(__name__)


def _strip_leading_comment_preamble(content: str) -> str:
    """Strip parser-generated leading comment blocks before chunk evaluation."""
    if not content:
        return content

    cleaned = content.lstrip("\ufeff")
    while True:
        updated = re.sub(r"^\s*<!--[\s\S]*?-->\s*", "", cleaned, count=1)
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned.lstrip()


def _call_ai_model(prompt: str, strategy: LLMPromptStrategy) -> str:
    """Helper function to call AI models"""
    if "gemini" in strategy.model:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=strategy.model,
            contents=prompt,
            config={'system_instruction': strategy.system_instruction}
        )
        if response.candidates and response.candidates[0].content.parts:
            return response.text
        else:
            raise ValueError("Gemini API call failed: No valid response part found. This may be due to safety settings or an issue with the prompt.")
    elif "grok" in strategy.model:
        api_key = os.environ.get("GROK_API_KEY")
        if not api_key:
            raise ValueError("GROK_API_KEY environment variable not set.")
        if GrokAPIClient is None:
            raise ValueError("GrokAPIClient not available. Cannot use Grok for chunking.")
        grok_client = GrokAPIClient(api_key, strategy.model)
        return grok_client._make_api_call(prompt, "chunking")
    elif "gpt" in strategy.model or strategy.model.startswith(("gpt-", "chatgpt")):
        # OpenAI models
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        try:
            from openai import OpenAI
        except ImportError:
            raise ValueError("OpenAI package not installed. Install with: pip install openai")

        client = OpenAI(api_key=api_key)

        messages = [{"role": "user", "content": prompt}]
        if strategy.system_instruction:
            messages.insert(0, {"role": "system", "content": strategy.system_instruction})

        response = client.chat.completions.create(
            model=strategy.model,
            messages=messages,
            **openai_sampling_options(strategy.model, temperature=0.2),
        )

        return response.choices[0].message.content.strip()
    else:
        # Prepare options for Ollama
        options = strategy.options if strategy.options else {}
        response = ollama.chat(
            model=strategy.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options=options if options else None,
        )
        return response["message"]["content"].strip()

def _parse_chunk_response(response: str) -> List[str]:
    """Parse AI response into chunk headings or chunks"""
    # Try to parse as JSON first
    try:
        parsed = json.loads(response.strip())
        if isinstance(parsed, list):
            return parsed  # Direct chunks (for logical chunking)
        elif isinstance(parsed, dict) and 'chunks' in parsed:
            return parsed['chunks']  # Chunks in metadata
    except json.JSONDecodeError:
        pass

    # Fallback to comma-separated headings (for boundary-based chunking)
    headings = [h.strip() for h in response.split(',') if h.strip()]
    return headings

def _chunk_with_llm(content: str, strategy: LLMPromptStrategy, manifest_data: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Uses an LLM to identify chunk boundaries based on a user-defined prompt.
    Supports multi-file documents with manifest context for intelligent cross-file chunking.
    Enhanced to handle structured JSON content.
    """
    if not content or not content.strip():
        return []

    # Check if content is JSON structured
    try:
        parsed_content = json.loads(content)
        if isinstance(parsed_content, dict) and 'content_type' in parsed_content:
            # Check if this has overlap_content (cross-file processing)
            if 'overlap_content' in parsed_content and parsed_content.get('overlap_content'):
                return _chunk_structured_content(parsed_content, strategy)
            else:
                # No overlap_content - extract main_content and process as regular text
                main_content = parsed_content.get('main_content', '')
                content = main_content  # Replace content with main_content for normal processing
    except json.JSONDecodeError:
        pass  # Not JSON, process as regular text

    # For semantic chunking, check if we should use logical chunking
    user_instruction = getattr(strategy, 'user_instruction', '')
    is_logical_chunking = any(phrase in user_instruction.lower() for phrase in [
        "split into", "create chunks", "major sections", "logical chunks"
    ]) and any(char.isdigit() for char in user_instruction)

    # Check if prompt_template has placeholders that need to be filled
    prompt_template = strategy.prompt_template
    has_placeholders = '{' in prompt_template and '}' in prompt_template

    if is_logical_chunking:
        if has_placeholders:
            # Try to format with available placeholders, fallback gracefully
            try:
                prompt = prompt_template.format(
                    user_instruction=user_instruction,
                    overlap_content="",  # Empty for regular content
                    main_content=content
                )
            except KeyError:
                # If placeholders don't match, just use template + content
                prompt = prompt_template.replace('{user_instruction}', user_instruction) + f"\n\n{content}"
        else:
            # No placeholders, use template as-is + content
            prompt = prompt_template + f"\n\n{content}"

    else:
        # Use original prompt template with manifest context
        manifest_context = ""
        if manifest_data:
            # Extract file relationship information from manifest
            total_pages = manifest_data.get('total_pages', 'unknown')
            pages_per_split = manifest_data.get('pages_per_split', 'unknown')
            overlap_pages = manifest_data.get('overlap_pages', 'unknown')
            num_splits = len(manifest_data.get('splits', []))

            manifest_context = f"""

**MULTI-FILE DOCUMENT CONTEXT:**
This is part of a {total_pages}-page document split into {num_splits} files ({pages_per_split} pages per file, {overlap_pages} pages overlap).
- Content may flow across file boundaries due to page overlaps
- Consider semantic continuity when determining chunk boundaries
- Avoid breaking topics or concepts at artificial file boundaries
- Maintain logical flow and coherence in your chunking decisions"""

        if has_placeholders:
            try:
                prompt = prompt_template.format(
                    user_instruction=strategy.user_instruction,
                    overlap_content="",  # Empty for regular content
                    main_content=content
                ) + manifest_context
            except KeyError:
                # If placeholders don't match, fallback to simple format
                prompt = prompt_template.replace('{user_instruction}', strategy.user_instruction) + manifest_context + f"\n\nHere is the markdown text:\n{content}"
        else:
            prompt = prompt_template.format(user_instruction=strategy.user_instruction) + manifest_context + f"\n\nHere is the markdown text:\n{content}"

    # Call AI model
    headings_str = _call_ai_model(prompt, strategy)

    headings = _parse_chunk_response(headings_str)

    modified_text = content
    for heading in headings:
        replacement_target = f"\n{heading}" if not modified_text.startswith(heading) else heading
        modified_text = modified_text.replace(replacement_target, f"\n---CHUNK_BOUNDARY---\n{heading}")

    chunks = [chunk.strip() for chunk in modified_text.split("---CHUNK_BOUNDARY---") if chunk.strip()]
    return chunks

def _chunk_structured_content(content_dict: Dict[str, Any], strategy: LLMPromptStrategy) -> List[str]:
    """Handle structured content with main_content and optional overlap_content"""

    main_content = content_dict.get('main_content', '')
    overlap_content = content_dict.get('overlap_content', '')  # Optional - defaults to empty string
    metadata = content_dict.get('metadata', {})

    # If no overlap_content, just process main_content as regular text (not as structured content)
    if not overlap_content:
        # Return main_content as a single item list to be processed by normal semantic chunking logic
        # This will cause the _chunk_with_llm function to process main_content as regular text
        return [main_content]

    # Use the original strategy's prompt template and user instruction
    # This ensures semantic chunking uses the user's specific requirements
    base_prompt = strategy.prompt_template.format(user_instruction=strategy.user_instruction)

    # Cross-file chunking with overlap content
    structured_prompt = f"""{base_prompt}

CONTENT TO CHUNK:
[OVERLAP CONTENT - Guaranteed chunk start boundary]
{overlap_content}

[MAIN CONTENT - Content to analyze and chunk]
{main_content}

CHUNKING RULES:
1. The overlap_content represents a guaranteed start boundary for the first chunk
2. Only create chunks that have BOTH clear start AND end boundaries within this content
3. Do not create incomplete chunks - if content doesn't have a clear end boundary, leave it unchunked
4. Use your chunking strategy to find natural semantic boundaries within the provided content
5. Return only complete, well-formed chunks that start and end within this text

PROCESSING: Find complete chunks starting from the overlap boundary, ensuring each chunk has both a clear beginning and end.
"""

    # Use AI to process structured content
    response = _call_ai_model(structured_prompt, strategy)
    chunks = _parse_chunk_response(response)

    return chunks

def _chunk_with_delimiter(content: str, strategy: DelimiterStrategy) -> List[str]:
    """
    Splits the text using a simple delimiter.
    """
    if not content or not content.strip():
        return []

    content = _strip_leading_comment_preamble(content)

    return [chunk.strip() for chunk in content.split(strategy.delimiter) if chunk.strip()]

def _chunk_with_characters(content: str, strategy: CharacterStrategy) -> List[str]:
    """
    Splits the text into chunks based on character count with overlap.

    Args:
        content: The text content to be chunked.
        strategy: The CharacterStrategy containing chunk_size and overlap.

    Returns:
        A list of text chunks.
    """
    if not content or not content.strip():
        return []

    if strategy.chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if strategy.overlap < 0:
        raise ValueError("overlap must be non-negative")
    if strategy.overlap >= strategy.chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    chunks = []
    start = 0
    content_length = len(content)
    
    while start < content_length:
        end = min(start + strategy.chunk_size, content_length)
        chunk = content[start:end]
        chunks.append(chunk)
        start += strategy.chunk_size - strategy.overlap

    return chunks

def _chunk_with_tokens(content: str, strategy: TokenStrategy) -> List[str]:
    """
    Splits the text into chunks based on token count with overlap.

    Args:
        content: The text content to be chunked.
        strategy: The TokenStrategy containing chunk_size, overlap, and model.

    Returns:
        A list of text chunks.
    """
    if not content or not content.strip():
        return []

    try:
        import tiktoken
    except ImportError:
        raise ImportError("tiktoken is required for token-based chunking. Install with: pip install tiktoken")

    if strategy.chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if strategy.overlap < 0:
        raise ValueError("overlap must be non-negative")
    if strategy.overlap >= strategy.chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    # Get the tokenizer
    try:
        encoding = tiktoken.get_encoding(strategy.model)
    except KeyError:
        # Fallback to common encodings
        if "cl100k" in strategy.model:
            encoding = tiktoken.get_encoding("cl100k_base")
        elif "p50k" in strategy.model:
            encoding = tiktoken.get_encoding("p50k_base")
        elif "r50k" in strategy.model:
            encoding = tiktoken.get_encoding("r50k_base")
        else:
            encoding = tiktoken.get_encoding("cl100k_base")  # Default to GPT-3.5/4 tokenizer

    # Tokenize the entire content
    tokens = encoding.encode(content)
    total_tokens = len(tokens)

    chunks = []
    start = 0

    while start < total_tokens:
        end = min(start + strategy.chunk_size, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += strategy.chunk_size - strategy.overlap

    return chunks

def join_schema_parts(parts: Iterable[str]) -> str:
    """Build a logical document without introducing regex EOFs at file seams."""
    fragments = []
    size = 0
    for part in parts:
        if not part:
            continue
        if fragments and not fragments[-1].endswith("\n") and not part.startswith("\n"):
            fragments.append("\n")
            size += 1
        size += len(part.encode("utf-8"))
        if size > MAX_CONTENT_SIZE:
            raise ValueError(f"Combined schema input exceeds maximum limit of {MAX_CONTENT_SIZE} bytes")
        fragments.append(part)
    return "".join(fragments)


@dataclass(frozen=True)
class SchemaChunkSpan:
    start: int
    end: int
    matched_rule: bool


def schema_chunk_spans(content: str, strategy: SchemaStrategy) -> List[SchemaChunkSpan]:
    """Return trimmed chunk spans in source coordinates, marking rule-backed starts."""
    cleaned = _strip_leading_comment_preamble(content)
    offset = len(content) - len(cleaned)
    schema = json.loads(strategy.json_schema)
    cut_before = schema.get("include_pattern", False)
    cut_sets = []
    for rule in schema["rules"]:
        delimiter = rule["type"] == "delimiter"
        pattern = re.escape(rule["value"]) if delimiter else rule["value"]
        cut_sets.append({
            match.start() if cut_before and not delimiter else match.end()
            for match in re.finditer(pattern, cleaned, re.MULTILINE)
        })
    cuts = set.union(*cut_sets) if schema["combine"] == "any" else set.intersection(*cut_sets)
    edges = sorted(cuts | {0, len(cleaned)})
    spans = []
    for start, end in zip(edges, edges[1:]):
        raw = cleaned[start:end]
        if raw.strip():
            spans.append(SchemaChunkSpan(
                offset + end - len(raw.lstrip()),
                offset + start + len(raw.rstrip()),
                start in cuts,
            ))
    return spans

def chunk_document(content: str, strategy: ChunkingStrategy, manifest: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Main function for the chunking engine.

    This function takes the document content and a chunking strategy,
    and dispatches to the appropriate chunking function.

    Args:
        content: The text content of the document to be chunked.
        strategy: The chunking strategy to be used.
        manifest: Optional manifest data for multi-file documents.

    Returns:
        A list of text chunks.
    """
    # Validate content size
    if len(content.encode('utf-8')) > MAX_CONTENT_SIZE:
        raise ValueError(f"Content size exceeds maximum limit of {MAX_CONTENT_SIZE} bytes")

    # Validate manifest data if provided
    if manifest is not None:
        try:
            ManifestData(**manifest)
        except Exception as e:
            raise ValueError(f"Invalid manifest data: {e}")

    if isinstance(strategy, LLMPromptStrategy):
        return _chunk_with_llm(content, strategy, manifest)
    elif isinstance(strategy, DelimiterStrategy):
        return _chunk_with_delimiter(content, strategy)
    elif isinstance(strategy, CharacterStrategy):
        return _chunk_with_characters(content, strategy)
    elif isinstance(strategy, TokenStrategy):
        return _chunk_with_tokens(content, strategy)
    elif isinstance(strategy, SchemaStrategy):
        return [content[span.start:span.end] for span in schema_chunk_spans(content, strategy)]
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy.strategy}")
