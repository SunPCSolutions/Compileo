import os
import re
import json
import ollama
from google import genai
from .schema import ChunkingStrategy, LLMPromptStrategy, DelimiterStrategy, CharacterStrategy, TokenStrategy, SchemaStrategy, ManifestData
from typing import List, Optional, Dict, Any
from ...core.logging import get_logger

logger = get_logger(__name__)

# Import Grok API client for chunking
try:
    from ..taxonomy.api_client import GrokAPIClient
except ImportError:
    GrokAPIClient = None

# Maximum content size to prevent memory issues (10MB)
MAX_CONTENT_SIZE = 10 * 1024 * 1024

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
            temperature=0.2  # Low temperature for consistent chunking
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
                logger.debug(f"DEBUG_PARSED_FILE: Processing parsed file without overlap_content, using main_content (length: {len(main_content)})")
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

        logger.debug(f"DEBUG_LOGICAL_CHUNKING: Using logical chunking with strategy prompt_template")
        logger.debug(f"DEBUG_LOGICAL_CHUNKING: User instruction: {user_instruction[:100]}...")
        logger.debug(f"DEBUG_LOGICAL_CHUNKING: Content length: {len(content)}")
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

    # DEBUG LOGGING: Capture the full prompt being sent to LLM
    logger.debug(f"DEBUG_CHUNKING_PROMPT: Model={strategy.model}")
    logger.debug(f"DEBUG_CHUNKING_PROMPT: Manifest present={manifest_data is not None}")
    logger.debug(f"DEBUG_CHUNKING_PROMPT: Content length={len(content)}")
    logger.debug(f"DEBUG_CHUNKING_PROMPT: Full prompt=\n{prompt}")
    logger.debug(f"DEBUG_CHUNKING_PROMPT: {'='*50}")

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
        logger.debug(f"DEBUG_STRUCTURED: No overlap_content, processing main_content as regular text (length: {len(main_content)})")
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

    logger.debug(f"DEBUG_STRUCTURED: Processing with overlap_content (length: {len(overlap_content)}) and main_content (length: {len(main_content)})")

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

def _chunk_with_schema(content: str, strategy: SchemaStrategy) -> List[str]:
    """
    Splits the text using a JSON schema that defines flexible splitting rules.

    The schema supports multiple criteria: patterns (regex), delimiters, and structural elements.
    Rules can be combined with 'any' (split if any rule matches) or 'all' (split if all rules match).

    For cross-file chunking, this function ensures that only complete sections are chunked.
    Incomplete sections (starting with a header but not ending with one) are left as single chunks
    to be handled by the cross-file logic.

    Args:
        content: The text content to be chunked.
        strategy: The SchemaStrategy containing the JSON schema.

    Returns:
        A list of text chunks.

    Raises:
        ValueError: If the JSON schema is invalid.
    """
    if not content or not content.strip():
        return []

    logger.debug(f"DEBUG_SCHEMA: Starting schema chunking with json_schema: {strategy.json_schema[:100]}...")

    try:
        schema = json.loads(strategy.json_schema)
        logger.debug(f"DEBUG_SCHEMA: Parsed schema: {schema}")
    except json.JSONDecodeError as e:
        logger.error(f"DEBUG_SCHEMA: JSON decode error: {e}")
        raise ValueError(f"Invalid JSON schema: {e}")

    rules = schema.get("rules", [])
    combine = schema.get("combine", "any")
    include_pattern = schema.get("include_pattern", False)  # Default to False for backward compatibility

    logger.debug(f"DEBUG_SCHEMA: Processing {len(rules)} rules with combine='{combine}'")

    # Collect split positions from all rules
    all_positions = []

    for rule in rules:
        rule_type = rule.get("type")
        value = rule.get("value")
        positions = set()

        logger.debug(f"DEBUG_SCHEMA: Processing rule type='{rule_type}', value='{value}'")
        logger.debug(f"DEBUG_SCHEMA: Content length: {len(content)}")
        logger.debug(f"DEBUG_SCHEMA: Content preview: {repr(content[:200])}")

        if rule_type == "pattern":
            # Find split positions based on include_pattern setting
            try:
                # Use MULTILINE flag to allow ^ and $ to match start/end of lines
                matches = list(re.finditer(value, content, flags=re.MULTILINE))
                logger.debug(f"DEBUG_SCHEMA: Regex '{value}' found {len(matches)} matches, include_pattern={include_pattern}")
                for match in matches:
                    if include_pattern:
                        positions.add(match.start())  # Split BEFORE the pattern (include pattern in chunk)
                    else:
                        positions.add(match.end())    # Split AFTER the pattern (exclude pattern from chunk)
            except re.error as e:
                logger.error(f"DEBUG_SCHEMA: Regex error: {e}")
                raise ValueError(f"Invalid regex pattern in schema rule: {e}")
        elif rule_type == "delimiter":
            # Find positions AFTER delimiter occurrences
            pos = 0
            while True:
                idx = content.find(value, pos)
                if idx == -1:
                    break
                positions.add(idx + len(value))  # Split AFTER the delimiter
                pos = idx + len(value)
        elif rule_type == "structure":
            if value == "heading":
                # Detect markdown headings (lines starting with #) and split after the heading line
                for match in re.finditer(r'^#+\s.*$', content, re.MULTILINE):
                    # Find the end of this line (including newline)
                    line_end = content.find('\n', match.end())
                    if line_end == -1:
                        # Last line, split at end of content
                        positions.add(len(content))
                    else:
                        # Split after this line
                        positions.add(line_end + 1)
            # Add more structure types as needed

        all_positions.append(positions)

    # Combine positions based on the combine strategy
    if combine == "any":
        # Union of all positions
        split_positions = set()
        for pos_set in all_positions:
            split_positions.update(pos_set)
    elif combine == "all":
        # Intersection of all positions (positions that appear in all rule sets)
        if all_positions:
            split_positions = all_positions[0]
            for pos_set in all_positions[1:]:
                split_positions = split_positions.intersection(pos_set)
        else:
            split_positions = set()
    else:
        raise ValueError(f"Invalid combine value: {combine}. Must be 'any' or 'all'.")

    # Sort positions and split content
    positions = sorted(split_positions)
    logger.debug(f"DEBUG_SCHEMA: Found {len(positions)} split positions: {positions}")

    # CROSS-FILE AWARE CHUNKING LOGIC
    # For schema chunking to work properly across files, we need to ensure that:
    # 1. Only complete sections (with both start and end boundaries) are split
    # 2. Incomplete sections are left as single chunks for cross-file handling

    # Check if this content represents an incomplete section (starts with schema pattern but no ending boundary)
    is_incomplete_section = False

    # Check if content starts with any of the schema patterns
    content_starts_with_pattern = False
    for rule in rules:
        rule_type = rule.get("type")
        value = rule.get("value")

        if rule_type == "pattern":
            # Check if content starts with this pattern
            try:
                # Use MULTILINE flag to allow ^ match start of string or line
                if re.match(value, content.strip(), flags=re.MULTILINE):
                    content_starts_with_pattern = True
                    logger.debug(f"DEBUG_SCHEMA: Content starts with pattern '{value}'")
                    break
            except re.error:
                continue  # Skip invalid regex patterns

    # If content starts with a pattern, check if it has a complete ending (another pattern match)
    if content_starts_with_pattern:
        # Look for additional pattern matches in the content (beyond the first one)
        has_additional_patterns = False
        for rule in rules:
            rule_type = rule.get("type")
            value = rule.get("value")

            if rule_type == "pattern":
                try:
                    # Use MULTILINE flag to match internal boundaries
                    matches = list(re.finditer(value, content, flags=re.MULTILINE))
                    if len(matches) > 1:  # More than one match means there's an ending boundary
                        has_additional_patterns = True
                        logger.debug(f"DEBUG_SCHEMA: Found {len(matches)} matches for pattern '{value}', has ending boundary")
                        break
                except re.error:
                    continue

        # If no additional patterns found, this is an incomplete section
        if not has_additional_patterns:
            is_incomplete_section = True
            logger.debug(f"DEBUG_SCHEMA: Content is incomplete section (starts with pattern but no ending boundary)")

    # If this is an incomplete section, return it as a single chunk for cross-file handling
    if is_incomplete_section:
        logger.debug(f"DEBUG_SCHEMA: Returning incomplete section as single chunk for cross-file handling")
        return [content.strip()]

    # Normal chunking logic for complete content
    chunks = []
    start = 0

    # Sort positions (these are now the split points - where we cut the content)
    for split_pos in sorted(positions):
        if split_pos > start:
            chunk = content[start:split_pos].strip()
            if chunk:
                chunks.append(chunk)
        start = split_pos

    # Add the remaining content as the last chunk
    last_chunk = content[start:].strip()
    if last_chunk:
        chunks.append(last_chunk)

    # If no splits were made, return the whole content as one chunk
    if not chunks:
        return [content.strip()]

    logger.debug(f"DEBUG_SCHEMA: Created {len(chunks)} chunks")
    return chunks

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
        logger.debug("DEBUG_SCHEMA: Calling _chunk_with_schema")
        return _chunk_with_schema(content, strategy)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy.strategy}")