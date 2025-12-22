import re
import json

from typing import Optional, List, Dict, Any
from ...core.logging import get_logger

logger = get_logger(__name__)
from .schema import ChunkingStrategy
from .engine import chunk_document

def chunk_file_with_strategy(file_path: str, strategy: ChunkingStrategy, manifest: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Chunks a file using the provided ChunkingStrategy.
    Handles both JSON-structured files (with main_content) and plain text files.

    Args:
        file_path: Path to the file to chunk (JSON or plain text)
        strategy: The chunking strategy to use
        manifest: Optional manifest data for multi-file documents

    Returns:
        List of text chunks
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw_content = f.read()

    # Check if this is a JSON-structured file (from sliding window parsing)
    content = raw_content
    try:
        json_data = json.loads(raw_content)
        if isinstance(json_data, dict) and 'main_content' in json_data:
            # Extract main_content from JSON structure
            content = json_data['main_content']
            logger.debug(f"DEBUG_JSON_CHUNKING: Extracted main_content from JSON file {file_path}, length: {len(content)}")
        else:
            logger.debug(f"DEBUG_JSON_CHUNKING: File {file_path} is not JSON-structured, using as plain text")
    except json.JSONDecodeError:
        logger.debug(f"DEBUG_JSON_CHUNKING: File {file_path} is not valid JSON, using as plain text")

    # Use the unified chunk_document function
    chunks = chunk_document(content, strategy, manifest)

    # Clean up each chunk to remove potential JSON-unfriendly content
    cleaned_chunks = []
    for chunk in chunks:
        # Remove image tags
        chunk = re.sub(r'\[Image:.*?\]', '', chunk)
        # Remove any other non-essential content that might interfere with parsing
        cleaned_chunks.append(chunk.strip())

    return cleaned_chunks


