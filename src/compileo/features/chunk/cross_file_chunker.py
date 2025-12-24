"""
Cross-File Chunker: Robust, scalable chunking system for multi-file documents.

This module provides a robust chunking system that handles cross-file boundaries,
maintaining state between files and ensuring complete chunks are formed even when
content spans multiple input files.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import json
import os
from ...core.logging import get_logger
import re
import uuid
from datetime import datetime

from .schema import CharacterStrategy, LLMPromptStrategy, SchemaStrategy, DelimiterStrategy, TokenStrategy

logger = get_logger(__name__)


@dataclass
class ChunkState:
    """State maintained between file processing operations."""
    last_incomplete_chunk: Optional[str] = None
    last_chunk_metadata: Optional[Dict[str, Any]] = None
    total_chunks_created: int = 0
    current_file_index: int = 0


class CrossFileChunker:
    """
    Handles chunking of multi-file documents with cross-file boundary support.

    This chunker maintains state between files to ensure that incomplete chunks
    at the end of one file are completed using content from subsequent files.
    """

    def __init__(self, db_connection=None):
        self.state = ChunkState()
        self.db_connection = db_connection
        self.logger = get_logger(__name__)

    def reset_state(self, document_id: Optional[str] = None) -> None:
        """Reset the chunker state for a new document."""
        self.state = ChunkState()
        self.current_document_id = document_id
        self.logger.info(f"Cross-file chunker state reset for document {document_id}")

    def process_document_files(self, files: List[Dict[str, Any]], strategy) -> List[str]:
        """Process a list of document files with cross-file chunking."""
        all_chunks = []

        for file_idx, file_data in enumerate(files):
            self.state.current_file_index = file_idx

            main_content = file_data.get("main_content", "")
            overlap_content = file_data.get("overlap_content", "")

            # Combine overlap + main content (overlap serves as guaranteed start boundary)
            combined_content = overlap_content + main_content if overlap_content else main_content

            # Process with chunking strategy
            file_chunks = self._process_file_with_strategy(combined_content, strategy)

            all_chunks.extend(file_chunks)

        return all_chunks

    def chunk_document(
        self,
        document_id: int,
        project_id: Union[str, int],
        manifest_data: Dict[str, Any],
        chunk_strategy: str,
        chunk_size: int,
        overlap: int,
        model: str,
        chunker: str,
        chunking_options: Optional[Dict[str, Any]] = None,
        schema_definition: str = "",
        delimiters: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Chunk a multi-file document using cross-file boundary logic.

        Args:
            document_id: ID of the document to chunk
            project_id: Project ID containing the document
            manifest_data: Manifest containing file information and overlap data
            chunk_strategy: Chunking strategy to use
            chunk_size: Size parameter for chunking
            overlap: Overlap parameter for chunking
            model: AI model to use
            chunker: Chunker type
            chunking_options: Additional options for chunking

        Returns:
            Dict containing chunking results and statistics
        """
        from ...storage.src.project.file_manager import FileManager
        from ...storage.src.project.database_repositories import ChunkRepository
        from ...core.settings import backend_settings

        # Validate manifest_data
        if manifest_data is None or not isinstance(manifest_data, dict):
            raise ValueError(f"Invalid manifest data: expected dict, got {type(manifest_data)}")

        self.logger.info(f"Starting cross-file chunking for document {document_id}")
        self.logger.debug(f"DEBUG_CROSS_FILE: manifest_data keys: {list(manifest_data.keys())}")
        self.logger.debug(f"DEBUG_CROSS_FILE: manifest_data parsed_files: {len(manifest_data.get('parsed_files', []))}")

        # Get API keys for chunkers that need them (only for AI-based strategies)
        needs_api_key = chunk_strategy in ["semantic", "llm_prompt"] or (chunk_strategy == "token" and chunker in ["gemini", "grok"])

        if needs_api_key:
            if chunker == "gemini":
                api_key = backend_settings.get_setting("gemini_api_key")
                if not api_key:
                    raise ValueError("Gemini API key not configured")
            elif chunker == "grok":
                api_key = backend_settings.get_setting("grok_api_key")
                if not api_key:
                    raise ValueError("Grok API key not configured")
            # Ollama doesn't need API key

        # chunking_options is now passed as parameter

        # Initialize components
        # FileManager handles storage paths internally, no storage_type arg needed
        file_manager = FileManager()
        chunk_repo = ChunkRepository(self.db_connection)

        parsed_files = manifest_data.get("parsed_files", [])
        if not parsed_files:
            raise ValueError("No parsed files found in manifest")

        self.logger.info(f"Processing {len(parsed_files)} files for document {document_id}")

        # Reset state for new document
        self.reset_state()

        # Process each file sequentially
        for file_idx, parsed_file_path in enumerate(parsed_files):
            self.state.current_file_index = file_idx

            if not os.path.exists(parsed_file_path):
                self.logger.warning(f"Skipping missing file: {parsed_file_path}")
                continue

            self.logger.info(f"Processing file {file_idx + 1}/{len(parsed_files)}: {os.path.basename(parsed_file_path)}")

            try:
                # Read and parse JSON file content
                with open(parsed_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                file_data = json.loads(content)
                self.logger.debug(f"DEBUG_FILE: file_data keys: {list(file_data.keys())}")
                self.logger.debug(f"DEBUG_FILE: main_content type: {type(file_data.get('main_content'))}")
                self.logger.debug(f"DEBUG_FILE: main_content preview: {repr(file_data.get('main_content', '')[:100])}")
                file_content = file_data["main_content"]
                self.logger.debug(f"DEBUG_FILE: extracted file_content length: {len(file_content)}")
                self.logger.debug(f"DEBUG_FILE: file_content preview: {repr(file_content[:100])}")

                # Create chunking strategy
                strategy = self._create_chunking_strategy(
                    chunk_strategy, chunk_size, overlap, model,
                    manifest_data, chunking_options, schema_definition, delimiters
                )

                # Process file with cross-file logic
                # Note: overlap_content from manifest is not used for chunking,
                # incomplete chunks are handled via state management
                is_last_file = (file_idx == len(parsed_files) - 1)
                file_chunks = self._process_file_with_overlap(
                    file_content, strategy, is_last_file
                )

                # Save completed chunks
                for chunk in file_chunks:
                    chunk_file_name = f"chunk_{self.state.total_chunks_created + 1}.md"
                    
                    # Ensure chunks directory exists
                    chunk_storage_path = file_manager.get_storage_base_path() / "chunks" / str(project_id) / str(document_id)
                    chunk_storage_path.mkdir(parents=True, exist_ok=True)
                    
                    # Save chunk file
                    full_chunk_path = chunk_storage_path / chunk_file_name
                    with open(full_chunk_path, "wb") as f:
                        f.write(chunk.encode("utf-8"))
                    
                    # Store relative path
                    relative_chunk_path = str(full_chunk_path.relative_to(file_manager.get_storage_base_path()))
                    
                    chunk_repo.create_chunk({
                        "id": str(uuid.uuid4()),
                        "document_id": document_id,
                        "chunk_index": self.state.total_chunks_created + 1,
                        "content": chunk,
                        "metadata": {"strategy": chunk_strategy, "file_path": relative_chunk_path},
                        "created_at": datetime.utcnow().isoformat()
                    })
                    self.state.total_chunks_created += 1

                self.logger.info(f"Saved {len(file_chunks)} chunks from file {os.path.basename(parsed_file_path)}")

            except Exception as e:
                self.logger.error(f"Failed to process file {parsed_file_path}: {e}")
                continue

        # Return results
        result = {
            "document_id": document_id,
            "status": "completed",
            "chunks_created": self.state.total_chunks_created,
            "files_processed": len(parsed_files),
            "chunk_strategy": chunk_strategy,
            "processing_mode": "cross_file_chunking"
        }

        self.logger.info(f"Cross-file chunking completed for document {document_id}: {self.state.total_chunks_created} chunks created")
        return result

    def _create_chunking_strategy(
        self,
        chunk_strategy: str,
        chunk_size: int,
        overlap: int,
        model: str,
        manifest_data: Dict[str, Any],
        chunking_options: Optional[Dict[str, Any]] = None,
        schema_definition: str = "",
        delimiters: Optional[List[str]] = None
    ):
        """Create the appropriate chunking strategy object."""
        # Normalize strategy to lowercase
        chunk_strategy = chunk_strategy.lower()

        if chunk_strategy == "character":
            return CharacterStrategy(
                strategy="character",
                chunk_size=chunk_size,
                overlap=overlap
            )
        elif chunk_strategy == "token":
            return TokenStrategy(
                strategy="token",
                chunk_size=chunk_size,
                overlap=overlap,
                model="cl100k_base"
            )
        elif chunk_strategy == "semantic":
            user_instruction = manifest_data.get("semantic_prompt", "Split the document by topic.")
            return LLMPromptStrategy(
                strategy="llm_prompt",
                model=model,
                prompt_template=self._get_semantic_prompt_template(user_instruction),
                user_instruction=user_instruction,
                options=chunking_options
            )
        elif chunk_strategy == "schema":
            return SchemaStrategy(
                strategy="schema",
                json_schema=schema_definition
            )
        elif chunk_strategy == "delimiter":
            # Use provided delimiters parameter, fallback to manifest_data or default
            delimiter_list = delimiters if delimiters is not None else manifest_data.get("delimiters", ["\n\n"])
            return DelimiterStrategy(
                strategy="delimiter",
                delimiter=delimiter_list[0] if delimiter_list else "\n\n"
            )
        else:
            # Default to character strategy
            return CharacterStrategy(
                strategy="character",
                chunk_size=chunk_size,
                overlap=overlap
            )

    def _get_semantic_prompt_template(self, user_instruction: str) -> str:
        """Get the appropriate semantic chunking prompt template."""
        # Detect if this is logical chunking or boundary-based chunking
        is_logical_chunking = any(phrase in user_instruction.lower() for phrase in [
            "split into", "create chunks", "major sections", "logical chunks"
        ]) and any(char.isdigit() for char in user_instruction)

        if is_logical_chunking:
            return """You are an expert document analysis tool. Your task is to split a document into logical chunks based on the user's instruction.

**User Instruction:**
{user_instruction}

**CRITICAL CROSS-FILE BOUNDARY PROCESSING:**
This content comes from a multi-part document where files have overlapping sections. The overlap content (if present) shows what came before this section.

**BOUNDARY-AWARE PROCESSING RULES:**
- Use the overlap content to understand the full context and semantic flow
- Only create splits where semantic units are complete within this window
- If a topic/section begins in overlap but continues here, include it in the first chunk
- If a topic/section begins here but might continue in the next file, prefer to keep it whole rather than split at artificial boundaries
- Prioritize semantic coherence over strict chunk size limits
- Consider the document's logical structure (sections, topics, concepts) over file boundaries

**CONTENT STRUCTURE:**
[OVERLAP CONTEXT - Previous section content for continuity]
{overlap_content}

[MAIN CONTENT - Current section to analyze and chunk]
{main_content}

**Output Requirements:**
- Analyze the combined content and create logical chunks according to the instruction
- Return the chunks as a JSON array of strings
- Each chunk should be a coherent, self-contained section
- Ensure chunks flow logically and maintain semantic coherence across boundaries
- Do not include chunk numbers or headers in the output, just the chunk content
- Skip any splits that would create incomplete semantic units

**Example Output Format:**
["First chunk content here...", "Second chunk content here...", "Third chunk content here..."]
"""
        else:
            # Boundary-based chunking
            return """You are an expert document analysis tool. Your task is to split a document into logical chunks based on a user's instruction. You will be given an instruction and the document text. You must identify the exact headings or titles that mark the beginning of a new chunk according to the instruction.

**CRITICAL: Preserve exact markdown formatting**
- Return headings exactly as they appear in the text, including # symbols and all markdown formatting
- Example: If text has "# PUTTING IT ALL TOGETHER", return exactly "# PUTTING IT ALL TOGETHER"
- Do not modify, clean, or reformat the headings in any way

**User Instruction:**
{user_instruction}

**CRITICAL CROSS-FILE BOUNDARY PROCESSING:**
This content comes from a multi-part document where files have overlapping sections. The overlap content (if present) shows what came before this section.

**BOUNDARY-AWARE PROCESSING RULES:**
- Use the overlap content to understand the full context and semantic flow
- Only identify splits where semantic units are complete within this window
- If a topic/section begins in overlap but continues here, consider it part of the current chunk
- If a topic/section begins here but might continue in the next file, prefer to keep it whole rather than split at artificial boundaries
- Prioritize semantic coherence over strict chunk size limits
- Consider the document's logical structure (sections, topics, concepts) over file boundaries

**CONTENT STRUCTURE:**
[OVERLAP CONTEXT - Previous section content for continuity]
{overlap_content}

[MAIN CONTENT - Current section to analyze and chunk]
{main_content}

**Output Requirements:**
- Return ONLY a comma-separated list of the exact heading strings with original markdown formatting
- Only include headings where the complete semantic unit (from heading to next heading) is present in the text
- Consider the overlap context when determining semantic boundaries
- Do not include any other text, explanations, or formatting.

**Example:**
If the instruction is "Split by chapter" and the text contains "# Chapter 1" and "# Chapter 2", your output should be:
# Chapter 1,# Chapter 2
"""

    def _process_file_with_overlap(
        self,
        file_content: str,
        strategy,
        is_last_file: bool = False
    ) -> List[str]:
        """
        Process a file with simplified cross-file chunking logic.

        Universal rules for all strategies:
        1. Create chunks in file based on strategy
        2. Don't create a chunk unless you can find an end boundary to the chunk
        3. If a chunk could not be completed, forward the content to next file

        Args:
            file_content: Main content of the current file
            strategy: Chunking strategy to use

        Returns:
            List of completed chunks
        """
        # STEP 1: Prepend content forwarded from previous file (if any)
        content_to_chunk = file_content
        if self.state.last_incomplete_chunk:
            # Add space between forwarded content and new content if needed
            separator = ""
            if (self.state.last_incomplete_chunk and
                file_content and
                not self.state.last_incomplete_chunk.endswith((' ', '\n', '\t')) and
                not file_content.startswith((' ', '\n', '\t'))):
                separator = " "
            content_to_chunk = self.state.last_incomplete_chunk + separator + file_content
            self.logger.debug(f"Prepended forwarded content (length: {len(self.state.last_incomplete_chunk)}) to current file")
            self.state.last_incomplete_chunk = None  # Clear after use

        # STEP 2: Chunk the content using the strategy
        from .engine import chunk_document
        all_chunks = chunk_document(content_to_chunk, strategy, None)

        # STEP 3: Check if chunking couldn't complete (content remains at end)
        completed_chunks = []
        if all_chunks:
            # If we have chunks, check if the last one represents unfinished content
            last_chunk = all_chunks[-1]
            if not is_last_file and self._has_unfinished_content(last_chunk, content_to_chunk, strategy):
                # Forward unfinished content to next file (only if not the last file)
                self.state.last_incomplete_chunk = last_chunk
                self.logger.debug(f"Forwarded unfinished content (length: {len(last_chunk)}) to next file")
                completed_chunks = all_chunks[:-1]  # Return all but the unfinished chunk
            else:
                # All content was successfully chunked OR this is the last file (accept all chunks)
                completed_chunks = all_chunks
                if is_last_file:
                    self.logger.debug("Last file processed - accepting all chunks including potentially incomplete ones")
                else:
                    self.logger.debug("All content successfully chunked")
        else:
            # No chunks created - this shouldn't happen with valid content
            completed_chunks = []

        return completed_chunks


    def _process_file_with_strategy(self, content: str, strategy) -> List[str]:
        """Process content using the chunking engine."""
        from .engine import chunk_document
        return chunk_document(content, strategy, None)

    def _has_unfinished_content(self, last_chunk: str, original_content: str, strategy) -> bool:
        """
        Determine if chunking couldn't complete and content should be forwarded.

        Universal logic for all strategies:
        - If the last chunk represents unfinished content at the end of the input, forward it
        - This happens when the chunking strategy couldn't find a proper boundary to end the chunk
        """
        # Strategy-specific logic for determining incomplete chunks

        if strategy.strategy == "character":
            # For character chunking, if the last chunk is significantly shorter than chunk_size,
            # it likely hit a file boundary and should be forwarded
            chunk_size = getattr(strategy, 'chunk_size', 1000)
            # If last chunk is less than 80% of expected chunk size, consider it incomplete
            if len(last_chunk) < (chunk_size * 0.8):
                self.logger.debug(f"Character chunk incomplete: {len(last_chunk)} < {chunk_size * 0.8} chars, forwarding to next file")
                return True

        elif strategy.strategy == "token":
            # For token chunking, use actual token counting
            chunk_size = getattr(strategy, 'chunk_size', 512)
            model = getattr(strategy, 'model', 'cl100k_base')

            try:
                import tiktoken
                # Get the tokenizer
                try:
                    encoding = tiktoken.get_encoding(model)
                except KeyError:
                    # Fallback to common encodings
                    if "cl100k" in model:
                        encoding = tiktoken.get_encoding("cl100k_base")
                    elif "p50k" in model:
                        encoding = tiktoken.get_encoding("p50k_base")
                    elif "r50k" in model:
                        encoding = tiktoken.get_encoding("r50k_base")
                    else:
                        encoding = tiktoken.get_encoding("cl100k_base")  # Default to GPT-3.5/4 tokenizer

                # Count actual tokens in the last chunk
                tokens_in_chunk = len(encoding.encode(last_chunk))

                # If significantly under expected size, forward
                if tokens_in_chunk < (chunk_size * 0.8):
                    self.logger.debug(f"Token chunk incomplete: {tokens_in_chunk} tokens < {chunk_size * 0.8} expected, forwarding to next file")
                    return True
            except ImportError:
                # Fallback to character-based estimation if tiktoken not available
                estimated_chars = len(last_chunk)
                if estimated_chars < (chunk_size * 4 * 0.8):  # ~4 chars per token
                    self.logger.debug(f"Token chunk incomplete (fallback): ~{estimated_chars//4} tokens < {chunk_size * 0.8} expected, forwarding to next file")
                    return True

        elif strategy.strategy in ["llm_prompt", "semantic"]:
            # For semantic strategies, check if it ends with proper punctuation or is too short
            if len(last_chunk) < 200:  # Minimum semantic chunk size
                self.logger.debug(f"Semantic chunk too short ({len(last_chunk)} chars), forwarding to next file")
                return True
            if not (last_chunk.endswith('.') or last_chunk.endswith('!') or last_chunk.endswith('?') or last_chunk.endswith('"')):
                self.logger.debug("Semantic chunk doesn't end with proper punctuation, forwarding to next file")
                return True

        elif strategy.strategy == "schema":
            # For schema chunking, check if chunk starts with a schema pattern but doesn't have additional patterns
            # This indicates an incomplete section that should be forwarded to the next file
            try:
                schema = json.loads(strategy.json_schema)
                rules = schema.get("rules", [])

                starts_with_pattern = False
                has_additional_patterns = False

                for rule in rules:
                    rule_type = rule.get("type")
                    value = rule.get("value")

                    if rule_type == "pattern":
                        try:
                            # Check if chunk starts with this pattern
                            if re.match(value, last_chunk.strip()):
                                starts_with_pattern = True

                            # Check if chunk has additional matches (beyond the first)
                            matches = list(re.finditer(value, last_chunk))
                            if len(matches) > 1:
                                has_additional_patterns = True
                                break
                        except re.error:
                            continue

                # If chunk starts with pattern but has no additional patterns, it's incomplete
                if starts_with_pattern and not has_additional_patterns:
                    self.logger.debug(f"Schema chunk incomplete: starts with pattern but no additional patterns, forwarding to next file")
                    return True

            except (json.JSONDecodeError, KeyError):
                # If schema parsing fails, use length-based heuristic
                if len(last_chunk) < 300:
                    self.logger.debug(f"Schema chunk incomplete (fallback): {len(last_chunk)} chars < 300, forwarding to next file")
                    return True

            return False

        elif strategy.strategy == "delimiter":
            # For delimiter chunking, if the last chunk doesn't end with the delimiter,
            # it hit a file boundary and should be forwarded
            delimiter = getattr(strategy, 'delimiter', '\n\n')
            if not last_chunk.endswith(delimiter):
                self.logger.debug(f"Delimiter chunk doesn't end with '{delimiter}', forwarding to next file")
                return True

        # Default: assume chunking was successful
        return False

    def get_chunking_stats(self) -> Dict[str, Any]:
        """Get statistics about the current chunking operation."""
        return {
            "total_chunks_created": self.state.total_chunks_created,
            "current_file_index": self.state.current_file_index,
            "has_incomplete_chunk": self.state.last_incomplete_chunk is not None,
            "incomplete_chunk_length": len(self.state.last_incomplete_chunk) if self.state.last_incomplete_chunk else 0
        }