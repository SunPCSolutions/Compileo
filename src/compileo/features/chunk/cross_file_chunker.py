"""
Cross-File Chunker: Robust, scalable chunking system for multi-file documents.

This module provides a robust chunking system that handles cross-file boundaries,
maintaining state between files and ensuring complete chunks are formed even when
content spans multiple input files.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import logging
import json
import os
import uuid
from datetime import datetime

from .schema import CharacterStrategy, LLMPromptStrategy, SchemaStrategy, DelimiterStrategy, TokenStrategy

logger = logging.getLogger(__name__)


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
    """

    def __init__(self, db_connection=None):
        self.state = ChunkState()
        self.db_connection = db_connection
        self.logger = logging.getLogger(__name__)

    def reset_state(self, document_id: Optional[str] = None) -> None:
        """Reset the chunker state for a new document."""
        self.state = ChunkState()
        self.current_document_id = document_id
        self.logger.info(f"Cross-file chunker state reset for document {document_id}")

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
        owner_id: Optional[str] = None,
        chunking_options: Optional[Dict[str, Any]] = None,
        schema_definition: str = "",
        delimiters: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Chunk a multi-file document using cross-file boundary logic.
        """
        from ...storage.src.project.file_manager import FileManager
        from ...storage.src.project.database_repositories import ChunkRepository
        from ...core.settings import BackendSettings

        user_settings = BackendSettings(owner_id=owner_id)

        if manifest_data is None or not isinstance(manifest_data, dict):
            raise ValueError(f"Invalid manifest data")

        self.logger.info(f"Starting cross-file chunking for document {document_id}")

        needs_api_key = chunk_strategy in ["semantic", "llm_prompt"] or (chunk_strategy == "token" and chunker in ["gemini", "grok"])
        if needs_api_key:
            if chunker == "gemini":
                api_key = user_settings.get_setting("gemini_api_key")
                if api_key: os.environ['GOOGLE_API_KEY'] = api_key
            elif chunker == "grok":
                api_key = user_settings.get_setting("grok_api_key")
                if api_key: os.environ['GROK_API_KEY'] = api_key

        file_manager = FileManager()
        chunk_repo = ChunkRepository(owner_id)

        parsed_files = manifest_data.get("parsed_files", [])
        if not parsed_files:
            raise ValueError("No parsed files found in manifest")

        self.reset_state()
        self.logger.info(f"DEBUG_CROSS_FILE: Starting to process {len(parsed_files)} files with strategy={chunk_strategy}")

        strategy = self._create_chunking_strategy(
            chunk_strategy, chunk_size, overlap, model, manifest_data,
            chunking_options, schema_definition, delimiters,
        )
        if isinstance(strategy, SchemaStrategy):
            from .engine import chunk_document, join_schema_parts

            content = join_schema_parts(self._read_parsed_file(path) for path in parsed_files)
            batches = [chunk_document(content, strategy)]
        else:
            batches = self._chunk_file_batches(parsed_files, strategy)

        storage_root = file_manager.get_storage_base_path()
        chunk_dir = (storage_root / owner_id if owner_id else storage_root) / "chunks" / str(project_id) / str(document_id)
        for file_chunks in batches:
            for chunk in file_chunks:
                index = self.state.total_chunks_created + 1
                chunk_dir.mkdir(parents=True, exist_ok=True)
                full_path = chunk_dir / f"chunk_{index}.md"
                full_path.write_bytes(chunk.encode("utf-8"))
                chunk_repo.create_chunk({
                    "id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "owner_id": owner_id,
                    "project_id": project_id,
                    "chunk_index": index,
                    "content": chunk,
                    "metadata": {"strategy": chunk_strategy, "file_path": str(full_path.relative_to(storage_root))},
                    "created_at": datetime.utcnow().isoformat(),
                })
                self.state.total_chunks_created += 1

        self.logger.info(f"DEBUG_CROSS_FILE: Finished processing. Total chunks created: {self.state.total_chunks_created}")
        return {
            "document_id": document_id,
            "status": "completed",
            "chunks_created": self.state.total_chunks_created,
            "files_processed": len(parsed_files),
            "chunk_strategy": chunk_strategy
        }

    @staticmethod
    def _read_parsed_file(path: str) -> str:
        try:
            with open(path, encoding="utf-8") as handle:
                content = json.load(handle)["main_content"]
            if not isinstance(content, str):
                raise ValueError("main_content must be a string")
            return content
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"Cannot read parsed file {path}: {exc}") from exc

    def _chunk_file_batches(self, paths, strategy):
        for index, path in enumerate(paths):
            self.state.current_file_index = index
            try:
                content = self._read_parsed_file(path)
                if content:
                    yield self._process_file_with_overlap(content, strategy, index == len(paths) - 1)
            except Exception:
                self.logger.exception("Error processing parsed file %s", path)

    def _create_chunking_strategy(self, chunk_strategy, chunk_size, overlap, model, manifest_data, options, schema, delimiters):
        chunk_strategy = chunk_strategy.lower().strip()
        
        if chunk_strategy == "character" or "character" in chunk_strategy:
            return CharacterStrategy(strategy="character", chunk_size=chunk_size, overlap=overlap)
        elif chunk_strategy == "token" or "token" in chunk_strategy:
            return TokenStrategy(strategy="token", chunk_size=chunk_size, overlap=overlap, model="cl100k_base")
        elif chunk_strategy == "semantic" or "semantic" in chunk_strategy:
            user_instr = manifest_data.get("semantic_prompt", "Split by topic.")
            return LLMPromptStrategy(
                strategy="llm_prompt", model=model, prompt_template=self._get_prompt(user_instr),
                user_instruction=user_instr, options=options
            )
        elif chunk_strategy == "schema" or "schema" in chunk_strategy:
            return SchemaStrategy(strategy="schema", json_schema=schema or "")
        elif chunk_strategy == "delimiter" or "delimiter" in chunk_strategy:
            dels = delimiters or ["\n\n"]
            return DelimiterStrategy(strategy="delimiter", delimiter=dels[0])
        
        self.logger.warning(f"Unknown chunk strategy '{chunk_strategy}', falling back to character")
        return CharacterStrategy(strategy="character", chunk_size=chunk_size, overlap=overlap)

    def _get_prompt(self, instr):
        return f"Split document into logical chunks. Instruction: {instr}"

    def _process_file_with_overlap(self, content, strategy, is_last):
        combined = content
        if self.state.last_incomplete_chunk:
            combined = self.state.last_incomplete_chunk + "\n" + content
            self.state.last_incomplete_chunk = None

        self.logger.info(f"DEBUG_PROCESS_FILE: Combined content length: {len(combined)}")
        
        from .engine import chunk_document
        all_chunks = chunk_document(combined, strategy, None)
        
        self.logger.info(f"DEBUG_PROCESS_FILE: chunk_document returned {len(all_chunks) if all_chunks else 0} chunks")
        
        if not all_chunks:
            self.logger.warning("DEBUG_PROCESS_FILE: No chunks generated!")
            return []

        if not is_last and self._is_incomplete(all_chunks[-1], combined, strategy):
            self.state.last_incomplete_chunk = all_chunks[-1]
            self.logger.info(f"DEBUG_PROCESS_FILE: Last chunk incomplete, forwarding {len(all_chunks[-1])} chars to next file")
            return all_chunks[:-1]
        return all_chunks

    def _is_incomplete(self, chunk, content, strategy):
        """
        Universal incomplete chunk detection.
        
        A chunk is considered incomplete if it doesn't end at a natural boundary
        defined by the strategy, and thus should be forwarded to the next file.
        """
        if strategy.strategy == "character":
            return len(chunk) < (strategy.chunk_size * 0.8)
            
        if strategy.strategy == "token":
            # Rough estimate: 1 token ~= 4 characters
            # strategy.chunk_size is in tokens
            char_count = len(chunk)
            estimated_tokens = char_count / 4
            return estimated_tokens < (strategy.chunk_size * 0.8)
            
        if strategy.strategy == "delimiter":
            # Delimiter strategy always produces complete chunks
            # When you split on a delimiter, ALL resulting chunks are complete by definition
            # There's no "leftover" concept - either the delimiter was found and splits happened,
            # or no delimiter was found and entire content becomes one chunk
            return False
            
        if strategy.strategy in ["schema", "llm_prompt"]:
            # These strategies split at specific boundaries.
            # If the engine returned multiple chunks, the last one is the leftover after the last split.
            # If it returned only one chunk, the whole thing is leftover.
            # In both cases, if we are not at the last file, it should be forwarded.
            return True
            
        return False
