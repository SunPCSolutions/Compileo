import json
from typing import List, Dict, Any, Tuple
from .engine import chunk_document
from .schema import ChunkingStrategy
from ...core.logging import get_logger

logger = get_logger(__name__)

class SlidingWindowChunker:
    """Processes parsed document chunks with semantic overlap context"""
    def __init__(self):
        pass  # No parameters needed for current semantic chunking architecture

    def chunk_large_document(self, markdown_files: List[str], strategy: ChunkingStrategy) -> List[str]:
        """Main entry point for sliding window chunking"""
        from .schema import LLMPromptStrategy

        # Load the JSON files
        json_chunks = self._load_json_chunks(markdown_files)

        # Process the chunks using the structured sliding window approach
        final_chunks = self._chunk_with_structured_sliding_window(json_chunks, strategy)

        return final_chunks

    # Legacy methods removed - current architecture uses structured sliding window approach
    # that processes parsed JSON chunks with semantic overlap context

    def _load_json_chunks(self, json_files: List[str]) -> List[Dict[str, Any]]:
        """Load and parse JSON chunk files"""
        chunks = []
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                    chunks.append(chunk_data)
            except Exception as e:
                logger.warning(f"Failed to load JSON chunk file {file_path}: {e}")
                continue
        return chunks


    def _chunk_with_structured_sliding_window(self, chunks: List[Dict[str, Any]], strategy: ChunkingStrategy) -> List[str]:
        """Process chunks using enhanced structured sliding window approach with boundary-aware chunking"""
        all_chunks = []

        # Process each chunk with enhanced overlap context for semantic coherence
        for i, chunk_data in enumerate(chunks):
            main_content = chunk_data.get('main_content', '')
            overlap_content = chunk_data.get('overlap_content', '')

            # Enhanced context integration: Use overlap content as semantic context for boundary-aware chunking
            # This allows the AI to understand semantic flow across file boundaries
            if overlap_content and main_content:
                # Create enhanced prompt context that provides overlap for continuity and main content for chunking
                enhanced_content = f"""[OVERLAP CONTEXT - Previous section content for continuity]
{overlap_content}

[MAIN CONTENT - Current section to analyze and chunk]
{main_content}""".strip()
            else:
                # Fallback for cases without overlap
                enhanced_content = main_content or overlap_content

            if enhanced_content:
                # Process this window of text with the chunking engine
                # The enhanced prompts now include boundary-aware instructions
                window_chunks = chunk_document(enhanced_content, strategy)
                all_chunks.extend(window_chunks)

        # Enhanced post-processing: Remove any empty chunks and deduplicate
        seen_chunks = set()
        final_chunks = []
        for chunk in all_chunks:
            chunk_text = chunk.strip()
            # Remove chunks that are just context markers or overlap content
            if (chunk_text and
                chunk_text not in seen_chunks and
                not chunk_text.startswith('[OVERLAP CONTEXT') and
                not chunk_text.startswith('[SEMANTIC CONTEXT') and
                not chunk_text.startswith('[MAIN CONTENT')):
                final_chunks.append(chunk_text)
                seen_chunks.add(chunk_text)

        return final_chunks