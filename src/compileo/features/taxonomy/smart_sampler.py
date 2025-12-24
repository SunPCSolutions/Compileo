"""
Smart Chunk Sampler

This module provides functionality to sample document chunks intelligently
for initial taxonomy generation, selecting representative chunks from the
beginning, middle, and end of the document.
"""

from typing import List
import math


class SmartChunkSampler:
    """Service for selecting representative chunks from a document."""

    @staticmethod
    def sample(chunks: List[str], sample_size: int = 10) -> List[str]:
        """
        Select a representative sample of chunks from the document.

        The sampling strategy ensures coverage across the entire document
        by picking chunks from the beginning, middle, and end, rather than
        just the first N chunks.

        Args:
            chunks: List of text chunks from the document
            sample_size: Number of chunks to select (default: 10)

        Returns:
            List of selected chunks
        """
        if not chunks:
            return []

        total_chunks = len(chunks)

        # If we have fewer chunks than the sample size, return all chunks
        if total_chunks <= sample_size:
            return chunks

        # Always include the first and last chunk
        selected_indices = {0, total_chunks - 1}

        # Calculate how many more chunks we need
        remaining_slots = sample_size - len(selected_indices)

        if remaining_slots > 0:
            # We want to pick 'remaining_slots' chunks from the range (0, total_chunks - 1)
            # We use equal spacing to distribute them
            step = (total_chunks - 1) / (remaining_slots + 1)
            
            for i in range(1, remaining_slots + 1):
                index = int(round(i * step))
                # Ensure we don't pick 0 or total_chunks-1 again
                if index > 0 and index < total_chunks - 1:
                    selected_indices.add(index)
        
        # If we still don't have enough (due to rounding collisions), fill gaps
        # This is a fallback and shouldn't happen often with reasonable inputs
        current_index = 1
        while len(selected_indices) < sample_size and current_index < total_chunks - 1:
            if current_index not in selected_indices:
                selected_indices.add(current_index)
            current_index += 1

        # Sort indices to maintain document order
        sorted_indices = sorted(list(selected_indices))
        
        return [chunks[i] for i in sorted_indices]