"""
Content processor for taxonomy generation.
"""

from typing import List


class ContentProcessor:
    """
    Processor for preparing content samples for taxonomy generation.
    """

    @staticmethod
    def prepare_content_sample(chunks: List[str], max_chars: int = 8000) -> str:
        """
        Prepare content from complete chunks in sequential order.
        Takes all available chunks up to a reasonable limit for token constraints.

        Args:
            chunks: All available chunks (already batched at generator level)
            max_chars: Maximum characters to include (may be exceeded for complete chunks)

        Returns:
            Concatenated content from complete chunks in order
        """
        if not chunks:
            return ""

        # Take complete chunks in sequential order - no sorting or prioritization
        # Use up to 10 complete chunks to ensure adequate content coverage
        # (batching is handled at the generator level)
        sample_parts = []
        total_chars = 0

        # Take chunks in the order they appear (up to 10 chunks for reasonable token limits)
        for chunk in chunks[:10]:
            sample_parts.append(chunk)
            total_chars += len(chunk)

        return "\n\n---\n\n".join(sample_parts)