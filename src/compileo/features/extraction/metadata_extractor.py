"""
Document metadata extraction utilities.

This module provides functions to automatically extract metadata from parsed documents,
including title, summary, author, and other relevant information.
"""

import re
from typing import Optional, List
from .context_models import DocumentContext


class MetadataExtractor:
    """Extracts metadata from document content and filenames."""

    @staticmethod
    def extract_document_metadata(content: str, filename: str) -> DocumentContext:
        """
        Extract metadata from document content and filename.

        Args:
            content: Parsed document content
            filename: Original filename

        Returns:
            DocumentContext with extracted metadata
        """
        # Extract title from filename or content
        title = MetadataExtractor._extract_title(content, filename)

        # Extract author if present
        author = MetadataExtractor._extract_author(content)

        # Extract publication date if present
        publication_date = MetadataExtractor._extract_publication_date(content)

        # Extract keywords/tags
        keywords = MetadataExtractor._extract_keywords(content)

        # Generate summary (placeholder - could use LLM in future)
        summary = MetadataExtractor._generate_summary(content, title)

        # Extract source information
        source = MetadataExtractor._extract_source(content, filename)

        return DocumentContext(
            title=title,
            summary=summary,
            author=author,
            publication_date=publication_date,
            source=source,
            keywords=keywords
        )

    @staticmethod
    def _extract_title(content: str, filename: str) -> Optional[str]:
        """Extract title from content or filename."""
        # Try to find title in content (common patterns)
        title_patterns = [
            r'^#\s+(.+)$',  # Markdown heading
            r'^(.+)\n={3,}',  # Underlined title
            r'^(.+)\n-{3,}',  # Underlined subtitle
            r'Title:\s*(.+)$',  # Explicit title field
        ]

        for pattern in title_patterns:
            match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                if len(title) > 5:  # Avoid very short titles
                    return title

        # Fallback to filename (remove extension and clean up)
        base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
        # Replace underscores and hyphens with spaces
        clean_name = re.sub(r'[_-]', ' ', base_name)
        # Title case
        return clean_name.title()

    @staticmethod
    def _extract_author(content: str) -> Optional[str]:
        """Extract author information from content."""
        author_patterns = [
            r'Author:\s*(.+)$',
            r'By:\s*(.+)$',
            r'Written by:\s*(.+)$',
            r'^(.+)\n.*(?:author|by|written)',
        ]

        for pattern in author_patterns:
            match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
            if match:
                author = match.group(1).strip()
                # Clean up common artifacts
                author = re.sub(r'[^\w\s,-]', '', author).strip()
                if len(author) > 2 and len(author) < 100:  # Reasonable length
                    return author

        return None

    @staticmethod
    def _extract_publication_date(content: str) -> Optional[str]:
        """Extract publication date from content."""
        date_patterns = [
            r'Date:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'Published:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})',  # General date pattern
            r'(\d{4}-\d{2}-\d{2})',  # ISO date
        ]

        for pattern in date_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                # Basic validation
                if re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', date_str) or \
                   re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                    return date_str

        return None

    @staticmethod
    def _extract_keywords(content: str) -> List[str]:
        """Extract keywords or tags from content."""
        keywords = []

        # Look for explicit keywords section
        keywords_match = re.search(r'Keywords?:\s*(.+)$', content, re.MULTILINE | re.IGNORECASE)
        if keywords_match:
            keywords_text = keywords_match.group(1)
            # Split by common delimiters
            keywords = re.split(r'[;,]', keywords_text)
            keywords = [k.strip() for k in keywords if k.strip()]

        # If no explicit keywords, extract from headings and emphasis
        if not keywords:
            # Extract from markdown headings
            heading_matches = re.findall(r'^#{1,3}\s+(.+)$', content, re.MULTILINE)
            keywords.extend([h.strip() for h in heading_matches if len(h.strip()) > 3])

            # Extract bold/italic text (potential keywords)
            bold_matches = re.findall(r'\*\*(.+?)\*\*', content)
            keywords.extend([b.strip() for b in bold_matches if len(b.strip()) > 3])

        # Remove duplicates and limit to top 10
        seen = set()
        unique_keywords = []
        for kw in keywords:
            lower_kw = kw.lower()
            if lower_kw not in seen and len(kw) > 2:
                unique_keywords.append(kw)
                seen.add(lower_kw)

        return unique_keywords[:10]  # Limit to 10 keywords

    @staticmethod
    def _generate_summary(content: str, title: Optional[str] = None) -> Optional[str]:
        """Generate a brief summary of the document."""
        # Simple extractive summary - first few sentences
        sentences = re.split(r'[.!?]+', content.strip())

        # Filter out very short sentences and clean up
        valid_sentences = [
            s.strip() for s in sentences
            if len(s.strip()) > 10 and not s.strip().startswith('#')
        ]

        if valid_sentences:
            # Take first 2-3 sentences
            summary_sentences = valid_sentences[:3]
            summary = '. '.join(summary_sentences)

            # Truncate if too long
            if len(summary) > 200:
                summary = summary[:197] + '...'

            return summary + '.' if not summary.endswith('.') else summary

        return None

    @staticmethod
    def _extract_source(content: str, filename: str) -> Optional[str]:
        """Extract source information."""
        # Look for source/citation information
        source_patterns = [
            r'Source:\s*(.+)$',
            r'From:\s*(.+)$',
            r'Citation:\s*(.+)$',
        ]

        for pattern in source_patterns:
            match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
            if match:
                source = match.group(1).strip()
                if len(source) > 3:
                    return source

        # Fallback to filename directory or just filename
        if '/' in filename or '\\' in filename:
            # Extract directory name as potential source
            path_parts = re.split(r'[/\\]', filename)
            if len(path_parts) > 1:
                return path_parts[-2]  # Parent directory

        return None