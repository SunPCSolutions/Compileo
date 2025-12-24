"""
Intelligent chunk sampling for taxonomy generation.

This module provides algorithms for sampling representative chunks from
large document collections for efficient taxonomy generation.
"""

import math
import random
from typing import List, Set, Dict, Any
from collections import Counter


class ChunkSampler:
    """
    Provides intelligent sampling algorithms for document chunks.
    """

    @staticmethod
    def sample_chunks(chunks: List[str], sample_size: int = 100) -> List[str]:
        """
        Sample representative chunks using intelligent algorithms.

        Args:
            chunks: List of all available chunks
            sample_size: Desired number of chunks to sample

        Returns:
            List of sampled chunks
        """
        if len(chunks) <= sample_size:
            return chunks.copy()

        # Apply multiple sampling strategies and combine results
        strategies = [
            ChunkSampler._stratified_sampling,
            ChunkSampler._diversity_sampling,
            ChunkSampler._information_density_sampling
        ]

        sampled_sets = []
        chunks_per_strategy = max(1, sample_size // len(strategies))

        for strategy in strategies:
            try:
                sampled = strategy(chunks, chunks_per_strategy)
                sampled_sets.append(set(sampled))
            except Exception:
                # Fallback to random sampling if strategy fails
                sampled_sets.append(set(random.sample(chunks, min(chunks_per_strategy, len(chunks)))))

        # Combine and deduplicate
        combined_sample = set()
        for sample_set in sampled_sets:
            combined_sample.update(sample_set)

        # If we have too many, trim to sample_size
        if len(combined_sample) > sample_size:
            combined_sample = set(random.sample(list(combined_sample), sample_size))

        # If we have too few, add random chunks
        while len(combined_sample) < sample_size and len(combined_sample) < len(chunks):
            remaining = [c for c in chunks if c not in combined_sample]
            if not remaining:
                break
            combined_sample.add(random.choice(remaining))

        return list(combined_sample)

    @staticmethod
    def _stratified_sampling(chunks: List[str], sample_size: int) -> List[str]:
        """
        Stratified sampling based on chunk position and length.

        Args:
            chunks: All chunks
            sample_size: Number to sample

        Returns:
            Stratified sample
        """
        if len(chunks) <= sample_size:
            return chunks.copy()

        # Divide chunks into strata based on position
        total_chunks = len(chunks)
        samples_per_stratum = max(1, sample_size // 3)  # 3 strata

        # Stratum 1: Beginning of documents (first 20%)
        start_idx = 0
        end_idx = max(1, int(total_chunks * 0.2))
        stratum1 = chunks[start_idx:end_idx]
        sample1 = random.sample(stratum1, min(samples_per_stratum, len(stratum1)))

        # Stratum 2: Middle section (middle 60%)
        start_idx = max(0, int(total_chunks * 0.2))
        end_idx = min(total_chunks, int(total_chunks * 0.8))
        stratum2 = chunks[start_idx:end_idx]
        sample2 = random.sample(stratum2, min(samples_per_stratum, len(stratum2)))

        # Stratum 3: End of documents (last 20%)
        start_idx = max(0, int(total_chunks * 0.8))
        end_idx = total_chunks
        stratum3 = chunks[start_idx:end_idx]
        sample3 = random.sample(stratum3, min(samples_per_stratum, len(stratum3)))

        return sample1 + sample2 + sample3

    @staticmethod
    def _diversity_sampling(chunks: List[str], sample_size: int) -> List[str]:
        """
        Diversity sampling to maximize content variety.

        Args:
            chunks: All chunks
            sample_size: Number to sample

        Returns:
            Diverse sample
        """
        if len(chunks) <= sample_size:
            return chunks.copy()

        # Calculate similarity scores between chunks
        similarities = ChunkSampler._calculate_chunk_similarities(chunks)

        # Use greedy selection to maximize diversity
        selected = [random.choice(chunks)]  # Start with random chunk
        candidates = [c for c in chunks if c != selected[0]]

        while len(selected) < sample_size and candidates:
            # Find candidate with maximum minimum similarity to selected chunks
            best_candidate = None
            best_score = -1

            for candidate in candidates[:50]:  # Limit search for performance
                min_similarity = min(
                    similarities.get((min(c1, candidate), max(c1, candidate)), 0)
                    for c1 in selected
                )
                if min_similarity > best_score:
                    best_score = min_similarity
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
                candidates.remove(best_candidate)
            else:
                # Fallback to random selection
                selected.append(random.choice(candidates))
                candidates.remove(selected[-1])

        return selected

    @staticmethod
    def _information_density_sampling(chunks: List[str], sample_size: int) -> List[str]:
        """
        Sample chunks with highest information density.

        Args:
            chunks: All chunks
            sample_size: Number to sample

        Returns:
            High-information chunks
        """
        if len(chunks) <= sample_size:
            return chunks.copy()

        # Calculate information density scores
        scored_chunks = []
        for chunk in chunks:
            score = ChunkSampler._calculate_information_density(chunk)
            scored_chunks.append((chunk, score))

        # Sort by score (descending) and take top samples
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in scored_chunks[:sample_size]]

    @staticmethod
    def _calculate_chunk_similarities(chunks: List[str]) -> Dict[tuple, float]:
        """
        Calculate Jaccard similarity between chunks based on word overlap.

        Args:
            chunks: List of text chunks

        Returns:
            Dictionary mapping chunk pairs to similarity scores
        """
        similarities = {}
        chunk_sets = []

        # Convert chunks to word sets
        for chunk in chunks:
            words = set(chunk.lower().split())
            # Remove common stop words for better similarity
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            words = words - stop_words
            chunk_sets.append(words)

        # Calculate similarities for reasonable number of pairs
        max_pairs = min(1000, len(chunks) * (len(chunks) - 1) // 2)

        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                if len(similarities) >= max_pairs:
                    break

                set1, set2 = chunk_sets[i], chunk_sets[j]
                if set1 and set2:
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    similarity = intersection / union if union > 0 else 0
                    similarities[(chunks[i], chunks[j])] = similarity

        return similarities

    @staticmethod
    def _calculate_information_density(chunk: str) -> float:
        """
        Calculate information density score for a chunk.

        Considers factors like:
        - Length (longer chunks generally have more information)
        - Word diversity (unique words / total words)
        - Term frequency distribution
        - Presence of technical/specialized terms

        Args:
            chunk: Text chunk to score

        Returns:
            Information density score (higher = more informative)
        """
        if not chunk or not chunk.strip():
            return 0.0

        words = chunk.split()
        if not words:
            return 0.0

        # Basic metrics
        length_score = min(1.0, len(chunk) / 1000)  # Normalize length
        word_count = len(words)
        unique_words = set(words)
        diversity_score = len(unique_words) / word_count if word_count > 0 else 0

        # Term frequency analysis (detect information-rich patterns)
        word_freq = Counter(words)
        most_common_freq = word_freq.most_common(1)[0][1] if word_freq else 0
        concentration_score = 1.0 - (most_common_freq / word_count)  # Lower concentration = more diverse

        # Detect technical/specialized content
        technical_indicators = [
            'diagnosis', 'treatment', 'patient', 'clinical', 'medical',
            'algorithm', 'system', 'process', 'method', 'analysis',
            'research', 'study', 'data', 'model', 'framework'
        ]

        technical_score = sum(1 for word in words if word.lower() in technical_indicators) / word_count

        # Combine scores with weights
        final_score = (
            length_score * 0.3 +
            diversity_score * 0.3 +
            concentration_score * 0.2 +
            technical_score * 0.2
        )

        return final_score

    @staticmethod
    def analyze_chunk_distribution(chunks: List[str]) -> Dict[str, Any]:
        """
        Analyze the distribution of chunks for sampling insights.

        Args:
            chunks: List of chunks to analyze

        Returns:
            Analysis results
        """
        lengths = [len(chunk) for chunk in chunks]
        word_counts = [len(chunk.split()) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_length": sum(lengths) / len(lengths) if lengths else 0,
            "avg_words": sum(word_counts) / len(word_counts) if word_counts else 0,
            "length_distribution": {
                "min": min(lengths) if lengths else 0,
                "max": max(lengths) if lengths else 0,
                "median": sorted(lengths)[len(lengths)//2] if lengths else 0
            },
            "word_distribution": {
                "min": min(word_counts) if word_counts else 0,
                "max": max(word_counts) if word_counts else 0,
                "median": sorted(word_counts)[len(word_counts)//2] if word_counts else 0
            }
        }