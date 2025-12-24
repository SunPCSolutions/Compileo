"""
Analytics for taxonomy generation.
"""

from typing import Dict, Any, List


class TaxonomyAnalytics:
    """
    Analytics generator for taxonomy data.
    """

    @staticmethod
    def generate_analytics(chunks: List[str], taxonomy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate analytics about the taxonomy and content.

        Args:
            chunks: Original chunks used for generation
            taxonomy: Generated taxonomy

        Returns:
            Analytics dictionary
        """
        # Count categories by level
        depth_counts = {}
        confidence_scores = []

        def analyze_node(node: Dict[str, Any], depth: int = 0):
            if depth not in depth_counts:
                depth_counts[depth] = 0
            depth_counts[depth] += 1

            if "confidence_threshold" in node:
                confidence_scores.append(node["confidence_threshold"])

            for child in node.get("children", []):
                analyze_node(child, depth + 1)

        analyze_node(taxonomy)

        # Calculate content statistics
        total_chars = sum(len(chunk) for chunk in chunks)
        total_words = sum(len(chunk.split()) for chunk in chunks)

        return {
            "category_distribution": {
                f"level_{depth}": count
                for depth, count in depth_counts.items()
            },
            "depth_analysis": {
                "max_depth": max(depth_counts.keys()) if depth_counts else 0,
                "total_categories": sum(depth_counts.values()),
                "avg_confidence": round(sum(confidence_scores) / len(confidence_scores), 3) if confidence_scores else 0
            },
            "content_coverage": {
                "chunks_analyzed": len(chunks),
                "total_characters": total_chars,
                "total_words": total_words,
                "avg_chunk_length": round(total_chars / len(chunks), 1) if chunks else 0
            }
        }

    @staticmethod
    def calculate_overall_confidence(taxonomy: Dict[str, Any]) -> float:
        """
        Calculate overall confidence score for the taxonomy.

        Args:
            taxonomy: Generated taxonomy

        Returns:
            Overall confidence score (0.0-1.0)
        """
        confidence_scores = []

        def collect_confidence(node: Dict[str, Any]):
            if "confidence_threshold" in node:
                confidence_scores.append(node["confidence_threshold"])
            for child in node.get("children", []):
                collect_confidence(child)

        collect_confidence(taxonomy)

        if not confidence_scores:
            return 0.7  # Default confidence when no scores provided

        # Filter out invalid scores (0.0 or negative)
        valid_scores = [score for score in confidence_scores if score > 0.0]

        if not valid_scores:
            return 0.7  # Default confidence when all scores are 0.0 or invalid

        # Weighted average favoring higher-level categories
        weights = []
        for i, score in enumerate(valid_scores):
            # Give more weight to top-level categories
            level_weight = max(1.0, 2.0 - (i * 0.1))  # Decreasing weight
            weights.append(level_weight)

        total_weight = sum(weights)
        weighted_sum = sum(score * weight for score, weight in zip(valid_scores, weights))

        result = round(weighted_sum / total_weight, 3)
        # Ensure result is never 0.0
        return max(result, 0.1)