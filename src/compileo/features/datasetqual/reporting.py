"""
Quality report generation and formatting.

Provides various output formats for quality analysis results.
"""

import json
from typing import Any, Dict
from pathlib import Path


class QualityReporter:
    """
    Generates formatted reports from quality analysis results.
    """

    @staticmethod
    def format_json(results: Dict[str, Any], pretty: bool = True) -> str:
        """
        Format results as JSON.

        Args:
            results: Quality analysis results
            pretty: Whether to pretty-print JSON

        Returns:
            JSON formatted string
        """
        if pretty:
            return json.dumps(results, indent=2, default=str)
        return json.dumps(results, default=str)

    @staticmethod
    def format_text(results: Dict[str, Any]) -> str:
        """
        Format results as human-readable text.

        Args:
            results: Quality analysis results

        Returns:
            Formatted text report
        """
        lines = []

        # Header
        lines.append("=" * 50)
        lines.append("DATASET QUALITY ANALYSIS REPORT")
        lines.append("=" * 50)

        if not results.get('enabled', True):
            lines.append("Quality analysis is DISABLED")
            return "\n".join(lines)

        # Summary
        summary = results.get('summary', {})
        lines.append(f"Dataset Size: {results.get('dataset_size', 'N/A')}")
        lines.append(f"Overall Score: {summary.get('overall_score', 'N/A')}")
        lines.append(f"Status: {'PASSED' if summary.get('passed', False) else 'FAILED'}")
        lines.append("")

        # Metric results
        lines.append("METRIC RESULTS:")
        lines.append("-" * 30)

        for metric_name, metric_data in results.get('results', {}).items():
            score = metric_data.get('score', 'N/A')
            threshold = metric_data.get('threshold')
            passed = metric_data.get('passed')

            status = "PASS" if passed else "FAIL" if passed is False else "N/A"
            threshold_str = f" (threshold: {threshold})" if threshold is not None else ""

            lines.append(f"{metric_name.capitalize()}: {score}{threshold_str} - {status}")

            # Show issues if any
            if 'issues' in metric_data.get('details', {}):
                for issue in metric_data['details']['issues'][:3]:  # Limit to 3 issues
                    lines.append(f"  - {issue}")

        lines.append("")

        # Issues summary
        issues = summary.get('issues', [])
        if issues:
            lines.append("ISSUES FOUND:")
            lines.append("-" * 30)
            for issue in issues[:5]:  # Limit to 5 issues
                lines.append(f"- {issue}")
        else:
            lines.append("No issues detected.")

        return "\n".join(lines)

    @staticmethod
    def format_markdown(results: Dict[str, Any]) -> str:
        """
        Format results as Markdown.

        Args:
            results: Quality analysis results

        Returns:
            Markdown formatted string
        """
        lines = []

        # Header
        lines.append("# Dataset Quality Analysis Report")
        lines.append("")

        if not results.get('enabled', True):
            lines.append("**Quality analysis is DISABLED**")
            return "\n".join(lines)

        # Summary
        summary = results.get('summary', {})
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Dataset Size:** {results.get('dataset_size', 'N/A')}")
        lines.append(f"- **Overall Score:** {summary.get('overall_score', 'N/A')}")
        lines.append(f"- **Status:** {'✅ PASSED' if summary.get('passed', False) else '❌ FAILED'}")
        lines.append("")

        # Metric results
        lines.append("## Metric Results")
        lines.append("")

        for metric_name, metric_data in results.get('results', {}).items():
            score = metric_data.get('score', 'N/A')
            threshold = metric_data.get('threshold')
            passed = metric_data.get('passed')

            status = "✅ PASS" if passed else "❌ FAIL" if passed is False else "⚪ N/A"
            threshold_str = f" (threshold: {threshold})" if threshold is not None else ""

            lines.append(f"### {metric_name.capitalize()}")
            lines.append(f"**Score:** {score}{threshold_str}")
            lines.append(f"**Status:** {status}")
            lines.append("")

            # Show details
            details = metric_data.get('details', {})
            if details:
                lines.append("**Details:**")
                for key, value in details.items():
                    if key != 'issues':  # Handle issues separately
                        lines.append(f"- {key}: {value}")
                lines.append("")

            # Show issues if any
            if 'issues' in details:
                lines.append("**Issues:**")
                for issue in details['issues'][:3]:
                    lines.append(f"- {issue}")
                lines.append("")

        # Issues summary
        issues = summary.get('issues', [])
        if issues:
            lines.append("## Issues Summary")
            lines.append("")
            for issue in issues[:10]:
                lines.append(f"- {issue}")

        return "\n".join(lines)

    @staticmethod
    def save_report(results: Dict[str, Any], output_path: str,
                   format_type: str = "json") -> None:
        """
        Save quality report to file.

        Args:
            results: Quality analysis results
            output_path: Path to save the report
            format_type: Format type ('json', 'text', 'markdown')
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format_type == "json":
            content = QualityReporter.format_json(results)
        elif format_type == "text":
            content = QualityReporter.format_text(results)
        elif format_type == "markdown":
            content = QualityReporter.format_markdown(results)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

    @staticmethod
    def get_report(results: Dict[str, Any], format_type: str = "json") -> str:
        """
        Get formatted quality report.

        Args:
            results: Quality analysis results
            format_type: Format type ('json', 'text', 'markdown')

        Returns:
            Formatted report string
        """
        if format_type == "json":
            return QualityReporter.format_json(results)
        elif format_type == "text":
            return QualityReporter.format_text(results)
        elif format_type == "markdown":
            return QualityReporter.format_markdown(results)
        else:
            raise ValueError(f"Unsupported format: {format_type}")