"""
Generic Q&A Dataset Generator from Entity Relationships

This module generates question-answer pairs from inferred entity relationships.
It supports configurable templates and can work across any domain where
entities from different categories have meaningful associations.
"""

import json
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from ...core.logging import get_logger

logger = get_logger(__name__)


class QAType(Enum):
    """Types of Q&A pairs that can be generated."""
    DIRECT_ASSOCIATION = "direct_association"  # "What is associated with X?"
    CAUSAL_RELATIONSHIP = "causal_relationship"  # "What causes Y?" or "What does X cause?"
    TEMPORAL_SEQUENCE = "temporal_sequence"  # "What happens after X?"
    HIERARCHICAL_RELATIONSHIP = "hierarchical_relationship"  # "What are the parts of X?"
    COMPARATIVE_ANALYSIS = "comparative_analysis"  # "How does X compare to Y?"
    DEFINITION_EXPLANATION = "definition_explanation"  # "What is X?"
    CUSTOM_TEMPLATE = "custom_template"  # User-defined templates


@dataclass
class QAPair:
    """Represents a question-answer pair with metadata."""
    question: str
    answer: str
    context: str
    qa_type: QAType
    confidence: float
    source_relationship: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class QATemplate:
    """Template for generating Q&A pairs from relationships."""
    name: str
    qa_type: QAType
    source_categories: List[str]
    target_categories: List[str]
    question_template: str
    answer_template: str
    description: str = ""
    conditions: Optional[Dict[str, Any]] = None


class QADatasetGenerator:
    """
    Generic Q&A dataset generator that creates question-answer pairs
    from entity relationships. Designed to work across domains.
    """

    def __init__(self):
        """Initialize the Q&A dataset generator."""
        self.templates: List[QATemplate] = []
        self._load_default_templates()

    def _load_default_templates(self):
        """Load default Q&A generation templates."""
        # Direct association template
        self.add_template(QATemplate(
            name="direct_association",
            qa_type=QAType.DIRECT_ASSOCIATION,
            source_categories=[],  # Any categories
            target_categories=[],  # Any categories
            question_template="What {target_category} are associated with {source_entity}?",
            answer_template="{target_entity} is associated with {source_entity}.",
            description="Generate Q&A pairs for direct entity associations"
        ))

        # Causal relationship template
        self.add_template(QATemplate(
            name="causal_relationship",
            qa_type=QAType.CAUSAL_RELATIONSHIP,
            source_categories=[],  # Any categories
            target_categories=[],  # Any categories
            question_template="What {target_category} can result from {source_entity}?",
            answer_template="{source_entity} can cause or lead to {target_entity}.",
            description="Generate Q&A pairs for causal relationships"
        ))

        # Definition/explanation template
        self.add_template(QATemplate(
            name="definition_explanation",
            qa_type=QAType.DEFINITION_EXPLANATION,
            source_categories=[],  # Any categories
            target_categories=[],  # Any categories
            question_template="What is {source_entity}?",
            answer_template="{source_entity} is associated with {target_entity}.",
            description="Generate explanatory Q&A pairs"
        ))

    def add_template(self, template: QATemplate):
        """Add a custom Q&A generation template."""
        self.templates.append(template)

    def remove_template(self, template_name: str):
        """Remove a Q&A generation template."""
        self.templates = [t for t in self.templates if t.name != template_name]

    def generate_qa_pairs(self, relationships: List[Dict[str, Any]],
                         max_pairs_per_relationship: int = 3) -> List[QAPair]:
        """
        Generate Q&A pairs from entity relationships.

        Args:
            relationships: List of entity relationship dictionaries
            max_pairs_per_relationship: Maximum Q&A pairs per relationship

        Returns:
            List of generated Q&A pairs
        """
        qa_pairs = []

        for relationship in relationships:
            relationship_pairs = self._generate_from_relationship(
                relationship, max_pairs_per_relationship
            )
            qa_pairs.extend(relationship_pairs)

        # Remove duplicates and sort by confidence
        unique_pairs = self._deduplicate_qa_pairs(qa_pairs)

        return sorted(unique_pairs, key=lambda x: x.confidence, reverse=True)

    def _generate_from_relationship(self, relationship: Dict[str, Any],
                                  max_pairs: int) -> List[QAPair]:
        """Generate Q&A pairs from a single relationship."""
        pairs = []

        # Find applicable templates
        applicable_templates = self._find_applicable_templates(relationship)

        for template in applicable_templates[:max_pairs]:  # Limit templates per relationship
            try:
                qa_pair = self._apply_template(template, relationship)
                if qa_pair:
                    pairs.append(qa_pair)
            except Exception as e:
                # Log error but continue with other templates
                logger.error(f"Error applying template {template.name}: {e}")
                continue

        return pairs

    def _find_applicable_templates(self, relationship: Dict[str, Any]) -> List[QATemplate]:
        """Find templates that can be applied to a relationship."""
        applicable = []

        source_cat = relationship.get('source_category', '')
        target_cat = relationship.get('target_category', '')
        rel_type = relationship.get('relationship_type', '')

        for template in self.templates:
            # Check category compatibility
            source_match = (not template.source_categories or
                          source_cat in template.source_categories)
            target_match = (not template.target_categories or
                          target_cat in template.target_categories)

            if source_match and target_match:
                applicable.append(template)

        return applicable

    def _apply_template(self, template: QATemplate, relationship: Dict[str, Any]) -> Optional[QAPair]:
        """Apply a template to generate a Q&A pair."""
        try:
            # Extract relationship data
            source_entities = relationship.get('source_entities', [])
            target_entities = relationship.get('target_entities', [])
            source_cat = relationship.get('source_category', '')
            target_cat = relationship.get('target_category', '')
            confidence = relationship.get('confidence', 0.5)

            if not source_entities or not target_entities:
                return None

            # Select representative entities (could be randomized for variety)
            source_entity = self._select_representative_entity(source_entities)
            target_entity = self._select_representative_entity(target_entities)

            # Apply templates
            question = template.question_template.format(
                source_entity=source_entity,
                target_entity=target_entity,
                source_category=source_cat,
                target_category=target_cat
            )

            answer = template.answer_template.format(
                source_entity=source_entity,
                target_entity=target_entity,
                source_category=source_cat,
                target_category=target_cat
            )

            # Generate context
            context = self._generate_context(relationship, source_entity, target_entity)

            return QAPair(
                question=question,
                answer=answer,
                context=context,
                qa_type=template.qa_type,
                confidence=confidence,
                source_relationship=relationship,
                metadata={
                    "template_name": template.name,
                    "template_description": template.description,
                    "all_source_entities": source_entities,
                    "all_target_entities": target_entities
                }
            )

        except (KeyError, ValueError) as e:
            logger.error(f"Template application error: {e}")
            return None

    def _select_representative_entity(self, entities: List[str]) -> str:
        """Select a representative entity from a list (could be enhanced with scoring)."""
        if not entities:
            return ""
        # For now, just return the first entity
        # Could be enhanced to select most representative or diverse entities
        return entities[0]

    def _generate_context(self, relationship: Dict[str, Any],
                         source_entity: str, target_entity: str) -> str:
        """Generate contextual information for the Q&A pair."""
        source_cat = relationship.get('source_category', '')
        target_cat = relationship.get('target_category', '')
        rationale = relationship.get('rationale', '')

        context_parts = [
            f"Source: {source_entity} ({source_cat})",
            f"Target: {target_entity} ({target_cat})"
        ]

        if rationale:
            context_parts.append(f"Relationship: {rationale}")

        return " | ".join(context_parts)

    def _deduplicate_qa_pairs(self, qa_pairs: List[QAPair]) -> List[QAPair]:
        """Remove duplicate Q&A pairs, keeping the highest confidence version."""
        seen = {}
        unique_pairs = []

        for pair in qa_pairs:
            key = (pair.question, pair.answer)

            if key not in seen or pair.confidence > seen[key].confidence:
                seen[key] = pair

        return list(seen.values())

    def get_generation_statistics(self, qa_pairs: List[QAPair]) -> Dict[str, Any]:
        """Generate statistics about Q&A pair generation."""
        if not qa_pairs:
            return {"total_pairs": 0}

        stats = {
            "total_pairs": len(qa_pairs),
            "qa_types": {},
            "avg_confidence": sum(p.confidence for p in qa_pairs) / len(qa_pairs),
            "unique_questions": len(set(p.question for p in qa_pairs)),
            "unique_answers": len(set(p.answer for p in qa_pairs))
        }

        for pair in qa_pairs:
            qa_type = pair.qa_type.value
            stats["qa_types"][qa_type] = stats["qa_types"].get(qa_type, 0) + 1

        return stats

    def export_qa_dataset(self, qa_pairs: List[QAPair],
                         format_type: str = "jsonl") -> str:
        """
        Export Q&A pairs in various formats.

        Args:
            qa_pairs: List of Q&A pairs to export
            format_type: Export format ("jsonl", "json", "csv")

        Returns:
            Formatted dataset string
        """
        if format_type == "jsonl":
            lines = []
            for pair in qa_pairs:
                record = {
                    "question": pair.question,
                    "answer": pair.answer,
                    "context": pair.context,
                    "qa_type": pair.qa_type.value,
                    "confidence": pair.confidence,
                    "metadata": pair.metadata or {}
                }
                lines.append(json.dumps(record, ensure_ascii=False))
            return "\n".join(lines)

        elif format_type == "json":
            records = []
            for pair in qa_pairs:
                records.append({
                    "question": pair.question,
                    "answer": pair.answer,
                    "context": pair.context,
                    "qa_type": pair.qa_type.value,
                    "confidence": pair.confidence,
                    "metadata": pair.metadata or {}
                })
            return json.dumps(records, indent=2, ensure_ascii=False)

        elif format_type == "csv":
            import csv
            import io
            output = io.StringIO()
            writer = csv.writer(output)

            # Write header
            writer.writerow(["Question", "Answer", "Context", "QA_Type", "Confidence"])

            # Write data
            for pair in qa_pairs:
                writer.writerow([
                    pair.question,
                    pair.answer,
                    pair.context,
                    pair.qa_type.value,
                    pair.confidence
                ])

            return output.getvalue()

        else:
            raise ValueError(f"Unsupported format: {format_type}")


# Global instance for easy access
qa_generator = QADatasetGenerator()


def generate_qa_from_relationships(relationships: List[Dict[str, Any]],
                                 max_pairs_per_relationship: int = 3) -> List[QAPair]:
    """
    Convenience function to generate Q&A pairs from relationships.

    Args:
        relationships: List of entity relationships
        max_pairs_per_relationship: Maximum pairs per relationship

    Returns:
        List of Q&A pairs
    """
    return qa_generator.generate_qa_pairs(relationships, max_pairs_per_relationship)


def add_domain_qa_templates(domain_templates: List[QATemplate]):
    """
    Add domain-specific Q&A generation templates.

    Args:
        domain_templates: List of templates for the domain
    """
    for template in domain_templates:
        qa_generator.add_template(template)


# Example domain-specific templates
def add_medical_qa_templates():
    """Add medical domain Q&A templates."""
    medical_templates = [
        QATemplate(
            name="symptom_diagnosis",
            qa_type=QAType.DIRECT_ASSOCIATION,
            source_categories=["symptoms"],
            target_categories=["disease_names"],
            question_template="A patient presents with {source_entity}. What is the most likely diagnosis?",
            answer_template="The patient likely has {target_entity} based on the presenting symptom of {source_entity}.",
            description="Generate clinical diagnosis Q&A from symptom-disease relationships"
        ),
        QATemplate(
            name="medication_indication",
            qa_type=QAType.DIRECT_ASSOCIATION,
            source_categories=["medications"],
            target_categories=["disease_names"],
            question_template="What condition is {source_entity} typically used to treat?",
            answer_template="{source_entity} is commonly used to treat {target_entity}.",
            description="Generate medication indication Q&A pairs"
        )
    ]
    add_domain_qa_templates(medical_templates)


def add_business_qa_templates():
    """Add business domain Q&A templates."""
    business_templates = [
        QATemplate(
            name="product_features",
            qa_type=QAType.HIERARCHICAL_RELATIONSHIP,
            source_categories=["products"],
            target_categories=["features"],
            question_template="What are the key features of {source_entity}?",
            answer_template="{source_entity} includes features such as {target_entity}.",
            description="Generate product feature Q&A pairs"
        ),
        QATemplate(
            name="company_products",
            qa_type=QAType.DIRECT_ASSOCIATION,
            source_categories=["companies"],
            target_categories=["products"],
            question_template="What products does {source_entity} offer?",
            answer_template="{source_entity} offers products including {target_entity}.",
            description="Generate company product Q&A pairs"
        )
    ]
    add_domain_qa_templates(business_templates)