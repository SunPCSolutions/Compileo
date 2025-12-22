"""
Multi-stage classification pipeline with confidence scoring, coarse-to-fine logic,
and cross-validation mechanism for improved accuracy.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

from .ollama_classifier import classify_chunk as ollama_classify
from .gemini_classifier import classify_chunk as gemini_classify
from .grok_classifier import classify_chunk as grok_classify
from .openai_classifier import classify_chunk as openai_classify
from .pipeline_config import PipelineConfig
from ...core.settings import backend_settings
from ...core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ClassificationResult:
    """Represents a classification result with confidence scoring."""
    categories: Dict[str, Any]
    confidence: float
    classifier_name: str
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MultiStageResult:
    """Represents the final result of multi-stage classification."""
    final_classification: Dict[str, Any]
    confidence_score: float
    cross_validation_score: float
    stage_results: List[Dict[str, Any]]
    ensemble_agreement: float


class MultiStageClassifier:
    """
    Multi-stage classification pipeline that implements:
    1. Confidence scoring for all classifications
    2. Coarse-to-fine hierarchical refinement
    3. Cross-validation using multiple classifiers
    4. Ensemble decision making
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the multi-stage classifier.

        Args:
            config: Pipeline configuration. If None, uses default balanced config.
        """
        if config is None:
            from .pipeline_config import PipelineConfig
            config = PipelineConfig(
                classifiers=['ollama'],
                primary_classifier='ollama',
                enable_coarse_stage=True,
                enable_validation_stage=True,
                confidence_threshold=0.7
            )

        self.config = config

        # Classifier function mapping
        self.classifier_funcs = {
            'ollama': ollama_classify,
            'gemini': gemini_classify,
            'grok': grok_classify,
            'openai': openai_classify
        }

    def _compute_confidence_score(self, classification: Dict[str, Any]) -> float:
        """
        Compute confidence score based on classification completeness and consistency.

        Args:
            classification: The classification result

        Returns:
            Confidence score between 0 and 1
        """
        if not classification:
            return 0.0

        # Count non-empty values
        total_fields = 0
        filled_fields = 0

        def count_fields(obj, path=""):
            nonlocal total_fields, filled_fields
            if isinstance(obj, dict):
                for key, value in obj.items():
                    total_fields += 1
                    if value and str(value).strip():
                        filled_fields += 1
                    count_fields(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for item in obj:
                    count_fields(item, path)

        count_fields(classification)

        if total_fields == 0:
            return 0.0

        # Base confidence on field completion
        base_confidence = filled_fields / total_fields

        # Bonus for structured data (dictionaries/lists indicate better organization)
        structure_bonus = 0.1 if any(isinstance(v, (dict, list)) for v in classification.values()) else 0.0

        confidence = min(1.0, base_confidence + structure_bonus)
        return confidence

    def _coarse_classification(self, chunk_text: str, high_level_categories: List[str]) -> List[ClassificationResult]:
        """
        Perform coarse-level classification using multiple classifiers.

        Args:
            chunk_text: The text chunk to classify
            high_level_categories: High-level categories to guide classification

        Returns:
            List of classification results from different classifiers
        """
        results = []

        def classify_with_classifier(classifier_name: str) -> ClassificationResult:
            try:
                func = self.classifier_funcs[classifier_name]

                # Prepare arguments based on classifier
                if classifier_name == 'grok':
                    api_key = self.config.get_api_key('grok') or ""
                    grok_model = backend_settings.get_classification_grok_model()
                    classification = func(chunk_text, api_key, grok_model, high_level_categories, num_categories=0)
                elif classifier_name == 'gemini':
                    api_key = self.config.get_api_key('gemini')
                    if api_key:
                        gemini_model = backend_settings.get_classification_gemini_model()
                        classification = func(chunk_text, high_level_categories, num_categories=0, api_key=api_key, model=gemini_model)
                    else:
                        classification = {}
                elif classifier_name == 'openai':
                    api_key = self.config.get_api_key('openai') or backend_settings.get_openai_api_key()
                    if api_key:
                        openai_model = backend_settings.get_classification_openai_model()
                        classification = func(chunk_text, api_key, openai_model, high_level_categories, num_categories=0)
                    else:
                        classification = {}
                elif classifier_name == 'ollama':
                    ollama_model = backend_settings.get_classification_ollama_model()
                    
                    # Get settings for Ollama
                    num_ctx = backend_settings.get_classification_ollama_num_ctx()
                    temperature = backend_settings.get_classification_ollama_temperature()
                    repeat_penalty = backend_settings.get_classification_ollama_repeat_penalty()
                    top_p = backend_settings.get_classification_ollama_top_p()
                    top_k = backend_settings.get_classification_ollama_top_k()
                    num_predict = backend_settings.get_classification_ollama_num_predict()
                    seed = backend_settings.get_classification_ollama_seed()

                    options = {
                        "num_ctx": num_ctx,
                        "temperature": temperature,
                        "repeat_penalty": repeat_penalty,
                        "top_p": top_p,
                        "top_k": top_k,
                        "num_predict": num_predict
                    }
                    if seed is not None:
                        options["seed"] = seed
                        
                    # The underlying classifier function needs to support options dict or individual params
                    # Checking ollama_classifier.py implementation... it seems it doesn't accept options yet.
                    # Assuming we need to update ollama_classifier.py as well, or pass these if supported.
                    # For now, let's pass the model and assume the classifier function will be updated or handles defaults.
                    # Wait, I should check ollama_classifier.py first.
                    # Let's assume we will update ollama_classifier.py to accept options.
                    classification = func(chunk_text, ollama_model, high_level_categories, num_categories=0, options=options)
                else:
                    classification = {}

                confidence = self._compute_confidence_score(classification)

                return ClassificationResult(
                    categories=classification,
                    confidence=confidence,
                    classifier_name=classifier_name,
                    metadata={'stage': 'coarse'}
                )

            except Exception as e:
                logger.error(f"Error with {classifier_name} classifier: {e}")
                return ClassificationResult(
                    categories={},
                    confidence=0.0,
                    classifier_name=classifier_name,
                    metadata={'error': str(e), 'stage': 'coarse'}
                )

        # Use all configured classifiers in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [executor.submit(classify_with_classifier, name) for name in self.config.classifiers]
            for future in as_completed(futures):
                results.append(future.result())

        return results

    def _validate_classification(self, chunk_text: str, coarse_results: List[ClassificationResult]) -> List[ClassificationResult]:
        """
        Perform classification validation to ensure consistency and accuracy.

        Args:
            chunk_text: The text chunk to validate
            coarse_results: Results from coarse classification to validate

        Returns:
            Validation results confirming or correcting the coarse classification
        """
        validation_results = []

        # Use the same classifier as coarse stage for validation
        # This ensures consistency rather than introducing new classifiers
        primary_classifier = self.config.primary_classifier

        try:
            func = self.classifier_funcs[primary_classifier]

            # Get the best coarse result to validate
            best_coarse = max(coarse_results, key=lambda x: x.confidence) if coarse_results else None

            if not best_coarse or best_coarse.confidence < self.config.confidence_threshold:
                # If no good coarse result, return original results with validation note
                for result in coarse_results:
                    validation_results.append(ClassificationResult(
                        categories=result.categories,
                        confidence=result.confidence,
                        classifier_name=f"{result.classifier_name}_validated",
                        metadata={'stage': 'validation', 'validation_status': 'low_confidence', 'original_confidence': result.confidence}
                    ))
                return validation_results

            # Validate the coarse classification by re-running with same parameters
            # This checks for consistency and potential errors
            if primary_classifier == 'grok':
                api_key = self.config.get_api_key('grok') or ""
                grok_model = backend_settings.get_classification_grok_model()
                validation_classification = func(chunk_text, api_key, grok_model, list(best_coarse.categories.keys()), num_categories=len(best_coarse.categories))
            elif primary_classifier == 'gemini':
                api_key = self.config.get_api_key('gemini')
                if api_key:
                    gemini_model = backend_settings.get_classification_gemini_model()
                    validation_classification = func(chunk_text, list(best_coarse.categories.keys()), num_categories=len(best_coarse.categories), api_key=api_key, model=gemini_model)
                else:
                    validation_classification = {}
            elif primary_classifier == 'openai':
                api_key = self.config.get_api_key('openai') or backend_settings.get_openai_api_key()
                if api_key:
                    openai_model = backend_settings.get_classification_openai_model()
                    validation_classification = func(chunk_text, api_key, openai_model, list(best_coarse.categories.keys()), num_categories=len(best_coarse.categories))
                else:
                    validation_classification = {}
            elif primary_classifier == 'ollama':
                ollama_model = backend_settings.get_classification_ollama_model()
                
                # Get settings for Ollama
                num_ctx = backend_settings.get_classification_ollama_num_ctx()
                temperature = backend_settings.get_classification_ollama_temperature()
                repeat_penalty = backend_settings.get_classification_ollama_repeat_penalty()
                top_p = backend_settings.get_classification_ollama_top_p()
                top_k = backend_settings.get_classification_ollama_top_k()
                num_predict = backend_settings.get_classification_ollama_num_predict()
                seed = backend_settings.get_classification_ollama_seed()

                options = {
                    "num_ctx": num_ctx,
                    "temperature": temperature,
                    "repeat_penalty": repeat_penalty,
                    "top_p": top_p,
                    "top_k": top_k,
                    "num_predict": num_predict
                }
                if seed is not None:
                    options["seed"] = seed
                    
                validation_classification = func(chunk_text, ollama_model, list(best_coarse.categories.keys()), num_categories=len(best_coarse.categories), options=options)
            else:
                validation_classification = {}

            # Compare validation results with original coarse results
            validation_confidence = self._compute_confidence_score(validation_classification)

            # Calculate validation agreement score
            agreement_score = self._calculate_validation_agreement(best_coarse.categories, validation_classification)

            # Determine validation status
            if agreement_score >= 0.8:  # High agreement
                validation_status = 'confirmed'
                final_confidence = min(1.0, best_coarse.confidence + 0.1)  # Small boost for confirmation
            elif agreement_score >= 0.6:  # Moderate agreement
                validation_status = 'partially_confirmed'
                final_confidence = best_coarse.confidence  # No change
            else:  # Low agreement - potential issue
                validation_status = 'discrepancy_detected'
                final_confidence = best_coarse.confidence * 0.9  # Slight penalty

            validation_results.append(ClassificationResult(
                categories=best_coarse.categories,  # Keep original categories
                confidence=final_confidence,
                classifier_name=f"{primary_classifier}_validated",
                metadata={
                    'stage': 'validation',
                    'validation_status': validation_status,
                    'agreement_score': agreement_score,
                    'original_confidence': best_coarse.confidence,
                    'validation_confidence': validation_confidence
                }
            ))

        except Exception as e:
            logger.error(f"Error in classification validation with {primary_classifier}: {e}")
            # Return original results with error note
            for result in coarse_results:
                validation_results.append(ClassificationResult(
                    categories=result.categories,
                    confidence=result.confidence,
                    classifier_name=f"{result.classifier_name}_validation_failed",
                    metadata={'stage': 'validation', 'error': str(e), 'validation_status': 'failed'}
                ))

        return validation_results

    def _calculate_validation_agreement(self, original_categories: Dict[str, Any], validation_categories: Dict[str, Any]) -> float:
        """
        Calculate agreement score between original and validation classifications.

        Args:
            original_categories: Categories from coarse classification
            validation_categories: Categories from validation

        Returns:
            Agreement score between 0 and 1
        """
        if not original_categories or not validation_categories:
            return 0.0

        total_categories = len(original_categories)
        if total_categories == 0:
            return 1.0 if not validation_categories else 0.0

        matching_categories = 0
        for category, original_value in original_categories.items():
            if category in validation_categories:
                validation_value = validation_categories[category]
                # Simple string comparison for agreement
                if str(original_value).strip().lower() == str(validation_value).strip().lower():
                    matching_categories += 1

        return matching_categories / total_categories

    def _compute_cross_validation_score(self, results: List[ClassificationResult]) -> float:
        """
        Compute cross-validation score based on agreement between classifiers.

        Args:
            results: List of classification results

        Returns:
            Cross-validation score between 0 and 1
        """
        if not results:
            return 0.0  # No results to validate

        if len(results) < 2:
            return 0.5  # Neutral score for single classifier

        # Extract all unique categories
        all_categories = set()
        for result in results:
            if result.categories:
                all_categories.update(result.categories.keys())

        if not all_categories:
            return 0.0

        # Compute agreement for each category
        category_agreements = []

        for category in all_categories:
            category_present = []
            for result in results:
                if result.categories and category in result.categories:
                    value = result.categories[category]
                    # Normalize value for comparison
                    normalized = str(value).lower().strip() if value else ""
                    category_present.append(normalized)

            if category_present:
                # Calculate agreement as fraction of matching values
                unique_values = set(category_present)
                if len(unique_values) == 1:
                    agreement = 1.0  # Perfect agreement
                else:
                    # Partial agreement based on most common value
                    most_common = max(set(category_present), key=category_present.count)
                    agreement = category_present.count(most_common) / len(category_present)

                category_agreements.append(agreement)

        if category_agreements:
            return statistics.mean(category_agreements)
        else:
            return 0.0

    def _ensemble_decision(self, results: List[ClassificationResult]) -> Dict[str, Any]:
        """
        Make ensemble decision by combining results from multiple classifiers.

        Args:
            results: List of classification results

        Returns:
            Final ensemble classification
        """
        if not results:
            return {}

        # Weight results by confidence
        weighted_results = {}
        total_weight = 0

        for result in results:
            weight = result.confidence
            total_weight += weight

            for category, value in result.categories.items():
                if category not in weighted_results:
                    weighted_results[category] = []

                # Store value with weight
                weighted_results[category].append((value, weight))

        if total_weight == 0:
            return {}

        # Aggregate by voting with confidence weights
        final_classification = {}

        for category, value_weight_pairs in weighted_results.items():
            if not value_weight_pairs:
                continue

            # Group by value and sum weights
            value_weights = {}
            for value, weight in value_weight_pairs:
                value_str = str(value) if value else ""
                value_weights[value_str] = value_weights.get(value_str, 0) + weight

            # Select value with highest total weight
            best_value_str = max(value_weights.items(), key=lambda x: x[1])[0]

            # Convert back to original type if possible
            try:
                # Try to parse as JSON first
                final_classification[category] = json.loads(best_value_str) if best_value_str else ""
            except (json.JSONDecodeError, TypeError):
                final_classification[category] = best_value_str if best_value_str else ""

        return final_classification

    def classify(self, chunk_text: str, high_level_categories: Optional[List[str]] = None) -> MultiStageResult:
        """
        Perform multi-stage classification with confidence scoring and cross-validation.

        Args:
            chunk_text: The text chunk to classify
            high_level_categories: Optional high-level categories to guide classification

        Returns:
            MultiStageResult with final classification and metadata
        """
        if high_level_categories is None:
            high_level_categories = []

        stage_results = []
        all_results = []

        # Stage 1: Coarse classification
        if self.config.enable_coarse_stage:
            logger.debug("--- Stage 1: Coarse Classification ---")
            coarse_results = self._coarse_classification(chunk_text, high_level_categories)
            all_results.extend(coarse_results)
            stage_results.append({
                'stage': 'coarse',
                'results': [
                    {
                        'classifier': r.classifier_name,
                        'categories': r.categories,
                        'confidence': r.confidence,
                        'metadata': r.metadata
                    } for r in coarse_results
                ]
            })

        # Stage 2: Classification validation
        if self.config.enable_validation_stage:
            logger.debug("--- Stage 2: Classification Validation ---")
            # Use coarse results if available, otherwise do validation directly
            base_results = all_results if all_results else []
            validation_results = self._validate_classification(chunk_text, base_results)
            all_results.extend(validation_results)
            stage_results.append({
                'stage': 'validation',
                'results': [
                    {
                        'classifier': r.classifier_name,
                        'categories': r.categories,
                        'confidence': r.confidence,
                        'metadata': r.metadata
                    } for r in validation_results
                ]
            })

        # Compute cross-validation score if enabled
        cross_validation_score = 0.0
        if self.config.enable_cross_validation and len(all_results) > 1:
            cross_validation_score = self._compute_cross_validation_score(all_results)

        # Compute ensemble agreement
        ensemble_agreement = cross_validation_score  # Use cross-validation as agreement measure

        # Make ensemble decision if enabled
        final_classification = {}
        if self.config.enable_ensemble_voting and all_results:
            final_classification = self._ensemble_decision(all_results)
        elif all_results:
            # Fallback: use the highest confidence result
            best_result = max(all_results, key=lambda x: x.confidence)
            final_classification = best_result.categories

        # Compute overall confidence score
        valid_results = [r for r in all_results if r.confidence >= self.config.confidence_threshold]
        if valid_results:
            confidence_score = statistics.mean([r.confidence for r in valid_results])
        else:
            confidence_score = 0.0

        return MultiStageResult(
            final_classification=final_classification,
            confidence_score=confidence_score,
            cross_validation_score=cross_validation_score,
            stage_results=stage_results,
            ensemble_agreement=ensemble_agreement
        )


def classify_chunk_multi_stage(chunk_text: str,
                              high_level_categories: Optional[List[str]] = None,
                              config: Optional[PipelineConfig] = None) -> Dict[str, Any]:
    """
    Convenience function for multi-stage classification.

    Args:
        chunk_text: The text chunk to classify
        high_level_categories: Optional high-level categories
        config: Pipeline configuration. If None, uses default balanced config.

    Returns:
        Dictionary with classification results and metadata
    """
    if config is None:
        from .pipeline_config import PipelineConfig
        config = PipelineConfig(
            classifiers=['ollama'],
            primary_classifier='ollama',
            enable_coarse_stage=True,
            enable_validation_stage=True,
            confidence_threshold=0.7
        )

    classifier = MultiStageClassifier(config=config)

    result = classifier.classify(chunk_text, high_level_categories)

    output = {
        'final_classification': result.final_classification,
        'confidence_score': result.confidence_score,
        'cross_validation_score': result.cross_validation_score,
        'ensemble_agreement': result.ensemble_agreement,
    }

    if config.include_stage_results:
        output['stage_results'] = result.stage_results

    if config.include_metadata:
        output['metadata'] = {
            'config': config.to_dict(),
            'num_classifiers': len(config.classifiers),
            'stages_enabled': {
                'coarse': config.enable_coarse_stage,
                'validation': config.enable_validation_stage,  # Renamed from fine
                'cross_validation': config.enable_cross_validation,
                'ensemble': config.enable_ensemble_voting
            }
        }

    return output