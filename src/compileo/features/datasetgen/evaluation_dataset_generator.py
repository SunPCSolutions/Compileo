"""
Evaluation Dataset Generator for creating comprehensive evaluation datasets.
"""

import random
from typing import List, Dict, Any, Tuple, Optional
from ...core.logging import get_logger

logger = get_logger(__name__)

from sklearn.model_selection import train_test_split, KFold
import numpy as np


class EvaluationDatasetGenerator:
    """
    Generates evaluation datasets including train/validation/test splits,
    cross-validation folds, adversarial examples, and difficulty stratification.
    """

    def __init__(self, random_seed: int = 42):
        """
        Initialize the evaluation dataset generator.

        Args:
            random_seed: Random seed for reproducible splits
        """
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

    def split_dataset(
        self,
        dataset: List[Dict[str, Any]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify_by: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split dataset into train/validation/test sets.

        Args:
            dataset: List of dataset entries
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            stratify_by: Field to stratify by (optional)

        Returns:
            Tuple of (train_set, val_set, test_set)
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")

        # Extract stratification labels if specified
        stratify_labels = None
        if stratify_by:
            stratify_labels = np.array([entry.get(stratify_by) for entry in dataset])

        # First split: separate test set
        test_size = test_ratio
        val_size = val_ratio / (1 - test_ratio)  # Adjust for remaining data

        if stratify_labels is not None:
            train_val, test_set = train_test_split(
                dataset,
                test_size=test_size,
                stratify=stratify_labels,
                random_state=self.random_seed
            )
            # Get stratification labels for remaining data
            train_val_labels = np.array([entry.get(stratify_by) for entry in train_val])
            train_set, val_set = train_test_split(
                train_val,
                test_size=val_size,
                stratify=train_val_labels,
                random_state=self.random_seed
            )
        else:
            train_val, test_set = train_test_split(
                dataset,
                test_size=test_size,
                random_state=self.random_seed
            )
            train_set, val_set = train_test_split(
                train_val,
                test_size=val_size,
                random_state=self.random_seed
            )

        return train_set, val_set, test_set

    def generate_cross_validation_folds(
        self,
        dataset: List[Dict[str, Any]],
        n_folds: int = 5,
        stratify_by: Optional[str] = None
    ) -> List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
        """
        Generate cross-validation folds.

        Args:
            dataset: List of dataset entries
            n_folds: Number of folds
            stratify_by: Field to stratify by (optional)

        Returns:
            List of (train_fold, val_fold) tuples
        """
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)

        indices = np.arange(len(dataset))
        folds = []

        if stratify_by:
            # Group indices by stratification label
            stratified_groups = {}
            for idx, entry in enumerate(dataset):
                label = entry.get(stratify_by)
                if label not in stratified_groups:
                    stratified_groups[label] = []
                stratified_groups[label].append(idx)

            # Create stratified folds using group labels
            group_labels = list(stratified_groups.keys())
            for train_group_indices, val_group_indices in kf.split(group_labels):
                train_fold = []
                val_fold = []

                for group_idx in train_group_indices:
                    label = group_labels[group_idx]
                    train_fold.extend([dataset[idx] for idx in stratified_groups[label]])

                for group_idx in val_group_indices:
                    label = group_labels[group_idx]
                    val_fold.extend([dataset[idx] for idx in stratified_groups[label]])

                folds.append((train_fold, val_fold))
        else:
            for train_indices, val_indices in kf.split(indices):
                train_fold = [dataset[idx] for idx in train_indices]
                val_fold = [dataset[idx] for idx in val_indices]
                folds.append((train_fold, val_fold))

        return folds

    def generate_adversarial_examples(
        self,
        dataset: List[Dict[str, Any]],
        adversarial_ratio: float = 0.1,
        llm_interaction: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate adversarial examples for robustness testing.

        Args:
            dataset: Original dataset
            adversarial_ratio: Ratio of adversarial examples to generate
            llm_interaction: LLM interaction object for generating adversarial examples

        Returns:
            List of adversarial examples
        """
        n_adversarial = int(len(dataset) * adversarial_ratio)
        adversarial_examples = []

        # Select random examples to create adversarial versions
        selected_indices = random.sample(range(len(dataset)), min(n_adversarial, len(dataset)))

        for idx in selected_indices:
            original = dataset[idx].copy()

            # Create adversarial example by modifying the question/answer
            if llm_interaction:
                # Use LLM to generate adversarial version
                adversarial_prompt = f"""
                Create an adversarial version of this question-answer pair that would be challenging for an AI model:

                Original Question: {original.get('question', '')}
                Original Answer: {original.get('answer', '')}

                Generate a modified version that:
                - Changes key terms to synonyms or related concepts
                - Adds ambiguity or complexity
                - Tests edge cases or uncommon scenarios
                - Maintains the same core meaning but increases difficulty

                Return in JSON format:
                {{
                    "question": "modified question",
                    "answer": "modified answer",
                    "adversarial_type": "synonym_replacement|ambiguity|edge_case|complexity"
                }}
                """

                try:
                    result = llm_interaction.generate(adversarial_prompt)
                    # Parse the result (assuming it's JSON)
                    import json
                    adversarial_data = json.loads(result.get('raw_response', '{}'))

                    adversarial_example = original.copy()
                    adversarial_example.update({
                        'question': adversarial_data.get('question', original.get('question', '')),
                        'answer': adversarial_data.get('answer', original.get('answer', '')),
                        'adversarial_type': adversarial_data.get('adversarial_type', 'unknown'),
                        'is_adversarial': True,
                        'original_index': idx
                    })
                    adversarial_examples.append(adversarial_example)

                except Exception as e:
                    logger.error(f"Failed to generate adversarial example: {e}")
                    # Fallback: create simple adversarial example
                    adversarial_example = original.copy()
                    adversarial_example.update({
                        'is_adversarial': True,
                        'original_index': idx,
                        'adversarial_type': 'fallback'
                    })
                    adversarial_examples.append(adversarial_example)
            else:
                # Simple fallback without LLM
                adversarial_example = original.copy()
                adversarial_example.update({
                    'is_adversarial': True,
                    'original_index': idx,
                    'adversarial_type': 'simple_modification'
                })
                adversarial_examples.append(adversarial_example)

        return adversarial_examples

    def stratify_by_difficulty(
        self,
        dataset: List[Dict[str, Any]],
        difficulty_criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Stratify dataset by difficulty levels.

        Args:
            dataset: List of dataset entries
            difficulty_criteria: Criteria for difficulty assessment

        Returns:
            Dictionary with 'easy', 'medium', 'hard' keys containing stratified datasets
        """
        if not difficulty_criteria:
            difficulty_criteria = {
                'question_length_thresholds': {'easy': 20, 'medium': 50},
                'answer_length_thresholds': {'easy': 10, 'medium': 30},
                'complexity_keywords': ['complex', 'advanced', 'difficult', 'challenging']
            }

        easy_set = []
        medium_set = []
        hard_set = []

        for entry in dataset:
            question = entry.get('question', '')
            answer = entry.get('answer', '')

            # Calculate difficulty score based on multiple factors
            difficulty_score = 0

            # Question length factor
            q_len = len(question.split())
            if q_len <= difficulty_criteria['question_length_thresholds']['easy']:
                difficulty_score += 1  # Easy
            elif q_len <= difficulty_criteria['question_length_thresholds']['medium']:
                difficulty_score += 2  # Medium
            else:
                difficulty_score += 3  # Hard

            # Answer length factor
            a_len = len(answer.split())
            if a_len <= difficulty_criteria['answer_length_thresholds']['easy']:
                difficulty_score += 1  # Easy
            elif a_len <= difficulty_criteria['answer_length_thresholds']['medium']:
                difficulty_score += 2  # Medium
            else:
                difficulty_score += 3  # Hard

            # Complexity keywords factor
            question_lower = question.lower()
            if any(keyword in question_lower for keyword in difficulty_criteria['complexity_keywords']):
                difficulty_score += 2  # Increase difficulty for complex keywords

            # Classify based on average difficulty score
            avg_score = difficulty_score / 3.0

            if avg_score <= 1.5:
                easy_set.append(entry)
            elif avg_score <= 2.5:
                medium_set.append(entry)
            else:
                hard_set.append(entry)

        return {
            'easy': easy_set,
            'medium': medium_set,
            'hard': hard_set
        }

    def create_evaluation_datasets(
        self,
        dataset: List[Dict[str, Any]],
        include_splits: bool = True,
        include_cv_folds: bool = True,
        include_adversarial: bool = True,
        include_difficulty_stratification: bool = True,
        llm_interaction: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create comprehensive evaluation datasets.

        Args:
            dataset: Original dataset
            include_splits: Whether to include train/val/test splits
            include_cv_folds: Whether to include cross-validation folds
            include_adversarial: Whether to include adversarial examples
            include_difficulty_stratification: Whether to include difficulty stratification
            llm_interaction: LLM interaction object for adversarial generation
            **kwargs: Additional parameters

        Returns:
            Dictionary containing all evaluation datasets
        """
        evaluation_datasets = {
            'original_dataset': dataset,
            'metadata': {
                'total_samples': len(dataset),
                'generation_timestamp': '2025-01-25T20:46:00Z',  # Should be dynamic
                'evaluation_components': []
            }
        }

        if include_splits:
            train_set, val_set, test_set = self.split_dataset(dataset, **kwargs)
            evaluation_datasets.update({
                'train_set': train_set,
                'validation_set': val_set,
                'test_set': test_set
            })
            evaluation_datasets['metadata']['evaluation_components'].append('train_val_test_split')

        if include_cv_folds:
            cv_folds = self.generate_cross_validation_folds(dataset, **kwargs)
            evaluation_datasets['cross_validation_folds'] = cv_folds
            evaluation_datasets['metadata']['evaluation_components'].append('cross_validation_folds')

        if include_adversarial:
            adversarial_examples = self.generate_adversarial_examples(dataset, llm_interaction=llm_interaction, **kwargs)
            evaluation_datasets['adversarial_examples'] = adversarial_examples
            evaluation_datasets['metadata']['evaluation_components'].append('adversarial_examples')

        if include_difficulty_stratification:
            difficulty_stratified = self.stratify_by_difficulty(dataset, **kwargs)
            evaluation_datasets['difficulty_stratification'] = difficulty_stratified
            evaluation_datasets['metadata']['evaluation_components'].append('difficulty_stratification')

        return evaluation_datasets