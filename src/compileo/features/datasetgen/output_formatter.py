import json
import logging
from io import BytesIO
from typing import Any, Dict, List, Union

import pandas as pd

logger = logging.getLogger(__name__)

class OutputFormatter:
    """
    Formats the generated dataset into the desired output format.
    """

    def format_dataset(
        self, dataset_content: Union[List[Dict[str, Any]], Dict[str, Any]], format_type: str
    ) -> Union[str, bytes, Dict[str, Union[str, bytes]]]:
        """
        Formats the dataset content.

        Args:
            dataset_content: The content of the dataset to format. Can be a list of entries
                            or a dictionary containing multiple datasets (evaluation sets).
            format_type: The desired output format (e.g., 'jsonl', 'parquet').

        Returns:
            The formatted dataset as a string, bytes, or dictionary of formatted datasets.
        """
        if isinstance(dataset_content, dict):
            # Handle evaluation datasets with multiple splits
            return self._format_evaluation_datasets(dataset_content, format_type)
        else:
            # Handle single dataset (backward compatibility)
            if format_type == "raw_list":
                return dataset_content # type: ignore
            elif format_type == "json":
                return self._to_json(dataset_content)
            elif format_type == "jsonl":
                return self._to_jsonl(dataset_content)
            elif format_type == "parquet":
                return self._to_parquet(dataset_content)
            else:
                # Check for plugin extensions - import plugin_manager locally to avoid pickle issues
                try:
                    from src.compileo.features.plugin.manager import plugin_manager
                    extensions = plugin_manager.get_extensions("compileo.datasetgen.formatter")
                    if format_type in extensions:
                        try:
                            formatter_class = extensions[format_type]
                            # Instantiate and call format method
                            # We assume the plugin class has a 'format_dataset' method or similar
                            # For simplicity, let's assume it implements a 'format' method taking the content
                            formatter = formatter_class()
                            if hasattr(formatter, 'format'):
                                return formatter.format(dataset_content)
                            else:
                                raise ValueError(f"Plugin for {format_type} does not implement 'format' method")
                        except Exception as e:
                            logger.error(f"Error using plugin formatter for {format_type}: {e}")
                            raise ValueError(f"Plugin formatter error: {e}")
                except ImportError:
                    logger.warning("Plugin manager not available, plugin formats disabled")

                raise ValueError(f"Unsupported format_type: {format_type}")

    def _to_json(self, dataset_content: List[Dict[str, Any]]) -> str:
        """
        Converts the dataset to a JSON array string.

        Args:
            dataset_content: The dataset content.

        Returns:
            A JSON formatted string.
        """
        return json.dumps(dataset_content, indent=2)

    def _to_jsonl(self, dataset_content: List[Dict[str, Any]]) -> str:
        """
        Converts the dataset to a JSON Lines string.

        Args:
            dataset_content: The dataset content.

        Returns:
            A JSON Lines formatted string.
        """
        return "\n".join(json.dumps(record) for record in dataset_content)

    def _to_parquet(self, dataset_content: List[Dict[str, Any]]) -> bytes:
        """
        Converts the dataset to a Parquet file in memory.

        Args:
            dataset_content: The dataset content.

        Returns:
            The Parquet file as bytes.
        """
        df = pd.DataFrame(dataset_content)
        buffer = BytesIO()
        df.to_parquet(buffer, index=False)
        return buffer.getvalue()

    def _format_evaluation_datasets(
        self, evaluation_datasets: Dict[str, Any], format_type: str
    ) -> Dict[str, Union[str, bytes]]:
        """
        Formats evaluation datasets with multiple splits.

        Args:
            evaluation_datasets: Dictionary containing evaluation datasets
            format_type: The desired output format

        Returns:
            Dictionary of formatted datasets
        """
        formatted_datasets = {}

        # Format individual dataset components
        dataset_keys = [
            'original_dataset', 'train_set', 'validation_set', 'test_set',
            'adversarial_examples'
        ]

        for key in dataset_keys:
            if key in evaluation_datasets:
                if key == 'original_dataset':
                    formatted_key = f"full_dataset.{format_type}"
                elif key == 'train_set':
                    formatted_key = f"train.{format_type}"
                elif key == 'validation_set':
                    formatted_key = f"validation.{format_type}"
                elif key == 'test_set':
                    formatted_key = f"test.{format_type}"
                elif key == 'adversarial_examples':
                    formatted_key = f"adversarial.{format_type}"
                else:
                    formatted_key = f"{key}.{format_type}"

                if format_type == "jsonl":
                    formatted_datasets[formatted_key] = self._to_jsonl(evaluation_datasets[key])
                elif format_type == "parquet":
                    formatted_datasets[formatted_key] = self._to_parquet(evaluation_datasets[key])
                else:
                     # Check plugins for evaluation datasets too
                    try:
                        from src.compileo.features.plugin.manager import plugin_manager
                        extensions = plugin_manager.get_extensions("compileo.datasetgen.formatter")
                        if format_type in extensions:
                             try:
                                formatter_class = extensions[format_type]
                                formatter = formatter_class()
                                if hasattr(formatter, 'format'):
                                    formatted_datasets[formatted_key] = formatter.format(evaluation_datasets[key])
                             except Exception as e:
                                 logger.error(f"Error using plugin formatter for {format_type}: {e}")
                    except ImportError:
                        logger.warning("Plugin manager not available, plugin formats disabled")

        # Format cross-validation folds
        if 'cross_validation_folds' in evaluation_datasets:
            for i, (train_fold, val_fold) in enumerate(evaluation_datasets['cross_validation_folds']):
                if format_type == "jsonl":
                    formatted_datasets[f"cv_fold_{i}_train.{format_type}"] = self._to_jsonl(train_fold)
                    formatted_datasets[f"cv_fold_{i}_val.{format_type}"] = self._to_jsonl(val_fold)
                elif format_type == "parquet":
                    formatted_datasets[f"cv_fold_{i}_train.{format_type}"] = self._to_parquet(train_fold)
                    formatted_datasets[f"cv_fold_{i}_val.{format_type}"] = self._to_parquet(val_fold)
                else:
                    try:
                        from src.compileo.features.plugin.manager import plugin_manager
                        extensions = plugin_manager.get_extensions("compileo.datasetgen.formatter")
                        if format_type in extensions:
                             try:
                                formatter_class = extensions[format_type]
                                formatter = formatter_class()
                                if hasattr(formatter, 'format'):
                                    formatted_datasets[f"cv_fold_{i}_train.{format_type}"] = formatter.format(train_fold)
                                    formatted_datasets[f"cv_fold_{i}_val.{format_type}"] = formatter.format(val_fold)
                             except Exception:
                                 pass
                    except ImportError:
                        pass

        # Format difficulty stratification
        if 'difficulty_stratification' in evaluation_datasets:
            difficulty_data = evaluation_datasets['difficulty_stratification']
            for level in ['easy', 'medium', 'hard']:
                if level in difficulty_data:
                    if format_type == "jsonl":
                        formatted_datasets[f"difficulty_{level}.{format_type}"] = self._to_jsonl(difficulty_data[level])
                    elif format_type == "parquet":
                        formatted_datasets[f"difficulty_{level}.{format_type}"] = self._to_parquet(difficulty_data[level])
                    else:
                        try:
                            from src.compileo.features.plugin.manager import plugin_manager
                            extensions = plugin_manager.get_extensions("compileo.datasetgen.formatter")
                            if format_type in extensions:
                                try:
                                    formatter_class = extensions[format_type]
                                    formatter = formatter_class()
                                    if hasattr(formatter, 'format'):
                                        formatted_datasets[f"difficulty_{level}.{format_type}"] = formatter.format(difficulty_data[level])
                                except Exception:
                                    pass
                        except ImportError:
                            pass

        # Include metadata
        if 'metadata' in evaluation_datasets:
            import json
            formatted_datasets['metadata.json'] = json.dumps(evaluation_datasets['metadata'], indent=2)

        return formatted_datasets
