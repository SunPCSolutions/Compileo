"""
GLUE (General Language Understanding Evaluation) benchmark implementation.
"""

from typing import Dict, List, Any, Optional
import logging
from .base import BaseBenchmark, BenchmarkResult
from ...datasetgen.llm_interaction import LLMInteraction

logger = logging.getLogger(__name__)


class GLUEBenchmark(BaseBenchmark):
    """
    GLUE benchmark suite implementation.

    GLUE consists of 9 tasks:
    - CoLA (Corpus of Linguistic Acceptability)
    - SST-2 (Stanford Sentiment Treebank)
    - MRPC (Microsoft Research Paraphrase Corpus)
    - QQP (Quora Question Pairs)
    - MNLI (Multi-Genre Natural Language Inference)
    - QNLI (Question-answering Natural Language Inference)
    - RTE (Recognizing Textual Entailment)
    - WNLI (Winograd Natural Language Inference)
    """

    GLUE_TASKS = [
        "cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "wnli"
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._datasets = {}
        self._available = self._check_dependencies()

    def _check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        try:
            import datasets
            import evaluate
            return True
        except ImportError:
            logger.warning("GLUE benchmark requires 'datasets' and 'evaluate' packages")
            return False

    def get_tasks(self) -> List[str]:
        """Get list of GLUE tasks."""
        return self.GLUE_TASKS.copy()

    def load_dataset(self, task_name: str) -> Any:
        """Load GLUE dataset for a specific task."""
        if not self._available:
            raise ImportError("GLUE dependencies not available")

        if task_name not in self.GLUE_TASKS:
            raise ValueError(f"Unknown GLUE task: {task_name}")

        if task_name in self._datasets:
            return self._datasets[task_name]

        try:
            from datasets import load_dataset

            # Map task names to dataset names
            dataset_mapping = {
                "cola": "glue",
                "sst2": "glue",
                "mrpc": "glue",
                "qqp": "glue",
                "mnli": "glue",
                "qnli": "glue",
                "rte": "glue",
                "wnli": "glue"
            }

            dataset = load_dataset(dataset_mapping[task_name], task_name)
            self._datasets[task_name] = dataset
            return dataset

        except Exception as e:
            logger.error(f"Failed to load GLUE dataset for task {task_name}: {e}")
            raise

    def evaluate(self, ai_config: Dict[str, Any], **kwargs) -> List[BenchmarkResult]:
        """
        Evaluate using AI provider configuration on GLUE tasks.

        Args:
            ai_config: AI provider configuration (gemini_api_key, grok_api_key, ollama_available, etc.)
            **kwargs: Additional evaluation parameters

        Returns:
            List of BenchmarkResult objects
        """
        if not self._available:
            raise ImportError("GLUE dependencies not available")

        results = []
        tasks_to_run = kwargs.get('tasks', self.GLUE_TASKS)

        for task_name in tasks_to_run:
            if task_name not in self.GLUE_TASKS:
                logger.warning(f"Skipping unknown GLUE task: {task_name}")
                continue

            try:
                result = self._evaluate_task(ai_config, task_name, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate GLUE task {task_name}: {e}")
                continue

        return results

    def _evaluate_task(self, ai_config: Dict[str, Any], task_name: str, **kwargs) -> BenchmarkResult:
        """Evaluate a single GLUE task."""
        dataset = self.load_dataset(task_name)

        # Get validation/test split
        eval_split = "validation"
        if task_name == "mnli":
            eval_split = "validation_matched"  # MNLI has matched/mismatched

        eval_dataset = dataset[eval_split]

        # Prepare inputs based on task
        inputs, labels = self._prepare_task_inputs(task_name, eval_dataset)

        # Get model predictions
        predictions = self._get_model_predictions(ai_config, inputs, task_name)

        # Calculate metrics
        metrics = self._calculate_metrics(task_name, predictions, labels)

        return BenchmarkResult(
            benchmark_name="GLUE",
            task_name=task_name,
            metrics=metrics,
            predictions=predictions,
            labels=labels
        )

    def _prepare_task_inputs(self, task_name: str, dataset) -> tuple:
        """Prepare inputs and labels for a GLUE task."""
        inputs = []
        labels = []

        for example in dataset:
            if task_name == "cola":
                text = example["sentence"]
                label = example["label"]
            elif task_name == "sst2":
                text = example["sentence"]
                label = example["label"]
            elif task_name == "mrpc":
                text = f"{example['sentence1']} [SEP] {example['sentence2']}"
                label = example["label"]
            elif task_name == "qqp":
                text = f"{example['question1']} [SEP] {example['question2']}"
                label = example["label"]
            elif task_name == "mnli":
                text = f"{example['premise']} [SEP] {example['hypothesis']}"
                label = example["label"]
            elif task_name == "qnli":
                text = f"{example['question']} [SEP] {example['sentence']}"
                label = example["label"]
            elif task_name in ["rte", "wnli"]:
                text = f"{example['sentence1']} [SEP] {example['sentence2']}"
                label = example["label"]
            else:
                raise ValueError(f"Unsupported GLUE task: {task_name}")

            inputs.append(text)
            labels.append(label)

        return inputs, labels

    def _get_model_predictions(self, ai_config: Dict[str, Any], inputs: List[str], task_name: str) -> List[Any]:
        """Get predictions from the AI provider."""
        predictions = []

        # Import AI interaction utilities
        from ...datasetgen.llm_interaction import LLMInteraction

        # Initialize LLM interaction with the provided config
        provider = self._get_provider_from_config(ai_config)
        api_key = self._get_api_key_from_config(ai_config, provider)
        model = ai_config.get('model_name', 'mistral:latest')

        llm = LLMInteraction(llm_provider=provider, api_key=api_key, model=model)

        for text_input in inputs:
            try:
                # Get prediction based on task type
                pred = self._get_single_prediction(llm, text_input, task_name, ai_config)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Model prediction failed for input: {text_input[:50]}...: {e}")
                predictions.append(self._get_default_prediction(task_name))

        return predictions

    def _get_provider_from_config(self, ai_config: Dict[str, Any]) -> str:
        """Determine which AI provider to use based on config."""
        # Check for explicit provider first
        if ai_config.get('provider'):
            return ai_config['provider']

        if ai_config.get('ollama_available') and ai_config.get('model_name'):
            return 'ollama'
        elif ai_config.get('gemini_api_key'):
            return 'gemini'
        elif ai_config.get('grok_api_key'):
            return 'grok'
        elif ai_config.get('openai_api_key'):
            return 'openai'
        else:
            return 'ollama'  # Default fallback

    def _get_api_key_from_config(self, ai_config: Dict[str, Any], provider: str) -> Optional[str]:
        """Get API key for the specified provider."""
        if provider == 'gemini':
            return ai_config.get('gemini_api_key')
        elif provider == 'grok':
            return ai_config.get('grok_api_key')
        elif provider == 'openai':
            return ai_config.get('openai_api_key')
        else:
            return None  # Ollama doesn't need API key

    def _get_single_prediction(self, llm: 'LLMInteraction', text_input: str, task_name: str, ai_config: Dict[str, Any]) -> Any:
        """Get prediction for a single input."""
        # Create task-specific prompt
        prompt = self._create_task_prompt(text_input, task_name)

        # Log input prompt (truncated)
        logger.info(f"[{task_name}] Input: {text_input[:50]}... | Prompt: {prompt[:100]}...")

        # Get response from LLM
        response = llm.generate(
            prompt=prompt,
            options={
                'temperature': 0.1,  # Low temperature for consistent evaluation
                'num_predict': 10  # Short responses for classification
            }
        )

        # Parse response based on task - extract answer from LLM response dict
        answer_text = response.get('answer', str(response))

        # Log model response (truncated)
        logger.info(f"[{task_name}] Response: {answer_text[:100]}...")

        return self._parse_response(answer_text, task_name)

    def _create_task_prompt(self, text_input: str, task_name: str) -> str:
        """Create appropriate prompt for the task."""
        if task_name == "cola":
            return f"Determine if this sentence is grammatically acceptable. Answer with only 'acceptable' or 'unacceptable': {text_input}"
        elif task_name == "sst2":
            return f"Determine the sentiment of this sentence. Answer with only 'positive' or 'negative': {text_input}"
        elif task_name == "mrpc":
            # For MRPC, we need to handle sentence pairs
            if isinstance(text_input, str) and '[SEP]' in text_input:
                sent1, sent2 = text_input.split('[SEP]', 1)
                return f"Determine if these two sentences have the same meaning. Answer with only 'equivalent' or 'not_equivalent': Sentence 1: {sent1.strip()} Sentence 2: {sent2.strip()}"
            return f"Determine if this text shows sentence equivalence. Answer with only 'equivalent' or 'not_equivalent': {text_input}"
        elif task_name in ["rte", "wnli"]:
            # Binary NLI tasks
            if isinstance(text_input, str) and '[SEP]' in text_input:
                premise, hypothesis = text_input.split('[SEP]', 1)
                return f"Determine if the hypothesis follows from the premise. Answer with only 'entailment' or 'not_entailment': Premise: {premise.strip()} Hypothesis: {hypothesis.strip()}"
            return f"Determine entailment relationship. Answer with only 'entailment' or 'not_entailment': {text_input}"
        elif task_name == "qnli":
            # QNLI is also binary (entailment/not_entailment) derived from SQuAD
            if isinstance(text_input, str) and '[SEP]' in text_input:
                premise, hypothesis = text_input.split('[SEP]', 1)
                return f"Determine if the answer to the question is contained in the sentence. Answer with only 'entailment' or 'not_entailment': Question: {premise.strip()} Sentence: {hypothesis.strip()}"
            return f"Determine entailment relationship. Answer with only 'entailment' or 'not_entailment': {text_input}"
        elif task_name == "mnli":
            # Multi-genre NLI
            if isinstance(text_input, str) and '[SEP]' in text_input:
                premise, hypothesis = text_input.split('[SEP]', 1)
                return f"Determine if the hypothesis follows from the premise. Answer with only 'entailment', 'contradiction', or 'neutral': Premise: {premise.strip()} Hypothesis: {hypothesis.strip()}"
            return f"Determine entailment relationship. Answer with only 'entailment', 'contradiction', or 'neutral': {text_input}"
        else:
            # QQP and other tasks
            if isinstance(text_input, str) and '[SEP]' in text_input:
                q1, q2 = text_input.split('[SEP]', 1)
                return f"Determine if these questions are duplicates. Answer with only 'duplicate' or 'not_duplicate': Question 1: {q1.strip()} Question 2: {q2.strip()}"
            return f"Analyze this text and provide a classification. Answer with only the predicted class: {text_input}"

    def _parse_response(self, response: str, task_name: str) -> Any:
        """Parse LLM response into appropriate format."""
        response = response.strip().lower()

        # Binary classification tasks
        if task_name in ["cola", "sst2", "mrpc", "qqp", "rte", "wnli", "qnli"]:
            # Check for task-specific positive keywords
            if ("acceptable" in response and task_name == "cola") or \
               ("positive" in response and task_name == "sst2") or \
               ("equivalent" in response and task_name == "mrpc") or \
               ("duplicate" in response and task_name == "qqp") or \
               ("entailment" in response and "not_entailment" not in response and task_name in ["rte", "wnli", "qnli"]):
                return 1
            # Check for task-specific negative keywords
            elif ("unacceptable" in response and task_name == "cola") or \
                 ("negative" in response and task_name == "sst2") or \
                 ("not_equivalent" in response and task_name == "mrpc") or \
                 ("not_duplicate" in response and task_name == "qqp") or \
                 ("not_entailment" in response and task_name in ["rte", "wnli", "qnli"]):
                return 0
            else:
                return self._get_default_prediction(task_name)

        # 3-class classification tasks
        elif task_name == "mnli":
            if "entailment" in response:
                return 2
            elif "contradiction" in response:
                return 0
            elif "neutral" in response:
                return 1
            else:
                return self._get_default_prediction(task_name)

        return self._get_default_prediction(task_name)

    def _get_default_prediction(self, task_name: str) -> Any:
        """Get default prediction when parsing fails."""
        if task_name in ["cola", "sst2", "mrpc", "qqp", "rte", "wnli", "qnli"]:
            return 0  # Default to negative/false class
        elif task_name == "mnli":
            return 1  # Default to neutral class
        else:
            return 0



    def _calculate_metrics(self, task_name: str, predictions: List[Any], labels: List[Any]) -> Dict[str, Any]:
        """Calculate metrics for a GLUE task."""
        try:
            import evaluate

            if task_name == "cola":
                metric = evaluate.load("glue", "cola")
            elif task_name == "sst2":
                metric = evaluate.load("glue", "sst2")
            elif task_name == "mrpc":
                metric = evaluate.load("glue", "mrpc")
            elif task_name == "qqp":
                metric = evaluate.load("glue", "qqp")
            elif task_name == "mnli":
                metric = evaluate.load("glue", "mnli")
            elif task_name == "qnli":
                metric = evaluate.load("glue", "qnli")
            elif task_name == "rte":
                metric = evaluate.load("glue", "rte")
            elif task_name == "wnli":
                metric = evaluate.load("glue", "wnli")
            else:
                raise ValueError(f"Unsupported GLUE task: {task_name}")

            results = metric.compute(predictions=predictions, references=labels)
            return results

        except ImportError:
            logger.warning(f"Evaluate library not found for {task_name}, falling back to basic accuracy")
            # Fallback to basic accuracy if evaluate not available
            correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
            accuracy = correct / len(labels) if labels else 0.0
            return {"accuracy": accuracy}
        except Exception as e:
            logger.error(f"Failed to calculate metrics for {task_name}: {e}")
            # Fallback to basic accuracy on other errors
            correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
            accuracy = correct / len(labels) if labels else 0.0
            return {"accuracy": accuracy, "error": str(e)}

    def is_available(self) -> bool:
        """Check if GLUE benchmark is available."""
        return self._available