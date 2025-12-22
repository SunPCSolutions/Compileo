"""
Text generation performance metrics.
"""

from typing import Dict, List, Any, Optional
import logging
from .base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


class BLEUScore(BaseMetric):
    """BLEU score for text generation evaluation."""

    def name(self) -> str:
        return "bleu"

    def calculate(self, predictions: List[Any], labels: List[Any], **kwargs) -> MetricResult:
        """Calculate BLEU score."""
        try:
            import sacrebleu
        except ImportError:
            logger.warning("sacrebleu not available, using basic BLEU calculation")
            return self._basic_bleu(predictions, labels)

        try:
            # Convert to strings if needed
            pred_strs = [str(p) for p in predictions]
            label_strs = [[str(l)] for l in labels]  # BLEU expects list of references

            bleu = sacrebleu.corpus_bleu(pred_strs, label_strs)
            score = bleu.score / 100.0  # Convert to 0-1 scale

            return MetricResult(
                metric_name=self.name(),
                value=score,
                metadata={
                    "bleu_score": bleu.score,
                    "precisions": bleu.precisions,
                    "bp": bleu.bp,
                    "ratio": bleu.ratio,
                    "ref_len": bleu.ref_len,
                    "sys_len": bleu.sys_len
                }
            )
        except Exception as e:
            logger.error(f"BLEU calculation failed: {e}")
            return MetricResult(metric_name=self.name(), value=0.0)

    def _basic_bleu(self, predictions: List[Any], labels: List[Any]) -> MetricResult:
        """Basic BLEU calculation using NLTK."""
        try:
            from nltk.translate.bleu_score import corpus_bleu
        except ImportError:
            logger.warning("NLTK not available for BLEU calculation")
            return MetricResult(metric_name=self.name(), value=0.0)

        try:
            pred_strs = [str(p).split() for p in predictions]
            label_strs = [[str(l).split()] for l in labels]

            bleu = corpus_bleu(label_strs, pred_strs)
            # corpus_bleu returns a float
            return MetricResult(
                metric_name=self.name(),
                value=bleu if isinstance(bleu, float) else 0.0
            )
        except Exception as e:
            logger.error(f"Basic BLEU calculation failed: {e}")
            return MetricResult(metric_name=self.name(), value=0.0)


class ROUGEScore(BaseMetric):
    """ROUGE score for text generation evaluation."""

    def name(self) -> str:
        return "rouge"

    def calculate(self, predictions: List[Any], labels: List[Any], **kwargs) -> MetricResult:
        """Calculate ROUGE scores."""
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            logger.warning("rouge-score not available")
            return MetricResult(metric_name=self.name(), value=0.0)

        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

            pred_strs = [str(p) for p in predictions]
            label_strs = [str(l) for l in labels]

            scores = {
                'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
                'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
                'rougeL': {'precision': [], 'recall': [], 'fmeasure': []}
            }

            for pred, label in zip(pred_strs, label_strs):
                score = scorer.score(label, pred)
                for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
                    scores[rouge_type]['precision'].append(score[rouge_type].precision)
                    scores[rouge_type]['recall'].append(score[rouge_type].recall)
                    scores[rouge_type]['fmeasure'].append(score[rouge_type].fmeasure)

            # Average scores
            avg_scores = {}
            for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
                avg_scores[f'{rouge_type}_precision'] = sum(scores[rouge_type]['precision']) / len(scores[rouge_type]['precision'])
                avg_scores[f'{rouge_type}_recall'] = sum(scores[rouge_type]['recall']) / len(scores[rouge_type]['recall'])
                avg_scores[f'{rouge_type}_fmeasure'] = sum(scores[rouge_type]['fmeasure']) / len(scores[rouge_type]['fmeasure'])

            # Use ROUGE-L F1 as primary score
            primary_score = avg_scores['rougeL_fmeasure']

            return MetricResult(
                metric_name=self.name(),
                value=primary_score,
                metadata=avg_scores
            )
        except Exception as e:
            logger.error(f"ROUGE calculation failed: {e}")
            return MetricResult(metric_name=self.name(), value=0.0)


class METEORScore(BaseMetric):
    """METEOR score for text generation evaluation."""

    def name(self) -> str:
        return "meteor"

    def calculate(self, predictions: List[Any], labels: List[Any], **kwargs) -> MetricResult:
        """Calculate METEOR score."""
        try:
            from nltk.translate.meteor_score import meteor_score
        except ImportError:
            logger.warning("NLTK not available for METEOR calculation")
            return MetricResult(metric_name=self.name(), value=0.0)

        try:
            pred_strs = [str(p) for p in predictions]
            label_strs = [str(l) for l in labels]

            scores = []
            for pred, label in zip(pred_strs, label_strs):
                pred_tokens = pred.split()
                label_tokens = label.split()
                score = meteor_score([label_tokens], pred_tokens)
                scores.append(score)

            avg_score = sum(scores) / len(scores) if scores else 0.0

            return MetricResult(
                metric_name=self.name(),
                value=avg_score,
                metadata={"individual_scores": scores}
            )
        except Exception as e:
            logger.error(f"METEOR calculation failed: {e}")
            return MetricResult(metric_name=self.name(), value=0.0)