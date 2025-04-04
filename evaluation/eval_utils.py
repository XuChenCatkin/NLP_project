from abc import ABC, abstractmethod
from typing import List, Union, Dict
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score
import math
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class Metric(ABC):
    """Abstract base class for all metrics."""

    def __init__(self, name: str, metric_type: str):
        self._name = name
        self._metric_type = metric_type

    @property
    def name(self):
        return self._name
    
    @property
    def metric_type(self):
        return self._metric_type

    @abstractmethod
    def update(self, predictions, references):
        """Update the state of the metric with new predictions and references."""
        pass

    @abstractmethod
    def compute(self):
        """Compute the final metric score based on all the updates."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the internal state for fresh computation."""
        pass

class RetrievalMetric(Metric):
    """Base class for retrieval metrics."""
    def __init__(self, name):
        super().__init__(name, metric_type="retrieval")

class GenerationMetric(Metric):
    """Base class for generation metrics."""
    def __init__(self, name):
        super().__init__(name, metric_type="generation")


class RecallAtK(RetrievalMetric):
    """Compute Recall@K for retrieval tasks."""
    def __init__(self, k=5):
        super().__init__(name=f"Recall@{k}")
        self._k = k
        self._correct = 0
        self._total = 0

    @property
    def k(self):
        return self._k
    
    @property
    def correct(self):
        return self._correct
    
    @property
    def total(self):
        return self._total
    
    def __str__(self):
        return f"{self._name}: {self.compute()}"
    
    def __repr__(self):
        return str(self)
    
    def update(
            self, 
            predictions: List[List[str]], 
            references: List[List[str]]):
        """
        predictions: list of list of retrieved doc IDs, or any representation
                     of top-K items for each query, e.g. [[id1, id2, ...], ...]
        references: list of list of ground-truth doc IDs, e.g. [[idX1, idX2], [idY1, idY2], ...]
        """
        for pred_list, ref_list in zip(predictions, references):
            self._total += len(ref_list)
            ground_truth = set(ref_list)
            top_k = set(pred_list[:self._k])
            self._correct += len(ground_truth.intersection(top_k))

    def compute(self):
        """Compute final Recall@K."""
        if self._total == 0:
            return 0.0
        return self._correct / self._total

    def reset(self):
        self._correct = 0
        self._total = 0

class PrecisionAtK(RetrievalMetric):
    """Compute Precision@K for retrieval tasks."""
    def __init__(self, k=5):
        super().__init__(name=f"Precision@{k}")
        self._k = k
        self._correct = 0
        self._total = 0

    @property
    def k(self):
        return self._k
    
    @property
    def correct(self):
        return self._correct
    
    @property
    def total(self):
        return self._total
    
    def __str__(self):
        return f"{self.name}: {self.compute()}"
    
    def __repr__(self):
        return str(self)
    
    def update(
            self, 
            predictions: List[List[str]], 
            references: List[List[str]]):
        """
        predictions: list of list of retrieved doc IDs, or any representation
                     of top-K items for each query, e.g. [[id1, id2, ...], ...]
        references: list of list of ground-truth doc IDs, e.g. [[idX1, idX2], [idY1, idY2], ...]
        """
        for pred_list, ref_list in zip(predictions, references):
            self._total += len(pred_list)
            ground_truth = set(ref_list)
            top_k = set(pred_list[:self._k])
            self._correct += len(ground_truth.intersection(top_k))

    def compute(self):
        if self._total == 0:
            return 0.0
        return self._correct / self._total

    def reset(self):
        self._correct = 0
        self._total = 0

class MRR(RetrievalMetric):
    def __init__(self):
        super().__init__(name="MRR")
        self._reciprocal_ranks = []

    @property
    def reciprocal_ranks(self):
        return self._reciprocal_ranks
    
    def __str__(self):
        return f"{self.name}: {self.compute()}"
    
    def __repr__(self):
        return str(self)

    def update(
            self, 
            predictions:List[List[str]], 
            references:List[List[str]]):
        """
        predictions: list of list of retrieved doc IDs
        references: list of List of ground-truth doc IDs
        """
        for pred_list, ref_list in zip(predictions, references):
            # Find the rank of the correct doc in the predicted list
            ground_truth = set(ref_list)
            for rank, doc_id in enumerate(pred_list, 1):
                if doc_id in ground_truth:
                    self._reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                self._reciprocal_ranks.append(0.0)

    def compute(self):
        if len(self._reciprocal_ranks) == 0:
            return 0.0
        return sum(self._reciprocal_ranks) / len(self._reciprocal_ranks)

    def reset(self):
        self._reciprocal_ranks = []

class nDCG(RetrievalMetric):
    def __init__(self, k=5):
        super().__init__(name=f"nDCG@{k}")
        self._k = k
        self._ndcg_scores = []

    @property
    def k(self):
        return self._k
    
    @property
    def ndcg_scores(self):
        return self._ndcg_scores
    
    def __str__(self):
        return f"{self.name}: {self.compute()}"
    
    def __repr__(self):
        return str(self)

    def update(
            self, 
            predictions: List[List[str]], 
            references: List[List[str]]):
        """
        predictions: list of list of retrieved doc IDs
        references: list of List of ground-truth doc IDs
        """
        for pred_list, ref_list in zip(predictions, references):
            dcg = 0.0
            # Only consider top k
            for rank, doc_id in enumerate(pred_list[: self._k], start=1):
                if doc_id in ref_list:
                    # standard discount factor: log base 2
                    dcg += 1.0 / math.log2(rank + 1)
            
            # IDCG for binary relevance
            ideal_relevant_count = min(len(ref_list), self._k)
            idcg = sum(
                1.0 / math.log2(r + 1)
                for r in range(1, ideal_relevant_count + 1)
            )
            
            ndcg = dcg / idcg if idcg > 0 else 0.0
            self._ndcg_scores.append(ndcg)

    def compute(self):
        if len(self._ndcg_scores) == 0:
            return 0.0
        return sum(self._ndcg_scores) / len(self._ndcg_scores)

    def reset(self):
        self._ndcg_scores = []

class BLEU(GenerationMetric):
    def __init__(self):
        super().__init__(name="BLEU")
        self._bleu_scores = []

    @property
    def bleu_scores(self):
        return self._bleu_scores
    
    def __str__(self):
        return f"{self.name}: {self.compute()}"
    
    def __repr__(self):
        return str(self)
    
    def update(
            self,
            predictions: List[str],
            references: List[List[str]]):
        """
        predictions: list of generated responses
        references: list of list of ground-truth responses
        """
        try:
            for pred, refs in enumerate(zip(predictions, references)):
                ref_tokens = [ref.split() for ref in refs]
                gen_tokens = pred.split()
                self._bleu_scores.append(sentence_bleu(ref_tokens, gen_tokens))
        except Exception as e:
            print(f"Error in BLEU computation: {e}")
            print(f"Generated: {pred}, References: {refs}")
    def compute(self):
        if len(self._bleu_scores) == 0:
            return 0.0
        return sum(self._bleu_scores) / len(self._bleu_scores)
    
    def reset(self):
        self._bleu_scores = []

class ROUGE(GenerationMetric):
    def __init__(self, type='rougeL'):
        super().__init__(name="ROUGE")
        self._rouge_scores = []
        self._type = type
        self._scorer = rouge_scorer.RougeScorer([self._type], use_stemmer=True)

    @property
    def rouge_scores(self):
        return self._rouge_scores
    
    @property
    def type(self):
        return self._type
    
    def __str__(self):
        return f"{self.name}: {self.compute()}"
    
    def __repr__(self):
        return str(self)
    
    def update(
            self,
            predictions: List[str],
            references: List[List[str]]):
        """
        predictions: list of generated responses
        references: list of list of ground-truth responses
        """
        for pred, refs in zip(predictions, references):
            score_per_example = 0.0
            for ref in refs:
                scores = self._scorer.score(pred, ref)
                score_per_example += scores[self._type].fmeasure
            self._rouge_scores.append(score_per_example / len(refs))

    def compute(self):
        if len(self._rouge_scores) == 0:
            return 0.0
        return sum(self._rouge_scores) / len(self._rouge_scores)
    
    def reset(self):
        self._rouge_scores = []

class BERTScore(GenerationMetric):
    def __init__(
            self,
            model_name_or_path: str = "bert-base-uncased", 
            batch_size: int = 16,
            device: str = "cuda"  # or "cpu"
        ):
        super().__init__(name="BERTScore")
        self._model_name_or_path = model_name_or_path
        self._batch_size = batch_size
        self._device = device

        # Internal buffers to store all predictions/references before compute()
        self._predictions = []
        self._references = []

    @property
    def model_name_or_path(self):
        return self._model_name_or_path
    
    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def device(self):
        return self._device
    
    @property
    def predictions(self):
        return self._predictions
    
    @property
    def references(self):
        return self._references

    def __str__(self):
        return f"{self.name}: {self.compute()}"
    
    def __repr__(self):
        return str(self)
    
    def update(self, predictions, references):
        """
        Args:
            predictions: list of strings (generated texts)
            references: list of strings (reference texts)
        """
        self._predictions.extend(predictions)
        self._references.extend(references)

    def compute(self):
        """
        Returns a dict with {'precision': P, 'recall': R, 'f1': F1}
        for the aggregated predictions/references.
        """

        if not self._predictions:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        P, R, F1 = score(
            cands=self._predictions,
            refs=self._references,
            model_type=self._model_name_or_path,
            batch_size=self._batch_size,
            device=self._device
        )

        return {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item()
        }

    def reset(self):
        self._predictions = []
        self._references = []


class MetricCollection:
    def __init__(self, metrics):
        """
        metrics: dict like {'recall@5': RecallAtK(5), 'bleu': BLEU(), ...}
        """
        self.metrics = metrics

    def update(self, predictions, references, metric_type='all'):
        for metric in self.metrics.values():
            if metric.metric_type == metric_type or metric_type == 'all':
                metric.update(predictions, references)
    
    def compute(self, metric_type='all')->Union[Dict[str, Union[Dict[str, float],float]], Dict[str, Dict[str, Union[Dict[str, float],float]]]]:
        if metric_type == 'all':
            return {
                metric.metric_type: {
                    name: metric.compute()
                    for name, metric in self.metrics.items()
                    if metric.metric_type == metric.metric_type
                }
                for metric in self.metrics.values()
            }
        else:
            return {
                name: metric.compute()
                for name, metric in self.metrics.items()
                if metric.metric_type == metric_type
            }

    def reset(self, metric_type='all'):
        for metric in self.metrics.values():
            if metric.metric_type == metric_type or metric_type == 'all':
                metric.reset()

    def get_metrics_by_type(self, metric_type: str):
        """Return a sub-dict of metrics that match the given type."""
        return {
            name: m
            for name, m in self.metrics.items()
            if m.metric_type == metric_type
        }
