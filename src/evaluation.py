"""
Evaluation framework for RAG system.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics."""
    precision: float
    recall: float
    f1: float
    mrr: float  # Mean Reciprocal Rank
    ndcg: float  # Normalized Discounted Cumulative Gain


@dataclass
class GenerationMetrics:
    """Generation evaluation metrics."""
    bleu: Optional[float] = None
    rouge_l: Optional[float] = None
    semantic_similarity: Optional[float] = None


class RAGEvaluator:
    """Evaluator for RAG system performance."""
    
    def evaluate_retrieval(
        self,
        retrieved_docs: List[Any],
        relevant_docs: List[Any],
        k: Optional[int] = None
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval performance.
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs
            k: Cutoff for evaluation (None = all retrieved)
            
        Returns:
            RetrievalMetrics object
        """
        if k:
            retrieved_docs = retrieved_docs[:k]
        
        if not retrieved_docs:
            return RetrievalMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Convert to sets for easier comparison
        retrieved_set = set(retrieved_docs)
        relevant_set = set(relevant_docs)
        
        # Precision: relevant retrieved / total retrieved
        if len(retrieved_set) == 0:
            precision = 0.0
        else:
            precision = len(retrieved_set & relevant_set) / len(retrieved_set)
        
        # Recall: relevant retrieved / total relevant
        if len(relevant_set) == 0:
            recall = 0.0
        else:
            recall = len(retrieved_set & relevant_set) / len(relevant_set)
        
        # F1 score
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        # MRR: Mean Reciprocal Rank
        mrr = 0.0
        for i, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_set:
                mrr = 1.0 / i
                break
        
        # NDCG: Normalized Discounted Cumulative Gain (simplified)
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_set:
                dcg += 1.0 / np.log2(i + 1)
        
        # Ideal DCG
        ideal_dcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant_set), len(retrieved_docs)) + 1))
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        
        return RetrievalMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            mrr=mrr,
            ndcg=ndcg
        )
    
    def evaluate_generation(
        self,
        generated: str,
        reference: str
    ) -> GenerationMetrics:
        """
        Evaluate generation quality.
        
        Args:
            generated: Generated text
            reference: Reference/ground truth text
            
        Returns:
            GenerationMetrics object
        """
        # Simple semantic similarity (would use embeddings in production)
        # For now, return basic metrics
        
        # BLEU score (simplified - would use nltk.translate.bleu_score)
        generated_tokens = generated.lower().split()
        reference_tokens = reference.lower().split()
        
        if not generated_tokens or not reference_tokens:
            bleu = 0.0
        else:
            # Simple unigram precision
            matches = len(set(generated_tokens) & set(reference_tokens))
            bleu = matches / len(generated_tokens) if generated_tokens else 0.0
        
        # ROUGE-L (simplified - longest common subsequence)
        rouge_l = self._rouge_l(generated_tokens, reference_tokens)
        
        return GenerationMetrics(
            bleu=bleu,
            rouge_l=rouge_l,
            semantic_similarity=None  # Would use sentence transformers
        )
    
    def _rouge_l(self, generated: List[str], reference: List[str]) -> float:
        """Calculate ROUGE-L score (simplified)."""
        if not generated or not reference:
            return 0.0
        
        # Longest common subsequence
        lcs = self._lcs(generated, reference)
        precision = lcs / len(generated) if generated else 0.0
        recall = lcs / len(reference) if reference else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def _lcs(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def evaluate_end_to_end(
        self,
        questions: List[str],
        ground_truths: List[Dict[str, Any]],
        agent,
        k: int = 3
    ) -> Dict[str, Any]:
        """
        Evaluate end-to-end RAG performance.
        
        Args:
            questions: List of test questions
            ground_truths: List of ground truth dicts with 'answer' and 'relevant_docs'
            agent: RAGAgent instance
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with evaluation results
        """
        retrieval_metrics = []
        generation_metrics = []
        
        for question, gt in zip(questions, ground_truths):
            # Query agent
            result = agent.query(question)
            
            # Evaluate retrieval
            retrieved_sources = result.get("sources", [])
            relevant_sources = gt.get("relevant_docs", [])
            ret_metrics = self.evaluate_retrieval(retrieved_sources, relevant_sources, k=k)
            retrieval_metrics.append(ret_metrics)
            
            # Evaluate generation
            generated_answer = result.get("answer", "")
            reference_answer = gt.get("answer", "")
            gen_metrics = self.evaluate_generation(generated_answer, reference_answer)
            generation_metrics.append(gen_metrics)
        
        # Aggregate metrics
        avg_retrieval = RetrievalMetrics(
            precision=np.mean([m.precision for m in retrieval_metrics]),
            recall=np.mean([m.recall for m in retrieval_metrics]),
            f1=np.mean([m.f1 for m in retrieval_metrics]),
            mrr=np.mean([m.mrr for m in retrieval_metrics]),
            ndcg=np.mean([m.ndcg for m in retrieval_metrics])
        )
        
        avg_generation = GenerationMetrics(
            bleu=np.mean([m.bleu for m in generation_metrics if m.bleu]),
            rouge_l=np.mean([m.rouge_l for m in generation_metrics if m.rouge_l]),
            semantic_similarity=None
        )
        
        return {
            "retrieval": avg_retrieval,
            "generation": avg_generation,
            "num_questions": len(questions)
        }

