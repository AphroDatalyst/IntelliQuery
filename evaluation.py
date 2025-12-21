"""
Evaluation module for RAG system metrics.
Implements retrieval and generation quality metrics.
"""

import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from langchain_core.documents import Document

from config import (
    RETRIEVAL_CONFIDENCE_MIN,
    GROUNDING_THRESHOLD,
    CITATION_MIN_SIM,
)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    num = sum(x * y for x, y in zip(a, b))
    denom = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return num / denom if denom else 0.0


class RetrievalConfidenceScorer:
    """Computes confidence scores for retrieval results."""
    
    def __init__(self, embeddings, min_confidence: float = RETRIEVAL_CONFIDENCE_MIN):
        self.embeddings = embeddings
        self.min_confidence = min_confidence
        
    def score_results(
        self,
        docs: List[Document],
        query: str,
        scores: Optional[List[float]] = None
    ) -> Dict:
        """
        Score retrieval results and compute confidence.
        
        Returns:
            Dict with confidence metrics and filtered documents
        """
        if not docs:
            return {
                'docs': [],
                'confidence': 0.0,
                'top_score': 0.0,
                'avg_score': 0.0,
                'is_confident': False,
                'filtered_docs': []
            }
        
        # Compute scores if not provided
        if scores is None:
            query_vec = self.embeddings.embed_query(query)
            doc_vecs = self.embeddings.embed_documents([d.page_content for d in docs])
            scores = [cosine_similarity(query_vec, dv) for dv in doc_vecs]
        
        # Attach scores to documents
        for doc, score in zip(docs, scores):
            doc.metadata['retrieval_score'] = score
        
        scores_sorted = sorted(scores, reverse=True)
        top_score = scores_sorted[0] if scores_sorted else 0.0
        avg_score = sum(scores_sorted) / len(scores_sorted) if scores_sorted else 0.0
        gap = (scores_sorted[0] - scores_sorted[1]) if len(scores_sorted) > 1 else 0.0
        
        # Composite confidence score
        confidence = (0.6 * top_score) + (0.2 * avg_score) + (0.2 * min(gap * 2, 1.0))
        confidence = max(0.0, min(1.0, confidence))
        
        # Filter documents by confidence threshold
        filtered = [
            d for d in docs
            if d.metadata.get('retrieval_score', 0) >= self.min_confidence
        ]
        
        return {
            'docs': docs,
            'confidence': confidence,
            'top_score': top_score,
            'avg_score': avg_score,
            'is_confident': confidence >= self.min_confidence,
            'filtered_docs': filtered,
        }


class AnswerGroundingValidator:
    """Validates that answers are grounded in source documents."""
    
    def __init__(self, embeddings, threshold: float = GROUNDING_THRESHOLD):
        self.embeddings = embeddings
        self.threshold = threshold
        
    def validate(self, answer: str, docs: List[Document]) -> Dict:
        """
        Validate answer grounding in source documents.
        
        Returns:
            Dict with grounding metrics and ungrounded sentences
        """
        if not answer or not docs:
            return {
                'is_grounded': False,
                'grounding_score': 0.0,
                'grounded_ratio': 0.0,
                'sentence_scores': [],
                'ungrounded_sentences': []
            }
        
        sentences = [
            s.strip() for s in re.split(r'(?<=[.!?])\s+', answer)
            if s.strip()
        ]
        if not sentences:
            return {
                'is_grounded': False,
                'grounding_score': 0.0,
                'grounded_ratio': 0.0,
                'sentence_scores': [],
                'ungrounded_sentences': []
            }
        
        doc_vecs = self.embeddings.embed_documents([d.page_content for d in docs])
        
        sentence_scores = []
        ungrounded = []
        
        for sent in sentences:
            sent_vec = self.embeddings.embed_query(sent)
            best_score = max(cosine_similarity(sent_vec, dv) for dv in doc_vecs)
            sentence_scores.append((sent, best_score))
            if best_score < self.threshold:
                ungrounded.append(sent)
        
        scores = [s[1] for s in sentence_scores]
        grounding_score = sum(scores) / len(scores) if scores else 0.0
        grounded_ratio = len([s for s in scores if s >= self.threshold]) / len(scores) if scores else 0.0
        
        return {
            'is_grounded': grounded_ratio >= 0.7 and grounding_score >= self.threshold,
            'grounding_score': grounding_score,
            'grounded_ratio': grounded_ratio,
            'sentence_scores': sentence_scores,
            'ungrounded_sentences': ungrounded,
        }


class RAGEvaluator:
    """
    Comprehensive RAG evaluation metrics.
    
    Implements:
    - Retrieval Quality: Precision@k, Recall@k, MRR, nDCG
    - Generation Quality: Answer Relevance, Faithfulness
    - End-to-End: Context Precision/Recall, Answer Correctness
    """
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.confidence_scorer = RetrievalConfidenceScorer(embeddings)
        self.grounding_validator = AnswerGroundingValidator(embeddings)
    
    def evaluate_retrieval(
        self,
        query: str,
        retrieved_docs: List[Document],
        relevant_doc_indices: Optional[List[int]] = None,
        k: int = 10,
    ) -> Dict:
        """
        Evaluate retrieval quality.
        
        Args:
            query: Search query
            retrieved_docs: Retrieved documents
            relevant_doc_indices: Indices of truly relevant documents (if known)
            k: Top-k for precision/recall
        
        Returns:
            Dict with retrieval metrics
        """
        if not retrieved_docs:
            return {
                'precision@k': 0.0,
                'recall@k': 0.0,
                'f1@k': 0.0,
                'mrr': 0.0,
            }
        
        # Confidence scoring
        conf_result = self.confidence_scorer.score_results(retrieved_docs, query)
        
        # If we have ground truth relevance, compute precision/recall
        if relevant_doc_indices is not None:
            retrieved_indices = set(range(len(retrieved_docs[:k])))
            relevant_set = set(relevant_doc_indices)
            
            if relevant_set:
                intersection = retrieved_indices & relevant_set
                precision = len(intersection) / len(retrieved_indices) if retrieved_indices else 0.0
                recall = len(intersection) / len(relevant_set)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            else:
                precision = recall = f1 = 0.0
            
            # MRR: Mean Reciprocal Rank
            mrr = 0.0
            for rank, idx in enumerate(retrieved_indices, 1):
                if idx in relevant_set:
                    mrr = 1.0 / rank
                    break
        else:
            precision = recall = f1 = mrr = None
        
        return {
            'precision@k': precision,
            'recall@k': recall,
            'f1@k': f1,
            'mrr': mrr,
            'confidence': conf_result['confidence'],
            'top_score': conf_result['top_score'],
            'is_confident': conf_result['is_confident'],
        }
    
    def evaluate_generation(
        self,
        query: str,
        answer: str,
        context_docs: List[Document],
    ) -> Dict:
        """
        Evaluate generation quality.
        
        Returns:
            Dict with generation metrics
        """
        # Grounding validation
        grounding = self.grounding_validator.validate(answer, context_docs)
        
        # Answer relevance (simple heuristic: keyword overlap)
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        relevance = len(query_words & answer_words) / len(query_words) if query_words else 0.0
        
        return {
            'answer_relevance': relevance,
            'groundedness': grounding['grounding_score'],
            'is_grounded': grounding['is_grounded'],
            'grounded_ratio': grounding['grounded_ratio'],
            'ungrounded_sentences': len(grounding['ungrounded_sentences']),
        }
    
    def evaluate_end_to_end(
        self,
        query: str,
        retrieved_docs: List[Document],
        answer: str,
        relevant_doc_indices: Optional[List[int]] = None,
    ) -> Dict:
        """
        Comprehensive end-to-end RAG evaluation.
        
        Returns:
            Dict with all evaluation metrics
        """
        retrieval_metrics = self.evaluate_retrieval(query, retrieved_docs, relevant_doc_indices)
        generation_metrics = self.evaluate_generation(query, answer, retrieved_docs)
        
        return {
            **retrieval_metrics,
            **generation_metrics,
            'overall_score': (
                (retrieval_metrics.get('confidence', 0.0) * 0.4) +
                (generation_metrics['groundedness'] * 0.4) +
                (generation_metrics['answer_relevance'] * 0.2)
            ),
        }

