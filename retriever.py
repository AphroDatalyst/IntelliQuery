"""
Retrieval module: Dense, BM25, and Hybrid search implementations.
Handles all retrieval logic including re-ranking and fusion.
"""

import os
import re
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from config import (
    USE_HYBRID,
    HYBRID_ALPHA,
    RRF_K,
    BM25_K1,
    BM25_B,
    RETRIEVAL_K,
    RERANK_K,
    DEBUG_RETRIEVAL,
)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    num = sum(x * y for x, y in zip(a, b))
    denom = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return num / denom if denom else 0.0


class BM25Index:
    """
    BM25 (Okapi BM25) lexical search index.
    
    BM25 is a probabilistic retrieval function that ranks documents based on 
    query term frequency while accounting for document length normalization.
    """
    
    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1  # Term frequency saturation parameter
        self.b = b    # Length normalization parameter
        
        self.corpus: List[Document] = []
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.term_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.N: int = 0  # Total documents
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer with lowercasing."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def fit(self, documents: List[Document]) -> 'BM25Index':
        """Build the BM25 index from documents."""
        self.corpus = documents
        self.N = len(documents)
        self.term_freqs = []
        self.doc_lengths = []
        
        # First pass: compute term frequencies and document frequencies
        for doc in documents:
            tokens = self._tokenize(doc.page_content)
            self.doc_lengths.append(len(tokens))
            
            # Term frequency for this document
            tf = defaultdict(int)
            for token in tokens:
                tf[token] += 1
            self.term_freqs.append(dict(tf))
            
            # Update document frequencies
            for term in tf.keys():
                self.doc_freqs[term] += 1
        
        # Compute average document length
        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 0
        
        # Compute IDF for all terms
        for term, df in self.doc_freqs.items():
            # IDF with smoothing to avoid log(0)
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
        
        return self
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[float, int, Document]]:
        """
        Search the index and return top-k results.
        
        Returns: List of (score, doc_index, document) tuples, sorted by score descending.
        """
        query_tokens = self._tokenize(query)
        scores = []
        
        for idx, doc in enumerate(self.corpus):
            score = 0.0
            doc_len = self.doc_lengths[idx]
            tf_doc = self.term_freqs[idx]
            
            for term in query_tokens:
                if term not in self.idf:
                    continue
                    
                tf = tf_doc.get(term, 0)
                idf = self.idf[term]
                
                # BM25 scoring formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * (numerator / denominator) if denominator > 0 else 0
            
            scores.append((score, idx, doc))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]
    
    def get_scores(self, query: str) -> List[float]:
        """Return BM25 scores for all documents (for fusion)."""
        query_tokens = self._tokenize(query)
        scores = []
        
        for idx in range(self.N):
            score = 0.0
            doc_len = self.doc_lengths[idx]
            tf_doc = self.term_freqs[idx]
            
            for term in query_tokens:
                if term not in self.idf:
                    continue
                tf = tf_doc.get(term, 0)
                idf = self.idf[term]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * (numerator / denominator) if denominator > 0 else 0
            
            scores.append(score)
        
        return scores


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 (lexical) and Dense (semantic) search.
    
    Fusion methods:
      1. Reciprocal Rank Fusion (RRF) - rank-based, parameter-free
      2. Linear Combination - weighted score combination
    """
    
    def __init__(
        self,
        dense_retriever,
        bm25_index: BM25Index,
        embeddings: OllamaEmbeddings,
        alpha: float = HYBRID_ALPHA,
        rrf_k: int = RRF_K,
    ):
        self.dense_retriever = dense_retriever
        self.bm25_index = bm25_index
        self.embeddings = embeddings
        self.alpha = alpha
        self.rrf_k = rrf_k
        
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Document],
        bm25_results: List[Tuple[float, int, Document]],
        top_k: int,
    ) -> List[Document]:
        """Combine rankings using Reciprocal Rank Fusion."""
        doc_scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, Document] = {}
        
        # Score from dense retrieval
        for rank, doc in enumerate(dense_results):
            doc_id = hash(doc.page_content[:100])
            doc_scores[doc_id] += self.alpha * (1.0 / (self.rrf_k + rank + 1))
            doc_map[doc_id] = doc
        
        # Score from BM25 retrieval
        for rank, (score, idx, doc) in enumerate(bm25_results):
            doc_id = hash(doc.page_content[:100])
            doc_scores[doc_id] += (1 - self.alpha) * (1.0 / (self.rrf_k + rank + 1))
            doc_map[doc_id] = doc
        
        # Sort by combined RRF score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top-k documents with RRF scores in metadata
        results = []
        for doc_id, rrf_score in sorted_docs[:top_k]:
            doc = doc_map[doc_id]
            doc.metadata['rrf_score'] = rrf_score
            results.append(doc)
        
        return results
    
    def retrieve(self, query: str, top_k: int = RETRIEVAL_K, method: str = "rrf") -> List[Document]:
        """
        Hybrid retrieval with configurable fusion method.
        
        Args:
            query: Search query
            top_k: Number of results to return
            method: "rrf" for Reciprocal Rank Fusion
        """
        # Get results from both retrievers
        dense_results = self.dense_retriever.invoke(query)
        bm25_results = self.bm25_index.search(query, top_k=top_k * 2)
        
        if DEBUG_RETRIEVAL:
            print(f"[Hybrid] Dense: {len(dense_results)} docs, BM25: {len(bm25_results)} docs")
        
        return self._reciprocal_rank_fusion(dense_results, bm25_results, top_k)


def rerank_documents(
    docs: List[Document],
    query: str,
    embeddings: OllamaEmbeddings,
    top_k: int = RERANK_K,
) -> Tuple[List[Document], List[float]]:
    """
    Re-rank documents using cosine similarity.
    
    Returns:
        Tuple of (reranked_documents, scores)
    """
    if not docs:
        return [], []
    
    query_vec = embeddings.embed_query(query)
    doc_vecs = embeddings.embed_documents([d.page_content for d in docs])
    
    scored = [(cosine_similarity(query_vec, vec), d) for d, vec in zip(docs, doc_vecs)]
    scored.sort(key=lambda x: x[0], reverse=True)
    
    top_results = scored[:top_k]
    return [d for _, d in top_results], [s for s, _ in top_results]


def build_context(docs: List[Document], show_scores: bool = False) -> str:
    """Format documents into context block with optional score display."""
    if not docs:
        return ""
    
    blocks = []
    for idx, d in enumerate(docs):
        meta = d.metadata
        source = meta.get("source", "unknown")
        # Extract just the filename if source is a path
        if isinstance(source, str) and os.sep in source:
            source = os.path.basename(source)
        loc = meta.get("page")
        loc_str = f"page {loc}" if loc is not None else f"chunk {meta.get('chunk', '?')}"
        
        score_str = ""
        if show_scores:
            if 'rrf_score' in meta:
                score_str = f" [RRF: {meta['rrf_score']:.4f}]"
            elif 'retrieval_score' in meta:
                score_str = f" [sim: {meta['retrieval_score']:.2f}]"
        
        blocks.append(f"[{idx}] Source: {source}, {loc_str}{score_str}\n{d.page_content}")
    
    return "\n\n".join(blocks)


def sentence_citations(
    answer: str,
    docs: List[Document],
    embeddings: OllamaEmbeddings,
    min_sim: float = 0.25,
) -> str:
    """Add sentence-level citations with confidence indicators."""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip()]
    
    if not docs:
        return "Not found in the provided documents." if not sentences else "\n".join(
            f"{s} (source: Not found in the provided documents.)" for s in sentences
        )
    
    doc_vectors = embeddings.embed_documents([d.page_content for d in docs])
    results = []
    
    for s in sentences:
        s_vec = embeddings.embed_query(s)
        best_idx, best_score = -1, -1.0
        for i, vec in enumerate(doc_vectors):
            score = cosine_similarity(s_vec, vec)
            if score > best_score:
                best_idx, best_score = i, score
        
        if best_idx == -1 or best_score < min_sim:
            results.append(f"{s} (source: Not found in the provided documents.)")
            continue
        
        meta = docs[best_idx].metadata
        source = meta.get("source", "unknown")
        # Extract just the filename if source is a path
        if isinstance(source, str) and os.sep in source:
            source = os.path.basename(source)
        loc = meta.get("page")
        loc_str = f"page {loc}" if loc is not None else f"chunk {meta.get('chunk', '?')}"
        results.append(f"{s} (source: {source}, {loc_str})")
    
    return "\n".join(results) if results else "Not found in the provided documents."


class RetrievalSystem:
    """Main retrieval system orchestrator."""
    
    def __init__(
        self,
        vectordb: Chroma,
        embeddings: OllamaEmbeddings,
        documents: List[Document],
        use_hybrid: bool = USE_HYBRID,
    ):
        self.vectordb = vectordb
        self.embeddings = embeddings
        self.documents = documents
        self.use_hybrid = use_hybrid
        
        # Initialize dense retriever
        self.dense_retriever = vectordb.as_retriever(search_kwargs={"k": RETRIEVAL_K})
        
        # Initialize BM25 if hybrid is enabled
        self.bm25_index = None
        self.hybrid_retriever = None
        if use_hybrid:
            self.bm25_index = BM25Index().fit(documents)
            self.hybrid_retriever = HybridRetriever(
                dense_retriever=self.dense_retriever,
                bm25_index=self.bm25_index,
                embeddings=embeddings,
            )
    
    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_K,
    ) -> List[Document]:
        """Retrieve documents for a query."""
        if self.use_hybrid and self.hybrid_retriever:
            return self.hybrid_retriever.retrieve(query, top_k=top_k)
        else:
            return self.dense_retriever.invoke(query)
    
    def retrieve_and_rerank(
        self,
        query: str,
        retrieval_k: int = RETRIEVAL_K,
        rerank_k: int = RERANK_K,
    ) -> Tuple[List[Document], List[float]]:
        """Retrieve and re-rank documents."""
        initial_docs = self.retrieve(query, top_k=retrieval_k)
        return rerank_documents(initial_docs, query, self.embeddings, top_k=rerank_k)

