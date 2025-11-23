"""
Advanced retrieval techniques for RAG.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from sentence_transformers import CrossEncoder
import numpy as np

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


class AdvancedRetriever:
    """Advanced retrieval with reranking, hybrid search, and MMR."""
    
    def __init__(
        self,
        vectorstore: VectorStore,
        enable_reranking: bool = False,
        enable_hybrid: bool = False,
        enable_mmr: bool = False,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        mmr_diversity: float = 0.5
    ):
        """
        Initialize advanced retriever.
        
        Args:
            vectorstore: Vector store for semantic search
            enable_reranking: Enable cross-encoder reranking
            enable_hybrid: Enable hybrid semantic + keyword search
            enable_mmr: Enable Maximal Marginal Relevance for diversity
            rerank_model: Model for reranking
            mmr_diversity: MMR diversity parameter (0-1)
        """
        self.vectorstore = vectorstore
        self.enable_reranking = enable_reranking
        self.enable_hybrid = enable_hybrid
        self.enable_mmr = enable_mmr
        self.mmr_diversity = mmr_diversity
        
        # Initialize reranker if needed
        self.reranker = None
        if enable_reranking:
            try:
                self.reranker = CrossEncoder(rerank_model)
            except Exception as e:
                print(f"⚠️  Failed to load reranker: {e}")
                self.enable_reranking = False
        
        # Initialize BM25 if hybrid search enabled
        self.bm25 = None
        self.documents = None
        if enable_hybrid and BM25_AVAILABLE:
            self._build_bm25_index()
    
    def _build_bm25_index(self) -> None:
        """Build BM25 index from documents."""
        try:
            # Get all documents from vectorstore
            if hasattr(self.vectorstore, '_collection'):
                collection = self.vectorstore._collection
                if collection:
                    results = collection.get()
                    if results and 'documents' in results:
                        self.documents = results['documents']
                        tokenized_docs = [doc.split() for doc in self.documents]
                        self.bm25 = BM25Okapi(tokenized_docs)
        except Exception as e:
            print(f"⚠️  Failed to build BM25 index: {e}")
            self.enable_hybrid = False
    
    def retrieve(
        self,
        query: str,
        k: int = 3,
        fetch_k: Optional[int] = None
    ) -> List[Document]:
        """
        Retrieve documents using configured strategies.
        
        Args:
            query: Query string
            k: Number of documents to return
            fetch_k: Number of documents to fetch before reranking/MMR
            
        Returns:
            List of retrieved documents
        """
        fetch_k = fetch_k or (k * 3 if (self.enable_reranking or self.enable_mmr) else k)
        
        # Step 1: Initial retrieval
        if self.enable_hybrid and self.bm25:
            docs = self._hybrid_search(query, fetch_k)
        else:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": fetch_k})
            # Use invoke() for newer LangChain versions, fallback to get_relevant_documents
            try:
                docs = retriever.invoke(query)
            except AttributeError:
                docs = retriever.get_relevant_documents(query)
        
        # Step 2: Reranking
        if self.enable_reranking and self.reranker and len(docs) > k:
            docs = self._rerank(query, docs, k)
        
        # Step 3: MMR for diversity
        if self.enable_mmr and len(docs) > k:
            docs = self._mmr(query, docs, k)
        
        return docs[:k]
    
    def _hybrid_search(self, query: str, k: int) -> List[Document]:
        """Combine semantic and keyword search."""
        # Semantic search
        semantic_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k * 2})
        try:
            semantic_docs = semantic_retriever.invoke(query)
        except AttributeError:
            semantic_docs = semantic_retriever.get_relevant_documents(query)
        
        # For now, return semantic results (full BM25 integration requires document storage)
        # In production, would combine scores: final_score = α * semantic_score + (1-α) * bm25_score
        return semantic_docs[:k]
    
    def _rerank(self, query: str, docs: List[Document], k: int) -> List[Document]:
        """Rerank documents using cross-encoder."""
        if not self.reranker:
            return docs[:k]
        
        # Prepare pairs for reranking
        pairs = [[query, doc.page_content] for doc in docs]
        
        # Get reranking scores
        scores = self.reranker.predict(pairs)
        
        # Sort by scores
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored_docs[:k]]
    
    def _mmr(self, query: str, docs: List[Document], k: int) -> List[Document]:
        """Maximal Marginal Relevance for diverse results."""
        if not docs:
            return []
        
        selected = [docs[0]]  # Start with most relevant
        remaining = docs[1:]
        
        # MMR uses document content similarity
        
        while len(selected) < k and remaining:
            best_score = -float('inf')
            best_idx = 0
            
            for i, doc in enumerate(remaining):
                # Relevance score (simplified - use position as proxy)
                relevance = 1.0 / (i + 1)
                
                # Diversity score (max similarity to selected docs)
                max_sim = 0.0
                for sel_doc in selected:
                    # Simplified similarity (would use embeddings in production)
                    similarity = len(set(doc.page_content.split()) & set(sel_doc.page_content.split())) / \
                                max(len(set(doc.page_content.split())), 1)
                    max_sim = max(max_sim, similarity)
                
                diversity = 1.0 - max_sim
                
                # MMR score
                mmr_score = self.mmr_diversity * relevance + (1 - self.mmr_diversity) * diversity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected

