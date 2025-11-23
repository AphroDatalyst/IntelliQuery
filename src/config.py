"""
Configuration management for RAG system.
"""

import os
from dataclasses import dataclass


@dataclass
class RAGConfig:
    """RAG system configuration."""
    # Model settings
    model_name: str = "llama2"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    
    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Retrieval settings
    top_k: int = 3
    retrieval_strategy: str = "semantic"  # semantic, hybrid, mmr
    
    # Vector store
    persist_directory: str = "./vectorstore"
    
    # Documents
    documents_path: str = "docs"
    
    # Memory settings
    max_history_length: int = 10
    enable_memory: bool = True
    
    # Reranking
    enable_reranking: bool = False
    rerank_top_n: int = 5
    
    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Load configuration from environment variables."""
        return cls(
            model_name=os.getenv("MODEL_NAME", "llama2"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            top_k=int(os.getenv("TOP_K", "3")),
            retrieval_strategy=os.getenv("RETRIEVAL_STRATEGY", "semantic"),
            persist_directory=os.getenv("VECTORSTORE_DIR", "./vectorstore"),
            documents_path=os.getenv("DOCUMENTS_PATH", "docs"),
            max_history_length=int(os.getenv("MAX_HISTORY_LENGTH", "10")),
            enable_memory=os.getenv("ENABLE_MEMORY", "true").lower() == "true",
            enable_reranking=os.getenv("ENABLE_RERANKING", "false").lower() == "true",
            rerank_top_n=int(os.getenv("RERANK_TOP_N", "5")),
        )

