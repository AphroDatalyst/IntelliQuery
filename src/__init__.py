"""
KnowledgeGraph-RAG: Local RAG with Ollama, LangChain, and ChromaDB.
"""

__version__ = "0.1.0"
__author__ = "KnowledgeGraph-RAG Contributors"

from .rag_agent import RAGAgent
from .document_loader import DocumentLoader
from .memory import ConversationMemory
from .retrieval import AdvancedRetriever
from .config import RAGConfig

__all__ = [
    "RAGAgent",
    "DocumentLoader",
    "ConversationMemory",
    "AdvancedRetriever",
    "RAGConfig"
]

