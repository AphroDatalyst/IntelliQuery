"""
Configuration module for IntelliQuery RAG system.
Centralizes all tunable parameters for easy maintenance and experimentation.
"""

import os
from typing import Dict


# DOCUMENT PROCESSING
PDF_DIR = "./docs"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


# RETRIEVAL SETTINGS
RETRIEVAL_K = 12  # Initial retrieval count
RERANK_K = 4      # Top-k after re-ranking

# OLLAMA MODELS (Local)
EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:latest"
LANGUAGE_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest"
LLM_TEMPERATURE = 0.1  # Deterministic inference


# VECTOR STORE
CHROMA_DIR = "./chroma_db"

# HYBRID SEARCH CONFIGURATION
USE_HYBRID = True             # Enable BM25 + Dense fusion
HYBRID_ALPHA = 0.5            # Balance: 0=pure BM25, 1=pure Dense, 0.5=equal
RRF_K = 60                     # Reciprocal Rank Fusion constant (standard=60)
BM25_K1 = 1.5                  # BM25 term frequency saturation
BM25_B = 0.75                  # BM25 length normalization

# CONFIDENCE & GROUNDING THRESHOLDS
CITATION_MIN_SIM = 0.25           # Minimum similarity for citation attribution
RETRIEVAL_CONFIDENCE_MIN = 0.30   # Minimum confidence to trust retrieval
GROUNDING_THRESHOLD = 0.35        # Minimum score to consider answer grounded


# QUERY PROCESSING
ENABLE_QUERY_NORMALIZATION = True
ENABLE_QUERY_EXPANSION = True
MAX_QUERY_EXPANSIONS = 2

# MEMORY & CONVERSATION
ENABLE_CONVERSATION_MEMORY = True
MAX_HISTORY_LENGTH = 10  # Maximum number of previous Q&A pairs to keep
MEMORY_WINDOW = 5        # Number of recent turns to use for context

# DEBUG & EVALUATION
DEBUG_RETRIEVAL = False  # Set True to see detailed retrieval metrics
ENABLE_EVALUATION = False  # Enable RAGAS evaluation metrics

# LANGSMITH TRACING (Optional)
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

# ABBREVIATION MAPPINGS (Domain-specific)
ABBREVIATION_MAP: Dict[str, str] = {
    # Scientific
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "cnn": "convolutional neural network",
    "rnn": "recurrent neural network",
    "lstm": "long short-term memory",
    "llm": "large language model",
    "bert": "bidirectional encoder representations from transformers",
    "rag": "retrieval augmented generation",
    # General
    "info": "information",
    "db": "database",
    "vs": "versus",
    "approx": "approximately",
    "govt": "government",
    "dept": "department",
}

