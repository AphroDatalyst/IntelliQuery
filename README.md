# KnowledgeGraph-RAG

A production-ready local AI agent implementation using **Ollama** for running LLMs locally, **LangChain** for orchestration, and **RAG (Retrieval-Augmented Generation)** for enhanced question answering. No API keys required - everything runs on your machine!

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://python.langchain.com/)

## ✨ Features

- 🚀 **Local LLM**: Uses Ollama to run language models locally (no API keys needed!)
- 📚 **RAG Pipeline**: Retrieval-Augmented Generation for context-aware responses
- 📄 **Document Processing**: Load and process PDF, TXT, and Markdown documents
- 🔍 **Automatic OCR**: Automatically detects and processes image-based PDFs with OCR
- 🎯 **Advanced Retrieval**: Reranking, hybrid search (semantic + keyword), and MMR diversity
- 💾 **Conversation Memory**: Multi-turn dialogue support with context management
- 🌐 **Web UI**: Interactive Streamlit interface for easy document Q&A
- 🔌 **REST API**: FastAPI server for programmatic access
- 🔍 **Vector Store**: ChromaDB for efficient similarity search
- 💬 **Interactive CLI**: Command-line interface for querying your documents
- 🎓 **Jupyter Notebook**: Complete end-to-end tutorial and workflow
- 🔧 **Modern LangChain**: Uses latest LCEL (LangChain Expression Language) API

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Web UI](#web-ui)
- [REST API](#rest-api)
- [Project Structure](#project-structure)
- [OCR Support](#ocr-support)
- [Configuration](#configuration)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## 🚀 Installation

### Prerequisites

1. **Python 3.8+**: Make sure you have Python installed
   ```bash
   python --version
   ```

2. **Install Ollama**: 
   - Visit [https://ollama.ai/](https://ollama.ai/) and download Ollama
   - After installation, pull a model:
     ```bash
     ollama pull llama2
     # or
     ollama pull mistral
     # or
     ollama pull llama3
     ```

3. **Install OCR Dependencies** (for image-based PDFs):
   - **Windows**: Download and install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract poppler`
   - **Linux (Ubuntu/Debian)**: 
     ```bash
     sudo apt-get install tesseract-ocr poppler-utils libmagic-dev
     ```
   - **Linux (Fedora)**: 
     ```bash
     sudo dnf install tesseract poppler-utils
     ```

### Install Dependencies

**Note**: For OCR support (automatic processing of image-based PDFs), ensure system dependencies are installed first (see Prerequisites above).

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/KnowledgeGraph-RAG.git
   cd KnowledgeGraph-RAG
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the package** (recommended for development):
   ```bash
   pip install -e .
   ```
   
   Or install dependencies only:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -m src.utils
   ```

## 🎯 Quick Start

### 1. Add Documents

Place your documents in the `docs/` folder:
- PDF files (`.pdf`) - Supports both text-based and image-based PDFs (with automatic OCR)
- Text files (`.txt`)
- Markdown files (`.md`, `.markdown`)

**Note**: Image-based PDFs are automatically detected and processed with OCR. No manual configuration needed!

### 2. Load Documents

**CLI**:
```bash
rag-agent --mode ingest
```

**Web UI**: Use the "Ingest Documents" button in the sidebar

**API**:
```bash
curl -X POST http://localhost:8000/ingest
```

### 3. Query Your Documents

**Web UI** (Recommended):
```bash
rag-ui
# or
streamlit run src/web_ui.py
```

**CLI - Interactive Mode**:
```bash
rag-agent --mode interactive
```

**CLI - Single Query**:
```bash
rag-agent --mode query --question "What is RAG?"
```

**REST API**:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'
```

## 📖 Usage

### Web UI

Launch the interactive web interface:

```bash
rag-ui
# or
streamlit run src/web_ui.py
```

Features:
- 💬 Interactive chat interface
- 📄 Document upload and management
- ⚙️ Configurable settings (model, chunk size, top_k, temperature)
- 🎯 Advanced features toggle (memory, reranking, hybrid search, MMR)
- 📚 Source citation visualization
- 💾 Conversation history

### REST API

Start the FastAPI server:

```bash
rag-api
# or
uvicorn src.api:app --reload
```

API Documentation: http://localhost:8000/docs

**Endpoints**:
- `GET /` - API information
- `GET /health` - Health check
- `POST /query` - Query documents
- `POST /ingest` - Ingest documents
- `POST /upload` - Upload a document file
- `GET /history` - Get conversation history
- `DELETE /history` - Clear conversation history

**Example Query**:
```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"question": "What is RAG?", "top_k": 5}
)
print(response.json())
```

### Command Line Interface

#### 1. Ingest Mode (Load Documents)

```bash
rag-agent --mode ingest
```

#### 2. Query Mode (Single Question)

```bash
rag-agent --mode query --question "Your question here"
```

#### 3. Interactive Mode (Chat)

```bash
rag-agent --mode interactive
```

Commands in interactive mode:
- Type your question and press Enter
- Type `clear` to clear conversation history
- Type `exit` or `quit` to end the session

### Advanced Options

```bash
# Use a different model
rag-agent --model mistral --mode interactive

# Specify custom documents folder
rag-agent --documents /path/to/documents --mode ingest
```

### Python API

```python
from src import RAGAgent

# Initialize agent
agent = RAGAgent(
    model_name="llama2",
    enable_memory=True,
    enable_reranking=True,
    enable_hybrid=True
)

# Load documents
agent.load_documents("docs/")

# Ask questions
result = agent.query("What is the main topic of the document?")
print(result["answer"])
print(f"Sources: {result['sources']}")
```

## 🔍 Advanced Features

### Advanced Retrieval

Enable sophisticated retrieval techniques:

```python
from src import RAGAgent

agent = RAGAgent(
    enable_reranking=True,   # Cross-encoder reranking
    enable_hybrid=True,      # Semantic + keyword search
    enable_mmr=True,         # Diversity in results
    top_k=5
)
```

- **Reranking**: Uses cross-encoder models to rerank initial results for better relevance
- **Hybrid Search**: Combines semantic (vector) and keyword (BM25) search
- **MMR**: Maximal Marginal Relevance for diverse, non-redundant results

### Conversation Memory

Multi-turn dialogue support:

```python
agent = RAGAgent(enable_memory=True)

# First question
result1 = agent.query("What is RAG?")

# Follow-up question (uses context)
result2 = agent.query("How does it work?")

# Clear history
agent.clear_history()
```

## 📁 Project Structure

```
KnowledgeGraph-RAG/
├── src/                    # Source code (flat structure)
│   ├── __init__.py
│   ├── rag_agent.py        # RAG agent implementation
│   ├── document_loader.py  # Document loading
│   ├── retrieval.py        # Advanced retrieval techniques
│   ├── memory.py           # Conversation memory
│   ├── config.py           # Configuration management
│   ├── cli.py              # CLI entry point
│   ├── web_ui.py           # Streamlit web UI
│   └── api.py              # FastAPI REST API
├── docs/                    # Your documents (PDF, TXT, MD)
├── notebooks/              # Jupyter notebooks
├── tests/                  # Test files
├── requirements.txt
├── pyproject.toml
├── README.md
└── LICENSE
```

## 🔍 OCR Support

The system automatically detects and processes image-based PDFs using OCR (Optical Character Recognition).

### How It Works

1. **Automatic Detection**: When loading a PDF, the system first attempts standard text extraction
2. **OCR Fallback**: If insufficient text is found (< 50 characters per page average), it automatically switches to OCR mode
3. **Seamless Processing**: OCR-processed documents are marked with `ocr_used: True` in metadata

### Supported PDF Types

- ✅ **Text-based PDFs**: Fast extraction using PyPDFLoader
- ✅ **Scanned PDFs**: Automatic OCR using UnstructuredPDFLoader
- ✅ **Mixed PDFs**: Text pages processed normally, image pages use OCR

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory:

```bash
OLLAMA_BASE_URL=http://localhost:11434
MODEL_NAME=llama2
TEMPERATURE=0.7
CHUNK_SIZE=1000
TOP_K=3
ENABLE_MEMORY=true
ENABLE_RERANKING=false
```

### Default Settings

- **Model**: `llama2` (can be changed via CLI or .env)
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Retrieval**: Top 3 most similar chunks
- **Temperature**: 0.7 (for LLM generation)

## 💡 Examples

### Example 1: Basic Usage

```python
from src import RAGAgent

agent = RAGAgent(model_name="llama2")
agent.load_documents("docs/")
result = agent.query("What is RAG?")
print(result["answer"])
```

### Example 2: With Advanced Features

```python
from src import RAGAgent

agent = RAGAgent(
    model_name="mistral",
    enable_memory=True,
    enable_reranking=True,
    enable_hybrid=True,
    top_k=5
)
agent.load_documents("docs/")

# Multi-turn conversation
result1 = agent.query("What is machine learning?")
result2 = agent.query("What are its applications?")
```

### Example 3: Using REST API

```python
import requests

# Ingest documents
requests.post("http://localhost:8000/ingest", json={
    "documents_path": "docs",
    "chunk_size": 1000
})

# Query
response = requests.post("http://localhost:8000/query", json={
    "question": "What is RAG?",
    "top_k": 5
})
print(response.json())
```

## 🔧 Troubleshooting

### Ollama Issues

**Problem**: `Ollama not found` or `Cannot connect to Ollama`

**Solutions**:
- Make sure Ollama is installed: [https://ollama.ai/](https://ollama.ai/)
- Verify Ollama is running: `ollama list`
- Check if service is listening: `curl http://localhost:11434/api/tags`

### Import Errors

**Problem**: `ModuleNotFoundError` or import errors

**Solution**:
```bash
pip install -r requirements.txt
# or
pip install --upgrade -r requirements.txt
```

### Vector Store Issues

**Problem**: Vector store errors or corruption

**Solution**:
```bash
# Delete and recreate vector store
rm -rf vectorstore/
rag-agent --mode ingest
```

## 🧪 Testing

Run tests:

```bash
pytest tests/
```

With coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## 🛠️ Technology Stack

- **Ollama**: Local LLM inference
- **LangChain**: Orchestration framework (LCEL)
- **ChromaDB**: Vector database
- **Streamlit**: Web UI
- **FastAPI**: REST API
- **Sentence Transformers**: Embeddings and reranking
- **Python 3.8+**: Core language

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for local LLM inference
- [LangChain](https://python.langchain.com/) for the RAG framework
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Streamlit](https://streamlit.io/) for web UI framework
- [FastAPI](https://fastapi.tiangolo.com/) for API framework

---

**Made with ❤️ for the open-source community**
