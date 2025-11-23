# Quick Start Guide

## 🚀 How to Run the RAG System

### Prerequisites

1. **Install Ollama** and pull a model:
   ```bash
   ollama pull llama2
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   # or
   pip install -e .
   ```

### 🎯 Three Ways to Use the System

## 1. 🌐 Streamlit Web UI (Recommended for Demo)

**Launch the interactive web interface:**

```bash
rag-ui
```

**Or manually:**
```bash
streamlit run src/web_ui.py
```

**Features:**
- 💬 Interactive chat interface
- 📄 Document upload and management
- ⚙️ Settings panel (model, chunk size, temperature)
- 🎯 Advanced features toggle (memory, reranking, hybrid search)
- 📚 Source citations
- 💾 Conversation history

**Access:** Opens in your browser at `http://localhost:8501`

---

## 2. 🔌 REST API Server

**Start the FastAPI server:**

```bash
rag-api
```

**Or manually:**
```bash
uvicorn src.api:app --reload
```

**Access:**
- API: `http://localhost:8000`
- Interactive Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

**Example API calls:**

```bash
# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "top_k": 5}'

# Ingest documents
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents_path": "docs"}'

# Upload file
curl -X POST http://localhost:8000/upload \
  -F "file=@your_document.pdf"
```

---

## 3. 💻 Command Line Interface

**Interactive Mode:**
```bash
rag-agent --mode interactive
```

**Single Query:**
```bash
rag-agent --mode query --question "What is RAG?"
```

**Ingest Documents:**
```bash
rag-agent --mode ingest
```

**With Custom Model:**
```bash
rag-agent --model mistral --mode interactive
```

---

## 4. 📓 Jupyter Notebook

**Open the demo notebook:**
```bash
jupyter notebook notebooks/demo.ipynb
```

**Or:**
```bash
jupyter lab notebooks/demo.ipynb
```

---

## 📋 Complete Workflow Example

### Step 1: Add Documents
Place your PDF, TXT, or MD files in the `docs/` folder

### Step 2: Ingest Documents

**Via Web UI:**
- Click "📥 Ingest Documents" button in sidebar

**Via CLI:**
```bash
rag-agent --mode ingest
```

**Via API:**
```bash
curl -X POST http://localhost:8000/ingest
```

### Step 3: Query Documents

**Via Web UI:**
- Type your question in the chat interface
- See answers with source citations

**Via CLI:**
```bash
rag-agent --mode interactive
```

**Via API:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Your question here"}'
```

**Via Python:**
```python
from src import RAGAgent

agent = RAGAgent(enable_memory=True)
agent.load_documents("docs/")
result = agent.query("What is RAG?")
print(result["answer"])
print(f"Sources: {result['sources']}")
```

---

## 🎯 Advanced Features

### Enable Advanced Retrieval

```python
from src import RAGAgent

agent = RAGAgent(
    enable_reranking=True,   # Better relevance
    enable_hybrid=True,      # Semantic + keyword search
    enable_mmr=True,         # Diverse results
    top_k=5
)
```

### Use Conversation Memory

```python
agent = RAGAgent(enable_memory=True)

# First question
result1 = agent.query("What is RAG?")

# Follow-up (uses context)
result2 = agent.query("How does it work?")
```

---

## 🔧 Troubleshooting

### Ollama Not Running
```bash
# Check if Ollama is running
ollama list

# If not, start it (usually auto-starts)
ollama serve
```

### Port Already in Use
```bash
# Streamlit (default: 8501)
streamlit run src/web_ui.py --server.port 8502

# FastAPI (default: 8000)
uvicorn src.api:app --reload --port 8001
```

### No Documents Found
- Make sure files are in `docs/` folder
- Supported formats: `.pdf`, `.txt`, `.md`
- Run `rag-agent --mode ingest` first

---

## 📊 Feature Status

✅ **Completed:**
- Web UI (Streamlit)
- REST API (FastAPI)
- Conversation Memory
- Advanced Retrieval (Reranking, Hybrid, MMR)
- Automatic OCR
- Evaluation Framework
- Testing Suite
- Configuration Management

🎉 **All Tier 1 features are complete and ready to use!**

