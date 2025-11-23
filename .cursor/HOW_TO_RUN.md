# 🚀 How to Run the RAG System

## Quick Start (3 Methods)

### 1. 🌐 Streamlit Web UI (Easiest - Recommended)

**Start the web interface:**
```bash
rag-ui
```

**Or manually:**
```bash
streamlit run src/web_ui.py
```

**What you'll see:**
- Opens in browser at `http://localhost:8501`
- Interactive chat interface
- Document upload button
- Settings panel
- Conversation history

**Steps:**
1. Click "📥 Ingest Documents" in sidebar
2. Type your question in the chat
3. See answer with sources!

---

### 2. 🔌 REST API Server

**Start the API:**
```bash
rag-api
```

**Or manually:**
```bash
uvicorn src.api:app --reload
```

**Access:**
- API: `http://localhost:8000`
- Interactive Docs: `http://localhost:8000/docs` ⭐ (Swagger UI)
- Health: `http://localhost:8000/health`

**Example:**
```bash
# Query via API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'
```

---

### 3. 💻 Command Line

**Interactive mode:**
```bash
rag-agent --mode interactive
```

**Single query:**
```bash
rag-agent --mode query --question "What is RAG?"
```

**Ingest documents:**
```bash
rag-agent --mode ingest
```

---

## 📋 Complete Workflow

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
# or
pip install -e .
```

### Step 2: Start Ollama
```bash
# Check if running
ollama list

# If not, it usually auto-starts
# Pull a model if needed
ollama pull llama2
```

### Step 3: Add Documents
Place your files in `docs/` folder:
- PDF files (`.pdf`) - with automatic OCR!
- Text files (`.txt`)
- Markdown files (`.md`)

### Step 4: Ingest Documents

**Option A - Web UI:**
1. Run `rag-ui`
2. Click "📥 Ingest Documents" in sidebar

**Option B - CLI:**
```bash
rag-agent --mode ingest
```

**Option C - API:**
```bash
curl -X POST http://localhost:8000/ingest
```

### Step 5: Query!

**Web UI:** Just type in the chat box!

**CLI:**
```bash
rag-agent --mode interactive
```

**API:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Your question"}'
```

**Python:**
```python
from src import RAGAgent

agent = RAGAgent(enable_memory=True)
agent.load_documents("docs/")
result = agent.query("What is RAG?")
print(result["answer"])
```

---

## 🎯 Advanced Features

### Enable All Advanced Features

```python
from src import RAGAgent

agent = RAGAgent(
    model_name="llama2",
    enable_memory=True,        # Multi-turn conversations
    enable_reranking=True,     # Better relevance
    enable_hybrid=True,        # Semantic + keyword search
    enable_mmr=True,           # Diverse results
    top_k=5
)
```

---

## 📓 Jupyter Notebook

**Open the demo:**
```bash
jupyter notebook notebooks/demo.ipynb
```

The notebook now includes:
- ✅ New query format (returns dict)
- ✅ Conversation memory examples
- ✅ Advanced retrieval demos
- ✅ Source citations

---

## ✅ All Features Status

**Tier 1 (Must Have) - ✅ 100% COMPLETE:**
1. ✅ Web UI (Streamlit) - `rag-ui`
2. ✅ Advanced Retrieval - Reranking, Hybrid, MMR
3. ✅ Conversation Memory - Multi-turn dialogues
4. ✅ FastAPI REST API - `rag-api`
5. ✅ Basic Testing Suite - `pytest tests/`

**Tier 2 (High Value) - 🟡 50% COMPLETE:**
6. ✅ Evaluation Framework
7. ⏳ Enhanced Chunking (planned)
8. ⏳ Query Enhancement (planned)
9. ⏳ Monitoring/Logging (planned)

**All critical portfolio gaps are addressed!** 🎉

---

## 🔧 Troubleshooting

**Port already in use?**
```bash
# Streamlit on different port
streamlit run src/web_ui.py --server.port 8502

# API on different port
uvicorn src.api:app --reload --port 8001
```

**Ollama not working?**
```bash
ollama list  # Check if running
ollama serve # Start if needed
```

**No documents found?**
- Make sure files are in `docs/` folder
- Run `rag-agent --mode ingest` first

---

## 📊 Quick Reference

| Feature | Command | URL |
|---------|---------|-----|
| Web UI | `rag-ui` | http://localhost:8501 |
| REST API | `rag-api` | http://localhost:8000 |
| API Docs | - | http://localhost:8000/docs |
| CLI | `rag-agent --mode interactive` | - |
| Notebook | `jupyter notebook notebooks/demo.ipynb` | - |

---

**🎉 You're all set! Start with `rag-ui` for the easiest experience!**

