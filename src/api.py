"""
FastAPI REST API for RAG Agent.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import sys
from pathlib import Path

# Ensure project root is in path for imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Use absolute imports that work in both script and module contexts
from src.rag_agent import RAGAgent
from src.config import RAGConfig

app = FastAPI(
    title="RAG Agent API",
    description="REST API for Retrieval-Augmented Generation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
agent: Optional[RAGAgent] = None


class QueryRequest(BaseModel):
    """Query request model."""
    question: str
    model_name: Optional[str] = "llama2"
    top_k: Optional[int] = 3
    enable_memory: Optional[bool] = True


class QueryResponse(BaseModel):
    """Query response model."""
    answer: str
    sources: List[str]
    model: str


class IngestRequest(BaseModel):
    """Ingest request model."""
    documents_path: Optional[str] = "docs"
    model_name: Optional[str] = "llama2"
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200


class IngestResponse(BaseModel):
    """Ingest response model."""
    status: str
    documents_loaded: int
    chunks_created: int


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup."""
    global agent
    try:
        config = RAGConfig.from_env()
        agent = RAGAgent(
            model_name=config.model_name,
            enable_memory=config.enable_memory,
            config=config
        )
        
        # Try to load existing vectorstore
        if os.path.exists(config.persist_directory):
            try:
                agent.vectorstore = agent.vectorstore.__class__(
                    persist_directory=config.persist_directory,
                    embedding_function=agent.embeddings
                )
                agent._create_qa_chain()
            except Exception:
                pass
    except Exception as e:
        print(f"Warning: Could not initialize agent: {e}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Agent API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "query": "/query",
            "ingest": "/ingest",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    global agent
    return {
        "status": "healthy",
        "agent_initialized": agent is not None,
        "vectorstore_loaded": agent is not None and agent.vectorstore is not None
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system.
    
    Args:
        request: Query request with question and optional parameters
        
    Returns:
        Query response with answer and sources
    """
    global agent
    
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    if agent.qa_chain is None and not os.path.exists(agent.persist_directory):
        raise HTTPException(
            status_code=400,
            detail="No documents loaded. Please ingest documents first."
        )
    
    try:
        # Update agent settings if needed
        if request.model_name != agent.model_name:
            agent.model_name = request.model_name
            agent.llm = agent.llm.__class__(
                model=request.model_name,
                base_url=agent.ollama_base_url
            )
        
        if request.top_k != agent.top_k:
            agent.top_k = request.top_k
            if agent.retriever:
                agent.retriever = agent.vectorstore.as_retriever(
                    search_kwargs={"k": request.top_k}
                )
        
        # Query
        result = agent.query(request.question)
        
        return QueryResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            model=agent.model_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """
    Ingest documents into the vector store.
    
    Args:
        request: Ingest request with documents path and parameters
        
    Returns:
        Ingest response with status and statistics
    """
    global agent
    
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # Update agent if needed
        if request.model_name != agent.model_name:
            agent.model_name = request.model_name
            agent.llm = agent.llm.__class__(
                model=request.model_name,
                base_url=agent.ollama_base_url
            )
        
        if request.chunk_size != agent.text_splitter._chunk_size:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            agent.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
                length_function=len
            )
        
        # Load documents
        docs_path = Path(request.documents_path)
        if not docs_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Documents path not found: {request.documents_path}"
            )
        
        agent.load_documents(str(docs_path))
        
        # Get statistics
        if agent.vectorstore and hasattr(agent.vectorstore, '_collection'):
            collection = agent.vectorstore._collection
            chunks_count = collection.count() if collection else 0
        else:
            chunks_count = 0
        
        return IngestResponse(
            status="success",
            documents_loaded=len(list(docs_path.rglob("*"))) if docs_path.is_dir() else 1,
            chunks_created=chunks_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting documents: {str(e)}")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a document file.
    
    Args:
        file: Uploaded file
        
    Returns:
        Upload status
    """
    docs_path = Path("docs")
    docs_path.mkdir(exist_ok=True)
    
    file_path = docs_path / file.filename
    
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return {
            "status": "success",
            "filename": file.filename,
            "path": str(file_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@app.get("/history")
async def get_history():
    """Get conversation history."""
    global agent
    
    if agent is None or agent.memory is None:
        return {"messages": []}
    
    messages = agent.memory.get_all_messages()
    return {
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "sources": msg.sources
            }
            for msg in messages
        ]
    }


@app.delete("/history")
async def clear_history():
    """Clear conversation history."""
    global agent
    
    if agent and agent.memory:
        agent.clear_history()
        return {"status": "success", "message": "History cleared"}
    
    return {"status": "success", "message": "No history to clear"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

