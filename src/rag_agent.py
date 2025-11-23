"""
RAG Agent - Retrieval-Augmented Generation with Ollama and LangChain.

Modern implementation using LCEL for clean, composable chains.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from .document_loader import DocumentLoader
from .memory import ConversationMemory
from .retrieval import AdvancedRetriever
from .config import RAGConfig


class RAGAgent:
    """RAG Agent combining document retrieval with LLM generation."""
    
    def __init__(
        self,
        model_name: str = "llama2",
        ollama_base_url: str = "http://localhost:11434",
        persist_directory: str = "./vectorstore",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 3,
        enable_memory: bool = True,
        enable_reranking: bool = False,
        enable_hybrid: bool = False,
        enable_mmr: bool = False,
        config: Optional[RAGConfig] = None
    ) -> None:
        """
        Initialize RAG Agent.
        
        Args:
            model_name: Ollama model name
            ollama_base_url: Ollama API base URL
            persist_directory: Vector store persistence directory
            chunk_size: Document chunk size
            chunk_overlap: Chunk overlap size
            top_k: Number of documents to retrieve
            enable_memory: Enable conversation memory
            enable_reranking: Enable reranking
            enable_hybrid: Enable hybrid search
            enable_mmr: Enable MMR diversity
            config: Optional RAGConfig object
        """
        # Use config if provided, otherwise use parameters
        if config:
            self.config = config
            model_name = config.model_name
            ollama_base_url = config.ollama_base_url
            persist_directory = config.persist_directory
            chunk_size = config.chunk_size
            chunk_overlap = config.chunk_overlap
            top_k = config.top_k
            enable_memory = config.enable_memory
            enable_reranking = config.enable_reranking
        else:
            self.config = None
        
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.persist_directory = persist_directory
        self.top_k = top_k
        
        # Initialize components
        temperature = self.config.temperature if self.config else 0.7
        self.llm = Ollama(model=model_name, base_url=ollama_base_url, temperature=temperature)
        self.embeddings = OllamaEmbeddings(model=model_name, base_url=ollama_base_url)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        self.document_loader = DocumentLoader()
        
        # Memory
        max_history = self.config.max_history_length if self.config else 10
        self.memory = ConversationMemory(max_history_length=max_history) if enable_memory else None
        
        # Will be initialized when documents are loaded
        self.vectorstore: Optional[Chroma] = None
        self.qa_chain = None
        self.retriever = None
        self.advanced_retriever: Optional[AdvancedRetriever] = None
        self.enable_reranking = enable_reranking
        self.enable_hybrid = enable_hybrid
        self.enable_mmr = enable_mmr
        
    def load_documents(self, documents_path: str) -> None:
        """Load documents and create vector store."""
        print(f"📂 Loading documents from: {documents_path}")
        
        documents = self.document_loader.load_directory(documents_path)
        if not documents:
            print("⚠️  No documents found!")
            return
        
        print(f"📄 Loaded {len(documents)} documents")
        print("✂️  Splitting documents...")
        
        chunks = self.text_splitter.split_documents(documents)
        print(f"📝 Created {len(chunks)} chunks")
        
        print("🔍 Creating vector store...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        self._create_qa_chain()
        
        # Initialize advanced retriever if enabled
        if self.enable_reranking or self.enable_hybrid or self.enable_mmr:
            self.advanced_retriever = AdvancedRetriever(
                vectorstore=self.vectorstore,
                enable_reranking=self.enable_reranking,
                enable_hybrid=self.enable_hybrid,
                enable_mmr=self.enable_mmr
            )
        
        print("✅ Vector store created!")
    
    def _create_qa_chain(self) -> None:
        """Create QA chain using LCEL."""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Load documents first.")
        
        # Build prompt with optional conversation history
        if self.memory:
            prompt_template = """Use the following context and conversation history to answer the question.
If you don't know the answer, say so. Don't make up an answer.

Previous conversation:
{history}

Context from documents:
{context}

Question: {question}

Answer:"""
        else:
            prompt_template = """Use the following context to answer the question.
If you don't know the answer, say so. Don't make up an answer.

Context: {context}

Question: {question}

Answer:"""
        
        if self.memory:
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["history", "context", "question"]
            )
        else:
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        def create_chain_input(question: str) -> Dict[str, Any]:
            # Use advanced retriever if available
            if self.advanced_retriever:
                docs = self.advanced_retriever.retrieve(question, k=self.top_k)
            else:
                # Use invoke() for newer LangChain versions, fallback to get_relevant_documents
                try:
                    docs = self.retriever.invoke(question)
                except AttributeError:
                    # Fallback for older LangChain versions
                    docs = self.retriever.get_relevant_documents(question)
            
            context = format_docs(docs)
            
            if self.memory:
                history = self.memory.get_conversation_context()
                return {"history": history, "context": context, "question": question}
            else:
                return {"context": context, "question": question}
        
        self.qa_chain = (
            RunnableLambda(create_chain_input)
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def query(self, question: str, return_sources: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: Question to ask
            return_sources: Whether to include source information
            
        Returns:
            Dictionary with 'answer' and optionally 'sources'
        """
        if self.qa_chain is None:
            if os.path.exists(self.persist_directory):
                try:
                    print("📂 Loading existing vector store...")
                    self.vectorstore = Chroma(
                        persist_directory=self.persist_directory,
                        embedding_function=self.embeddings
                    )
                    self._create_qa_chain()
                    if self.enable_reranking or self.enable_hybrid or self.enable_mmr:
                        self.advanced_retriever = AdvancedRetriever(
                            vectorstore=self.vectorstore,
                            enable_reranking=self.enable_reranking,
                            enable_hybrid=self.enable_hybrid,
                            enable_mmr=self.enable_mmr
                        )
                except Exception as e:
                    return {"answer": f"❌ Error loading vector store: {e}", "sources": []}
            else:
                return {"answer": "❌ No documents loaded. Use --mode ingest first.", "sources": []}
        
        # Add to memory if enabled
        if self.memory:
            self.memory.add_user_message(question)
        
        try:
            answer = self.qa_chain.invoke(question)
            
            # Get sources
            sources = []
            if return_sources:
                try:
                    if self.advanced_retriever:
                        source_docs = self.advanced_retriever.retrieve(question, k=self.top_k)
                    else:
                        try:
                            source_docs = self.retriever.invoke(question)
                        except AttributeError:
                            source_docs = self.retriever.get_relevant_documents(question)
                    
                    sources = [
                        Path(doc.metadata.get("source", "")).name
                        for doc in source_docs
                        if doc.metadata.get("source")
                    ]
                    sources = list(set(sources))  # Deduplicate
                except Exception:
                    pass
            
            # Add to memory if enabled
            if self.memory:
                self.memory.add_assistant_message(answer, sources)
            
            return {"answer": answer, "sources": sources}
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            if self.memory:
                self.memory.add_assistant_message(error_msg)
            return {"answer": error_msg, "sources": []}
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        if self.memory:
            self.memory.clear()

