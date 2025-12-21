"""
Streamlit UI for IntelliQuery RAG system.
Features persistent chat interface with conversation memory.
"""

import os
import tempfile
import time
from typing import List, Optional

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Import modular components
from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVAL_K,
    RERANK_K,
    EMBEDDING_MODEL,
    LANGUAGE_MODEL,
    LLM_TEMPERATURE,
    CHROMA_DIR,
    USE_HYBRID,
    ENABLE_CONVERSATION_MEMORY,
    DEBUG_RETRIEVAL,
)
from prompts import get_single_turn_prompt, get_multi_turn_prompt
from query_processing import QueryProcessor
from retriever import RetrievalSystem, build_context, sentence_citations
from memory import ConversationMemory, MemoryAwareQueryProcessor
from evaluation import RetrievalConfidenceScorer, AnswerGroundingValidator


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_ollama_connection(model_name: str, max_retries: int = 3, retry_delay: int = 2) -> bool:
    """
    Check if Ollama is accessible and model is available.
    
    Args:
        model_name: Name of the model to check
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
    
    Returns:
        True if connection successful, False otherwise
    """
    from langchain_ollama import OllamaEmbeddings
    
    for attempt in range(max_retries):
        try:
            # Try to create embeddings instance and do a simple test
            test_embeddings = OllamaEmbeddings(model=model_name)
            # Test with a small query
            _ = test_embeddings.embed_query("test")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            else:
                st.warning(f"⚠️ Ollama connection issue (attempt {attempt + 1}/{max_retries}): {str(e)}")
                return False
    return False


def create_embeddings_with_retry(model_name: str, max_retries: int = 3, retry_delay: int = 2):
    """
    Create OllamaEmbeddings with retry logic.
    
    Args:
        model_name: Name of the embedding model
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
    
    Returns:
        OllamaEmbeddings instance
    """
    for attempt in range(max_retries):
        try:
            embeddings = OllamaEmbeddings(model=model_name)
            # Test the connection
            _ = embeddings.embed_query("test connection")
            return embeddings
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            else:
                raise ConnectionError(
                    f"Failed to connect to Ollama after {max_retries} attempts. "
                    f"Make sure Ollama is running and the model '{model_name}' is available. "
                    f"Error: {str(e)}"
                )


def create_llm_with_retry(model_name: str, temperature: float = 0, max_retries: int = 3, retry_delay: int = 2):
    """
    Create ChatOllama with retry logic.
    
    Args:
        model_name: Name of the language model
        temperature: Temperature for generation
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
    
    Returns:
        ChatOllama instance
    """
    for attempt in range(max_retries):
        try:
            llm = ChatOllama(model=model_name, temperature=temperature)
            # Test the connection with a simple prompt
            _ = llm.invoke("test")
            return llm
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            else:
                raise ConnectionError(
                    f"Failed to connect to Ollama after {max_retries} attempts. "
                    f"Make sure Ollama is running and the model '{model_name}' is available. "
                    f"Error: {str(e)}"
                )


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "chain" not in st.session_state:
        st.session_state.chain = None
        st.session_state.embeddings = None
        st.session_state.llm = None
        st.session_state.retrieval_system = None
        st.session_state.vectordb = None
        st.session_state.documents = None
    
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory(enabled=ENABLE_CONVERSATION_MEMORY)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "query_processor" not in st.session_state:
        st.session_state.query_processor = QueryProcessor()
    
    if "memory_processor" not in st.session_state:
        st.session_state.memory_processor = MemoryAwareQueryProcessor(st.session_state.memory)


# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

def build_vectordb_from_uploads(files, embeddings: OllamaEmbeddings) -> tuple:
    """
    Build vector database from uploaded PDF files.
    
    Returns:
        Tuple of (vectordb, documents)
    """
    import shutil
    
    docs: List[Document] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # Clear existing database to avoid mixing with ./docs files
    # Use retry logic for Windows file locking issues
    if os.path.exists(CHROMA_DIR):
        max_retries = 5
        retry_delay = 0.5
        for attempt in range(max_retries):
            try:
                # Close any existing connections first
                if "vectordb" in st.session_state and st.session_state.vectordb is not None:
                    try:
                        # Try to delete the persist directory reference
                        del st.session_state.vectordb
                    except:
                        pass
                
                # Small delay to allow file handles to close
                time.sleep(retry_delay)
                
                # Remove the directory
                shutil.rmtree(CHROMA_DIR)
                break  # Success, exit retry loop
            except (PermissionError, OSError) as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    # Last attempt failed, try to continue anyway
                    # The database will be overwritten
                    st.warning(f"⚠️ Could not fully clear old database: {str(e)}. Continuing anyway...")
                    break
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for uploaded in files:
            file_path = os.path.join(tmpdir, uploaded.name)
            with open(file_path, "wb") as f:
                f.write(uploaded.read())
            loader = PyPDFLoader(file_path)
            loaded = loader.load()
            for d in loaded:
                for i, chunk in enumerate(splitter.split_text(d.page_content)):
                    # Ensure source is set to uploaded filename, not file path
                    source_name = uploaded.name
                    docs.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": source_name,  # Use uploaded filename only
                                "page": d.metadata.get("page"),
                                "chunk": i,
                            },
                        )
                    )
    
    # Build fresh vector database with only uploaded documents
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectordb, docs


# ============================================================================
# RAG CHAIN CONSTRUCTION
# ============================================================================

def make_rag_chain(
    retrieval_system: RetrievalSystem,
    embeddings: OllamaEmbeddings,
    llm: ChatOllama,
    memory: ConversationMemory,
    use_memory: bool = True,
):
    """
    Build RAG chain with optional conversation memory.
    
    Args:
        retrieval_system: Retrieval system instance
        embeddings: Embeddings model
        llm: Language model
        memory: Conversation memory
        use_memory: Whether to use conversation memory
    
    Returns:
        Runnable chain
    """
    query_processor = QueryProcessor()
    memory_processor = MemoryAwareQueryProcessor(memory)
    
    def retrieve_and_prepare(x: dict) -> dict:
        """Retrieve documents and prepare context."""
        question = x["question"]
        
        # Process query with memory awareness
        if use_memory and memory.enabled:
            query_info = memory_processor.process_with_context(question)
            processed_query = query_info["enhanced_query"]
            conversation_context = query_info["conversation_context"]
        else:
            query_info = query_processor.process(question)
            processed_query = query_info["normalized"]
            conversation_context = ""
        
        # Retrieve and re-rank
        reranked_docs, scores = retrieval_system.retrieve_and_rerank(
            processed_query,
            retrieval_k=RETRIEVAL_K,
            rerank_k=RERANK_K,
        )
        
        # Build context
        context = build_context(reranked_docs, show_scores=DEBUG_RETRIEVAL)
        
        return {
            "question": question,
            "processed_query": processed_query,
            "context": context,
            "conversation_context": conversation_context,
            "reranked_docs": reranked_docs,
            "scores": scores,
        }
    
    def generate_answer(x: dict) -> dict:
        """Generate answer using LLM."""
        question = x["question"]
        context = x["context"]
        conversation_context = x.get("conversation_context", "")
        reranked_docs = x["reranked_docs"]
        
        # Select prompt based on memory usage
        if use_memory and conversation_context:
            prompt = get_multi_turn_prompt()
            prompt_input = {
                "question": question,
                "context": context,
                "conversation_history": conversation_context,
            }
        else:
            prompt = get_single_turn_prompt()
            prompt_input = {
                "question": question,
                "context": context,
            }
        
        # Generate answer
        raw_answer = (prompt | llm | StrOutputParser()).invoke(prompt_input)
        
        # Add citations
        cited_answer = sentence_citations(
            raw_answer,
            reranked_docs,
            embeddings,
            min_sim=0.25,
        )
        
        return {
            **x,
            "raw_answer": raw_answer,
            "final_answer": cited_answer,
        }
    
    def post_process(x: dict) -> str:
        """Post-process answer and update memory."""
        final_answer = x["final_answer"]
        question = x["question"]
        context = x["context"]
        
        # Update conversation memory
        if use_memory and memory.enabled:
            memory.add_turn(question, final_answer, context)
        
        return final_answer
    
    # Build chain
    chain = (
        RunnablePassthrough()
        .assign(prep=RunnableLambda(retrieve_and_prepare))
        .assign(
            question=lambda x: x["question"],
            context=lambda x: x["prep"]["context"],
            conversation_context=lambda x: x["prep"]["conversation_context"],
            reranked_docs=lambda x: x["prep"]["reranked_docs"],
            scores=lambda x: x["prep"]["scores"],
        )
        .assign(answer=RunnableLambda(generate_answer))
        .assign(final_answer=lambda x: x["answer"]["final_answer"])
        | RunnableLambda(post_process)
    )
    
    return chain


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="IntelliQuery RAG",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 IntelliQuery - Intelligent RAG System")
    st.caption(
        "Fully local RAG with conversation memory, hybrid search, and citation-grounded answers."
    )
    
    initialize_session_state()

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        model_embed = st.text_input("Embedding model", EMBEDDING_MODEL)
        model_llm = st.text_input("Chat model", LANGUAGE_MODEL)
        top_k = st.slider("Retriever k", 4, 20, RETRIEVAL_K)
        rerank_k = st.slider("Rerank top_k", 2, 10, RERANK_K)
        use_memory = st.checkbox("Enable conversation memory", ENABLE_CONVERSATION_MEMORY)
        use_hybrid = st.checkbox("Enable hybrid search", USE_HYBRID)
        
        st.divider()
        
        st.header("📄 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("🔨 Build Index", type="primary"):
            with st.spinner("Building vector index..."):
                try:
                    # Close any existing database connections first
                    if st.session_state.get("vectordb") is not None:
                        try:
                            # Chroma doesn't have explicit close, but we can delete the reference
                            del st.session_state.vectordb
                        except:
                            pass
                        st.session_state.vectordb = None
                        # Give Windows time to release file handles
                        time.sleep(0.5)
                    
                    # Check Ollama connection first
                    st.info("🔍 Checking Ollama connection...")
                    if not check_ollama_connection(model_embed):
                        st.error(
                            f"❌ Cannot connect to Ollama. Please ensure:\n"
                            f"1. Ollama is running (check with `ollama serve`)\n"
                            f"2. Model '{model_embed}' is available (pull with `ollama pull {model_embed}`)\n"
                            f"3. Your firewall/antivirus isn't blocking the connection"
                        )
                        return
                    
                    # Create embeddings with retry
                    st.info("📊 Initializing embeddings model...")
                    embeddings = create_embeddings_with_retry(model_embed)
                    
                    # Create LLM with retry
                    st.info("🤖 Initializing language model...")
                    llm = create_llm_with_retry(model_llm, temperature=LLM_TEMPERATURE)
                    
                    # Build vector database
                    st.info("📚 Processing documents and building index...")
                    vectordb, documents = build_vectordb_from_uploads(uploaded_files, embeddings)
                    
                    # Initialize retrieval system
                    retrieval_system = RetrievalSystem(
                        vectordb=vectordb,
                        embeddings=embeddings,
                        documents=documents,
                        use_hybrid=use_hybrid,
                    )
                    
                    # Build chain
                    chain = make_rag_chain(
                        retrieval_system,
                        embeddings,
                        llm,
                        st.session_state.memory,
                        use_memory=use_memory,
                    )
                    
                    # Update session state
                    st.session_state.chain = chain
                    st.session_state.embeddings = embeddings
                    st.session_state.llm = llm
                    st.session_state.retrieval_system = retrieval_system
                    st.session_state.vectordb = vectordb
                    st.session_state.documents = documents
                    st.session_state.memory.enabled = use_memory
                    
                    st.success(f"✅ Index ready! {len(documents)} chunks indexed.")
                except ConnectionError as e:
                    st.error(f"❌ Connection Error: {str(e)}")
                    st.info("💡 Troubleshooting tips:\n"
                           "1. Ensure Ollama is running: `ollama serve`\n"
                           "2. Check if models are available: `ollama list`\n"
                           "3. Pull missing models: `ollama pull <model_name>`\n"
                           "4. Check network/firewall settings")
                except Exception as e:
                    st.error(f"❌ Error building index: {str(e)}")
                    import traceback
                    with st.expander("🔍 Show detailed error"):
                        st.code(traceback.format_exc())
        
        # Memory management
        if st.session_state.memory.enabled:
            st.divider()
            st.header("💬 Conversation")
            if st.button("🗑️ Clear History"):
                st.session_state.memory.clear()
                st.session_state.messages = []
                st.rerun()
            
            history_summary = st.session_state.memory.get_history_summary()
            st.caption(f"Turns: {history_summary['total_turns']}")
    
    # Main chat interface
    if not st.session_state.get("chain"):
        st.info("👆 Upload PDFs and click 'Build Index' to start.")
        return

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = st.session_state.chain.invoke({"question": prompt})
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
