"""
Streamlit web UI for RAG Agent.
"""

import streamlit as st
import os
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is in path for imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Use absolute imports that work in both script and module contexts
from src.rag_agent import RAGAgent
from src.config import RAGConfig
from src.retrieval import AdvancedRetriever

# Try to import requests for Ollama API calls
try:
    import requests
except ImportError:
    requests = None


def get_available_models(ollama_base_url: str = "http://localhost:11434") -> List[str]:
    """
    Get list of available Ollama models.
    
    Args:
        ollama_base_url: Base URL for Ollama API
        
    Returns:
        List of available model names (base names without tags)
    """
    if requests is None:
        return []
    
    try:
        response = requests.get(f"{ollama_base_url}/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            # Extract base model names (remove tags like :latest, :7b, etc.)
            model_names = []
            seen = set()
            for m in models:
                name = m.get("name", "")
                if name:
                    # Remove tag suffix (e.g., "llama2:latest" -> "llama2")
                    base_name = name.split(":")[0]
                    if base_name not in seen:
                        model_names.append(base_name)
                        seen.add(base_name)
            return sorted(model_names)
    except Exception:
        pass
    return []


def init_session_state():
    """Initialize session state variables."""
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore_loaded" not in st.session_state:
        st.session_state.vectorstore_loaded = False
    if "available_models" not in st.session_state:
        st.session_state.available_models = []
    if "ollama_base_url" not in st.session_state:
        st.session_state.ollama_base_url = "http://localhost:11434"


def load_agent(
    model_name: str,
    enable_memory: bool,
    enable_reranking: bool,
    enable_hybrid: bool,
    enable_mmr: bool,
    chunk_size: int,
    top_k: int,
    temperature: float,
    ollama_base_url: str = "http://localhost:11434"
) -> Optional[RAGAgent]:
    """
    Initialize or reload RAG agent.
    
    Returns:
        RAGAgent instance or None if initialization fails
    """
    try:
        config = RAGConfig(
            model_name=model_name,
            ollama_base_url=ollama_base_url,
            chunk_size=chunk_size,
            top_k=top_k,
            temperature=temperature,
            enable_memory=enable_memory,
            enable_reranking=enable_reranking
        )
        
        agent = RAGAgent(
            model_name=model_name,
            ollama_base_url=ollama_base_url,
            enable_memory=enable_memory,
            enable_reranking=enable_reranking,
            enable_hybrid=enable_hybrid,
            enable_mmr=enable_mmr,
            chunk_size=chunk_size,
            top_k=top_k,
            config=config
        )
        
        # Try to load existing vectorstore
        if os.path.exists("./vectorstore"):
            try:
                agent.vectorstore = agent.vectorstore.__class__(
                    persist_directory="./vectorstore",
                    embedding_function=agent.embeddings
                )
                agent._create_qa_chain()
                if enable_reranking or enable_hybrid or enable_mmr:
                    agent.advanced_retriever = AdvancedRetriever(
                        vectorstore=agent.vectorstore,
                        enable_reranking=enable_reranking,
                        enable_hybrid=enable_hybrid,
                        enable_mmr=enable_mmr
                    )
                st.session_state.vectorstore_loaded = True
            except Exception as e:
                st.warning(f"⚠️ Could not load existing vectorstore: {e}")
                st.session_state.vectorstore_loaded = False
        
        return agent
    except ValueError as e:
        error_msg = str(e)
        if "not found" in error_msg.lower() or "404" in error_msg:
            st.error(f"❌ Model '{model_name}' not found")
            st.info(f"💡 Run: `ollama pull {model_name}` to install the model")
        else:
            st.error(f"❌ Error initializing agent: {error_msg}")
        return None
    except Exception as e:
        st.error(f"❌ Error initializing agent: {str(e)}")
        return None
    
    # Try to load existing vectorstore
    if os.path.exists("./vectorstore"):
        try:
            agent.vectorstore = agent.vectorstore.__class__(
                persist_directory="./vectorstore",
                embedding_function=agent.embeddings
            )
            agent._create_qa_chain()
            if enable_reranking or enable_hybrid or enable_mmr:
                agent.advanced_retriever = AdvancedRetriever(
                    vectorstore=agent.vectorstore,
                    enable_reranking=enable_reranking,
                    enable_hybrid=enable_hybrid,
                    enable_mmr=enable_mmr
                )
            st.session_state.vectorstore_loaded = True
        except Exception as e:
            st.error(f"Error loading vectorstore: {e}")
            st.session_state.vectorstore_loaded = False
    
    return agent


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Ortho Agent",
        page_icon="🦴",
        layout="wide"
    )
    
    init_session_state()
    
    # Auto-refresh models on startup
    if not st.session_state.available_models:
        with st.spinner("🔍 Detecting available Ollama models..."):
            st.session_state.available_models = get_available_models(st.session_state.ollama_base_url)
    
    st.title("🦴 Ortho Agent")
    st.markdown(
        """
        <div style='font-size: 0.9em; color: #666; margin-top: -10px; margin-bottom: 20px;'>
        <strong>Orthopedic Radiology Intelligence System</strong><br>
        An advanced AI-powered assistant specialized in orthopedic radiology documentation analysis. 
        Leveraging Retrieval-Augmented Generation (RAG) technology, Ortho Agent provides intelligent 
        insights from medical imaging reports, radiological findings, and orthopedic documentation. 
        Designed for healthcare professionals seeking accurate, context-aware analysis of clinical documents.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        st.subheader("🤖 Model Selection")
        available_models = st.session_state.available_models
        
        if available_models:
            # Use available models
            default_models = ["llama2", "llama3", "mistral", "codellama", "phi", "gemma", "qwen"]
            # Add any default models that aren't in available list (for display)
            all_options = sorted(list(set(available_models + default_models)))
            
            # Find default index
            default_index = 0
            if "llama2" in available_models:
                default_index = all_options.index("llama2")
            elif "llama3" in available_models:
                default_index = all_options.index("llama3")
            elif available_models:
                default_index = all_options.index(available_models[0])
            
            model_name = st.selectbox(
                "Select Model",
                all_options,
                index=default_index,
                help="Choose an Ollama model. Models in green are available."
            )
            
            # Show model status
            if model_name in available_models:
                st.success(f"✅ {model_name} is available")
            else:
                st.error(f"❌ {model_name} not found")
                st.info(f"💡 Run: `ollama pull {model_name}` to install")
        else:
            # Fallback to default list if no models detected
            st.warning("⚠️ Cannot connect to Ollama or no models found")
            st.info("💡 Make sure Ollama is running: `ollama serve`")
            
            default_models = ["llama2", "llama3", "mistral", "codellama", "phi"]
            model_name = st.selectbox(
                "Select Model (may not be available)",
                default_models,
                index=0,
                help="Model may not be installed. Check Ollama connection."
            )
            st.error(f"⚠️ {model_name} may not be available. Install with: `ollama pull {model_name}`")
        
        # Advanced features
        st.subheader("Features")
        enable_memory = st.checkbox("Conversation Memory", value=True)
        enable_reranking = st.checkbox("Reranking", value=False)
        enable_hybrid = st.checkbox("Hybrid Search", value=False)
        enable_mmr = st.checkbox("MMR Diversity", value=False)
        
        # Parameters
        st.subheader("Parameters")
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
        top_k = st.slider("Top K", 1, 10, 3)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        
        # Document management
        st.header("📄 Documents")
        
        # File upload option
        st.subheader("📤 Upload Files")
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, or MD files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help="Upload files directly instead of using folder path"
        )
        
        if uploaded_files:
            if st.button("📥 Process Uploaded Files", type="primary"):
                if st.session_state.agent is None:
                    agent = load_agent(
                        model_name, enable_memory, enable_reranking,
                        enable_hybrid, enable_mmr, chunk_size, top_k, temperature,
                        st.session_state.ollama_base_url
                    )
                    if agent is None:
                        st.error("❌ Cannot initialize agent. Check model availability.")
                        st.stop()
                    st.session_state.agent = agent
                
                # Save uploaded files temporarily
                import tempfile
                temp_dir = tempfile.mkdtemp()
                saved_files = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text(f"💾 Saving {len(uploaded_files)} file(s)...")
                    for i, uploaded_file in enumerate(uploaded_files):
                        file_path = Path(temp_dir) / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        saved_files.append(str(file_path))
                        progress_bar.progress((i + 1) / len(uploaded_files) * 0.3)
                    
                    status_text.text("📂 Loading documents...")
                    progress_bar.progress(0.4)
                    
                    # Load all uploaded files
                    documents = []
                    for i, file_path in enumerate(saved_files):
                        try:
                            loaded_docs = st.session_state.agent.document_loader.load_file(file_path)
                            documents.extend(loaded_docs)
                            status_text.text(f"📄 Loaded {i+1}/{len(saved_files)} files...")
                            progress_bar.progress(0.4 + (i + 1) / len(saved_files) * 0.2)
                        except Exception as e:
                            st.warning(f"⚠️ Error loading {Path(file_path).name}: {e}")
                    
                    if not documents:
                        st.error("❌ No documents could be loaded!")
                        st.stop()
                    
                    status_text.text(f"✂️ Splitting {len(documents)} documents into chunks...")
                    progress_bar.progress(0.7)
                    
                    # Split and create vector store
                    chunks = st.session_state.agent.text_splitter.split_documents(documents)
                    
                    status_text.text(f"🔍 Creating vector store from {len(chunks)} chunks...")
                    progress_bar.progress(0.85)
                    
                    from langchain_community.vectorstores import Chroma
                    st.session_state.agent.vectorstore = Chroma.from_documents(
                        documents=chunks,
                        embedding=st.session_state.agent.embeddings,
                        persist_directory=str(st.session_state.agent.persist_directory)
                    )
                    
                    status_text.text("🔗 Initializing QA chain...")
                    progress_bar.progress(0.95)
                    
                    st.session_state.agent._create_qa_chain()
                    
                    if enable_reranking or enable_hybrid or enable_mmr:
                        st.session_state.agent.advanced_retriever = AdvancedRetriever(
                            vectorstore=st.session_state.agent.vectorstore,
                            enable_reranking=enable_reranking,
                            enable_hybrid=enable_hybrid,
                            enable_mmr=enable_mmr
                        )
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Complete!")
                    
                    st.session_state.vectorstore_loaded = True
                    st.success(f"✅ Successfully processed {len(uploaded_files)} file(s) with {len(chunks)} chunks!")
                    
                except Exception as e:
                    st.error(f"❌ Error processing files: {str(e)}")
                    st.exception(e)
                finally:
                    progress_bar.empty()
                    status_text.empty()
                    # Clean up temp files
                    import shutil
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass
        
        st.markdown("---")
        st.subheader("📁 Or Use Folder Path")
        
        # Document path input
        docs_path = st.text_input(
            "Documents Folder Path",
            value=os.getenv("DOCUMENTS_PATH", "docs"),
            help="Path to folder containing PDF, TXT, or MD files (excludes README.md automatically)"
        )
        
        if st.button("🔄 Reload Agent"):
            agent = load_agent(
                model_name, enable_memory, enable_reranking,
                enable_hybrid, enable_mmr, chunk_size, top_k, temperature,
                st.session_state.ollama_base_url
            )
            if agent is not None:
                st.session_state.agent = agent
                st.success("✅ Agent reloaded!")
            else:
                st.error("❌ Failed to reload agent. Check model availability.")
        
        if st.button("📥 Ingest Documents", type="primary"):
            # Validate path
            docs_path_obj = Path(docs_path)
            if not docs_path_obj.exists():
                st.error(f"❌ Path not found: {docs_path}")
                st.info(f"💡 Please provide a valid path to your documents folder")
                st.stop()
            
            # Check if path has any supported files (excluding README.md, LICENSE, etc.)
            supported_extensions = {".pdf", ".txt", ".md", ".markdown"}
            exclude_patterns = ['README.md', 'readme.md', 'LICENSE', 'CHANGELOG.md', 'CONTRIBUTING.md']
            files = [f for f in docs_path_obj.rglob("*") 
                    if f.is_file() and f.suffix.lower() in supported_extensions
                    and not any(f.name == pattern or f.match(pattern) for pattern in exclude_patterns)]
            
            if not files:
                st.warning(f"⚠️ No supported documents found in: {docs_path}")
                st.info("💡 Supported formats: PDF, TXT, MD (README.md and similar files are excluded)")
                st.stop()
            
            st.info(f"📋 Found {len(files)} document(s) to process (excluding README.md, LICENSE, etc.)")
            
            if st.session_state.agent is None:
                agent = load_agent(
                    model_name, enable_memory, enable_reranking,
                    enable_hybrid, enable_mmr, chunk_size, top_k, temperature,
                    st.session_state.ollama_base_url
                )
                if agent is None:
                    st.error("❌ Cannot initialize agent. Check model availability.")
                    st.stop()
                st.session_state.agent = agent
            
            # Progress bar for document ingestion
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("📂 Loading documents...")
                progress_bar.progress(0.2)
                
                # Load documents (exclude README.md and similar files)
                exclude_patterns = ['README.md', 'readme.md', 'LICENSE', 'CHANGELOG.md', 'CONTRIBUTING.md']
                documents = st.session_state.agent.document_loader.load_directory(docs_path, exclude_patterns=exclude_patterns)
                if not documents:
                    st.error("❌ No documents could be loaded!")
                    st.stop()
                
                status_text.text(f"✂️ Splitting {len(documents)} documents into chunks...")
                progress_bar.progress(0.5)
                
                # Split documents
                chunks = st.session_state.agent.text_splitter.split_documents(documents)
                
                status_text.text(f"🔍 Creating vector store from {len(chunks)} chunks...")
                progress_bar.progress(0.7)
                
                # Create vector store
                from langchain_community.vectorstores import Chroma
                st.session_state.agent.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=st.session_state.agent.embeddings,
                    persist_directory=str(st.session_state.agent.persist_directory)
                )
                
                status_text.text("🔗 Initializing QA chain...")
                progress_bar.progress(0.9)
                
                # Create QA chain
                st.session_state.agent._create_qa_chain()
                
                # Initialize advanced retriever if enabled
                if enable_reranking or enable_hybrid or enable_mmr:
                    st.session_state.agent.advanced_retriever = AdvancedRetriever(
                        vectorstore=st.session_state.agent.vectorstore,
                        enable_reranking=enable_reranking,
                        enable_hybrid=enable_hybrid,
                        enable_mmr=enable_mmr
                    )
                
                progress_bar.progress(1.0)
                status_text.text("✅ Complete!")
                
                st.session_state.vectorstore_loaded = True
                st.success(f"✅ Successfully ingested {len(documents)} document(s) with {len(chunks)} chunks!")
                
            except Exception as e:
                st.error(f"❌ Error during ingestion: {str(e)}")
                st.exception(e)
            finally:
                progress_bar.empty()
                status_text.empty()
        
        if st.button("🗑️ Clear History"):
            if st.session_state.agent and st.session_state.agent.memory:
                st.session_state.agent.clear_history()
            st.session_state.messages = []
            st.success("History cleared!")
        
        # Status
        st.header("📊 Status")
        if st.session_state.vectorstore_loaded:
            st.success("✅ Vectorstore loaded")
        else:
            st.warning("⚠️ No vectorstore. Ingest documents first.")
    
    # Initialize agent if not done
    if st.session_state.agent is None:
        agent = load_agent(
            model_name, enable_memory, enable_reranking,
            enable_hybrid, enable_mmr, chunk_size, top_k, temperature,
            st.session_state.ollama_base_url
        )
        if agent is not None:
            st.session_state.agent = agent
        else:
            st.warning("⚠️ Agent not initialized. Please check model availability and reload.")
    
    # Chat interface
    st.header("💬 Chat")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.text(source)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your orthopedic documents..."):
        if st.session_state.agent is None:
            st.error("❌ Agent not initialized. Please reload the agent first.")
        elif not st.session_state.vectorstore_loaded:
            st.warning("⚠️ No documents loaded. Please ingest documents using the '📥 Ingest Documents' button in the sidebar.")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.agent.query(prompt)
                        answer = result["answer"]
                        sources = result.get("sources", [])
                        
                        st.markdown(answer)
                        
                        if sources:
                            with st.expander("📚 Sources"):
                                for source in sources:
                                    st.text(source)
                    except ValueError as e:
                        error_msg = str(e)
                        if "not found" in error_msg.lower() or "404" in error_msg:
                            st.error(f"❌ Model '{model_name}' not found")
                            st.info(f"💡 Run: `ollama pull {model_name}` to install")
                        else:
                            st.error(f"❌ Error: {error_msg}")
                    except Exception as e:
                        st.error(f"❌ Error processing query: {str(e)}")
                        st.info("💡 Try reloading the agent or check if the model is available")
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })


if __name__ == "__main__":
    main()

