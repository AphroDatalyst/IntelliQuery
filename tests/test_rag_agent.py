"""
Unit tests for RAG Agent.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.rag_agent import RAGAgent
from src.memory import ConversationMemory, Message


class TestRAGAgent:
    """Test cases for RAGAgent."""
    
    @pytest.fixture
    def mock_ollama(self):
        """Mock Ollama LLM."""
        with patch('src.rag_agent.Ollama') as mock:
            mock_instance = Mock()
            mock.return_value = mock_instance
            yield mock_instance
    
    @pytest.fixture
    def mock_embeddings(self):
        """Mock embeddings."""
        with patch('src.rag_agent.OllamaEmbeddings') as mock:
            mock_instance = Mock()
            mock.return_value = mock_instance
            yield mock_instance
    
    @pytest.fixture
    def agent(self, mock_ollama, mock_embeddings):
        """Create RAG agent instance."""
        return RAGAgent(
            model_name="llama2",
            enable_memory=False  # Disable memory for simpler tests
        )
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.model_name == "llama2"
        assert agent.top_k == 3
        assert agent.vectorstore is None
        assert agent.qa_chain is None
    
    def test_agent_with_memory(self, mock_ollama, mock_embeddings):
        """Test agent with memory enabled."""
        agent = RAGAgent(enable_memory=True)
        assert agent.memory is not None
        assert isinstance(agent.memory, ConversationMemory)
    
    def test_load_documents_no_files(self, agent, tmp_path):
        """Test loading documents when no files exist."""
        agent.load_documents(str(tmp_path))
        # Should not raise error, just print warning
    
    @patch('src.rag_agent.Chroma')
    def test_load_documents_success(self, mock_chroma, agent, tmp_path):
        """Test successful document loading."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test document content")
        
        # Mock document loader
        with patch.object(agent.document_loader, 'load_directory') as mock_load:
            from langchain_core.documents import Document
            mock_load.return_value = [
                Document(page_content="Test content", metadata={"source": str(test_file)})
            ]
            
            # Mock Chroma
            mock_chroma_instance = Mock()
            mock_chroma.from_documents.return_value = mock_chroma_instance
            
            agent.load_documents(str(tmp_path))
            
            assert agent.vectorstore is not None
            mock_load.assert_called_once()


class TestConversationMemory:
    """Test cases for ConversationMemory."""
    
    def test_memory_initialization(self):
        """Test memory initializes correctly."""
        memory = ConversationMemory(max_history_length=5)
        assert memory.max_history_length == 5
        assert len(memory.messages) == 0
        assert memory.summary is None
    
    def test_add_messages(self):
        """Test adding messages to memory."""
        memory = ConversationMemory()
        memory.add_user_message("Hello")
        memory.add_assistant_message("Hi there")
        
        assert len(memory.messages) == 2
        assert memory.messages[0].role == "user"
        assert memory.messages[1].role == "assistant"
    
    def test_history_trimming(self):
        """Test history is trimmed when exceeding max length."""
        memory = ConversationMemory(max_history_length=2)
        
        for i in range(10):
            memory.add_user_message(f"Question {i}")
            memory.add_assistant_message(f"Answer {i}")
        
        # Should be trimmed to max_history_length * 2 (4 messages)
        assert len(memory.messages) <= 4
    
    def test_get_conversation_context(self):
        """Test getting formatted conversation context."""
        memory = ConversationMemory()
        memory.add_user_message("What is RAG?")
        memory.add_assistant_message("RAG is Retrieval-Augmented Generation")
        
        context = memory.get_conversation_context()
        assert "What is RAG?" in context
        assert "RAG is Retrieval-Augmented Generation" in context
    
    def test_clear_memory(self):
        """Test clearing memory."""
        memory = ConversationMemory()
        memory.add_user_message("Test")
        memory.clear()
        
        assert len(memory.messages) == 0
        assert memory.summary is None

