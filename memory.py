"""
Conversation memory module for multi-turn dialogue.
Maintains conversation history and provides context for follow-up questions.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from config import MAX_HISTORY_LENGTH, MEMORY_WINDOW, ENABLE_CONVERSATION_MEMORY


@dataclass
class Message:
    """Single message in conversation history."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert message to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ConversationTurn:
    """A single Q&A turn in the conversation."""
    question: str
    answer: str
    context_used: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert turn to dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "context_used": self.context_used,
            "timestamp": self.timestamp.isoformat(),
        }


class ConversationMemory:
    """
    Manages conversation history for multi-turn dialogue.
    
    Maintains:
    - Full conversation history (Q&A pairs)
    - Recent message history (for context)
    - Automatic pruning to prevent overflow
    """
    
    def __init__(
        self,
        max_history_length: int = MAX_HISTORY_LENGTH,
        memory_window: int = MEMORY_WINDOW,
        enabled: bool = ENABLE_CONVERSATION_MEMORY,
    ):
        self.max_history_length = max_history_length
        self.memory_window = memory_window
        self.enabled = enabled
        self.turns: List[ConversationTurn] = []
        self.messages: List[Message] = []
    
    def add_turn(self, question: str, answer: str, context_used: Optional[str] = None):
        """Add a Q&A turn to the conversation history."""
        if not self.enabled:
            return
        
        turn = ConversationTurn(
            question=question,
            answer=answer,
            context_used=context_used,
        )
        self.turns.append(turn)
        
        # Add messages
        self.messages.append(Message(role="user", content=question))
        self.messages.append(Message(role="assistant", content=answer))
        
        # Prune if necessary
        if len(self.turns) > self.max_history_length:
            self.turns = self.turns[-self.max_history_length:]
            self.messages = self.messages[-self.max_history_length * 2:]
    
    def get_recent_context(self, window: Optional[int] = None) -> str:
        """
        Get recent conversation context as formatted string.
        Args:
            window: Number of recent turns to include (defaults to memory_window)
        Returns:
            Formatted conversation history string
        """
        if not self.enabled or not self.turns:
            return ""
        
        window = window or self.memory_window
        recent_turns = self.turns[-window:]
        
        context_parts = []
        for turn in recent_turns:
            context_parts.append(f"Q: {turn.question}")
            context_parts.append(f"A: {turn.answer}")
        
        return "\n".join(context_parts)
    
    def get_recent_messages(self, window: Optional[int] = None) -> List[Message]:
        """Get recent messages for context."""
        if not self.enabled:
            return []
        
        window = window or self.memory_window
        return self.messages[-window * 2:]  # Each turn has 2 messages
    
    def clear(self):
        """Clear all conversation history."""
        self.turns = []
        self.messages = []
    
    def get_history_summary(self) -> Dict:
        """Get summary of conversation history."""
        return {
            "total_turns": len(self.turns),
            "enabled": self.enabled,
            "recent_context": self.get_recent_context(),
        }
    
    def format_for_prompt(self) -> str:
        """Format conversation history for inclusion in prompt."""
        return self.get_recent_context()


class MemoryAwareQueryProcessor:
    """
    Query processor that uses conversation memory to enhance queries.
    
    Handles:
    - Follow-up question resolution
    - Reference resolution (e.g., "What about that?", "Tell me more")
    - Context-aware query expansion
    """
    
    def __init__(self, memory: ConversationMemory):
        self.memory = memory
    
    def process_with_context(self, query: str) -> Dict:
        """
        Process query with conversation context.
        
        Returns:
            Dict with processed query and context information
        """
        # Check if query is a follow-up
        is_followup = self._is_followup(query)
        
        # Get conversation context
        context = self.memory.get_recent_context() if self.memory.enabled else ""
        
        # Enhance query if it's a follow-up
        enhanced_query = query
        if is_followup and context:
            enhanced_query = self._enhance_followup(query, context)
        
        return {
            "original_query": query,
            "enhanced_query": enhanced_query,
            "is_followup": is_followup,
            "conversation_context": context,
        }
    
    def _is_followup(self, query: str) -> bool:
        """Detect if query is a follow-up question."""
        followup_indicators = [
            "what about",
            "tell me more",
            "how about",
            "what else",
            "and",
            "also",
            "what is",
            "who is",
            "where is",
            "when is",
            "why is",
            "that",
            "this",
            "it",
            "they",
            "them",
        ]
        query_lower = query.lower().strip()
        return any(query_lower.startswith(ind) for ind in followup_indicators) or len(query.split()) < 5
    
    def _enhance_followup(self, query: str, context: str) -> str:
        """
        Enhance follow-up query with context.
        
        For very short queries or pronouns, try to infer from context.
        """
        # Simple heuristic: if query is very short, prepend context keywords
        if len(query.split()) <= 3 and context:
            # Extract key terms from recent context
            recent_qa = self.memory.turns[-1] if self.memory.turns else None
            if recent_qa:
                # Use previous question's key terms
                prev_keywords = self._extract_keywords(recent_qa.question)
                if prev_keywords:
                    return f"{query} {prev_keywords}"
        
        return query
    
    def _extract_keywords(self, text: str) -> str:
        """Extract key terms from text (simple implementation)."""
        # Remove common stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how", "why", "when", "where"}
        words = text.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        return " ".join(keywords[:5])  # Top 5 keywords


# Global memory instance (can be replaced with session-specific instances)
_global_memory = ConversationMemory()

def get_memory() -> ConversationMemory:
    """Get global conversation memory instance."""
    return _global_memory

def create_memory() -> ConversationMemory:
    """Create a new conversation memory instance."""
    return ConversationMemory()

