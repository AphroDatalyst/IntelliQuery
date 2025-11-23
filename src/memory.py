"""
Conversation memory management for multi-turn dialogues.
"""

from typing import List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Message:
    """Single message in conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    sources: Optional[List[str]] = None


class ConversationMemory:
    """Manages conversation history with context window management."""
    
    def __init__(self, max_history_length: int = 10):
        """
        Initialize conversation memory.
        
        Args:
            max_history_length: Maximum number of exchanges to keep
        """
        self.max_history_length = max_history_length
        self.messages: List[Message] = []
        self.summary: Optional[str] = None
    
    def add_user_message(self, content: str) -> None:
        """Add user message to history."""
        self.messages.append(Message(role="user", content=content))
        self._trim_history()
    
    def add_assistant_message(self, content: str, sources: Optional[List[str]] = None) -> None:
        """Add assistant response to history."""
        self.messages.append(Message(role="assistant", content=content, sources=sources))
        self._trim_history()
    
    def _trim_history(self) -> None:
        """Trim history to max length, keeping most recent messages."""
        if len(self.messages) > self.max_history_length * 2:  # *2 because each exchange = 2 messages
            # Keep most recent messages
            self.messages = self.messages[-self.max_history_length * 2:]
    
    def get_conversation_context(self, include_recent: int = 3) -> str:
        """
        Get formatted conversation context.
        
        Args:
            include_recent: Number of recent exchanges to include
            
        Returns:
            Formatted conversation history string
        """
        if not self.messages:
            return ""
        
        # Get recent messages
        recent_messages = self.messages[-include_recent * 2:] if include_recent > 0 else []
        
        context_parts = []
        if self.summary:
            context_parts.append(f"Previous conversation summary: {self.summary}\n")
        
        for msg in recent_messages:
            role_label = "Human" if msg.role == "user" else "Assistant"
            context_parts.append(f"{role_label}: {msg.content}")
            if msg.sources and msg.role == "assistant":
                context_parts.append(f"Sources: {', '.join(msg.sources)}")
        
        return "\n".join(context_parts)
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.messages = []
        self.summary = None
    
    def get_all_messages(self) -> List[Message]:
        """Get all messages in history."""
        return self.messages.copy()
    
    def get_exchange_count(self) -> int:
        """Get number of question-answer exchanges."""
        return len([m for m in self.messages if m.role == "user"])

