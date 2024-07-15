from typing import List, Dict, Tuple
from langchain_core.chat_history import BaseChatMessageHistory
# from langchain.memory import 
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)
    k: int = 5 # last k messages

    def add_message(self, message: BaseMessage) -> None:
        """Add a self-created message to the store"""
        self.messages.append(message)
        self.messages = self.messages[-self.k:]

    def clear(self) -> None:
        self.messages = []


class ChatStore:
    """A store for managing chat histories."""

    def __init__(self, k: int = 3):
        self.store: Dict[Tuple[str, str], InMemoryHistory] = {}
        self.k = k

    def get_session_history(self, user_id: str, conversation_id: str) -> BaseChatMessageHistory:
        if (user_id, conversation_id) not in self.store:
            self.store[(user_id, conversation_id)] = InMemoryHistory(k=self.k)
        return self.store[(user_id, conversation_id)]

    def clear_session_history(self, user_id: str, conversation_id: str) -> None:
        """Clear the session history for a given user and conversation."""
        if (user_id, conversation_id) in self.store:
            self.store[(user_id, conversation_id)].clear()

    def add_message_to_history(self, user_id: str, conversation_id: str, message: BaseMessage) -> None:
        """Add a message to the session history for a given user and conversation."""
        history = self.get_session_history(user_id, conversation_id)
        history.add_message(message)

    def clear_all(self) -> None:
        """Clear all session histories."""
        for history in self.store.values():
            history.clear()

    def get_all_history(self) -> Dict[Tuple[str, str], InMemoryHistory]:
        return self.store

