from typing import List, Tuple


class ChatHistory:
    """Stores and formats chat history."""

    def __init__(self, max_history_length: int = 10):
        self.max_history_length = max_history_length
        self.history = []

    def add_chat(self, human_message: str, assistant_message: str) -> None:
        """Add a dialogue to the chat history."""
        self.history.append((human_message, assistant_message))
        if len(self.history) > self.max_history_length:
            self.history = self.history[-self.max_history_length:]

    def get_history(self) -> List[Tuple[str, str]]:
        """Retrieve the entire chat history."""
        return self.history

    def get_formatted_history(self) -> str:
        """Format chat history into a readable string."""
        formatted_history = ""
        for dialogue_turn in self.history:
            human = "Human: " + dialogue_turn[0]
            assistant = "Assistant: " + dialogue_turn[1]
            formatted_history += 10*"-" + "\n" + human + "\n" + 10*"-" + "\n" + assistant + "\n"
        return formatted_history

    def clear_history(self) -> None:
        """Clear the chat history."""
        self.history = []
        print("Chat history cleared.")
