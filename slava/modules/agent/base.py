from abc import ABC, abstractmethod


class BaseHandler(ABC):

    @abstractmethod
    async def ahandle_prompt(self, prompt: str, chat_id: str) -> str:
        """Handle prompt from user."""
