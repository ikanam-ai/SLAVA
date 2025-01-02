from langchain.chat_models import GigaChat

from slava.config import (
    GIGACHAT_MODEL_MAX_TOKENS,
    GIGACHAT_MODEL_SCOPE,
    GIGACHAT_MODEL_TEMPERATURE,
    GIGACHAT_MODEL_TOP_K,
)


class GigaChatModel:
    def __init__(
        self,
        api_key: str = None,
        scope: str = GIGACHAT_MODEL_SCOPE,
        model_name: str = None,
        temperature: float = GIGACHAT_MODEL_TEMPERATURE,
        max_tokens: int = GIGACHAT_MODEL_MAX_TOKENS,
        top_k: int = GIGACHAT_MODEL_TOP_K,
    ):
        self.model = GigaChat(
            credentials=api_key,
            scope=scope,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
        )
