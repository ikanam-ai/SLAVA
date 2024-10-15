from langchain.chat_models import GigaChat

from slava.config import GIGACHAT_SCOPE


class GigaChatModel:
    def __init__(self, api_key: str = None, scope: str = GIGACHAT_SCOPE, model_name: str = None):
        self.model = GigaChat(
            credentials=api_key,
            scope=scope,
            model=model_name,
        )
