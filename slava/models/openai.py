from langchain.llms import OpenAI

from slava.config import OPENAI_MODEL_TEMPERATURE


class OpenAIModel:
    def __init__(self, api_key: str = None, temperature: float = OPENAI_MODEL_TEMPERATURE):
        self.model = OpenAI(
            api_key=api_key,
            temperature=temperature,
        )
