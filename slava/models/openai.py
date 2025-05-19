from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from openai import OpenAI

from slava.config import OPENAI_MODEL_MAX_TOKENS, OPENAI_MODEL_NAME, OPENAI_MODEL_TEMPERATURE


class OpenAIModel:
    def __init__(self, api_key: str, base_url: str = None, model_name: str = OPENAI_MODEL_NAME):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.model = self.Model(self.client, self.model_name)

    class Model(Runnable):
        def __init__(self, client, model_name: str = OPENAI_MODEL_NAME):
            self.client = client
            self.model_name = model_name

        def invoke(self, input: dict, config=None, **kwargs) -> str:
            prompt = str(input)
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]

                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=OPENAI_MODEL_MAX_TOKENS,
                    temperature=OPENAI_MODEL_TEMPERATURE,
                )

                return completion.choices[0].message.content.strip()
            except Exception as e:
                return f"Error: {e}"
