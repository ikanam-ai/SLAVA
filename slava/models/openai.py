from langchain_core.messages import HumanMessage
from openai import OpenAI

from slava.config import OPENAI_MODEL_MAX_TOKENS, OPENAI_MODEL_NAME, OPENAI_MODEL_TEMPERATURE


class OpenAIModel:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = self.Model(self.client)

    class Model:
        def __init__(self, client):
            self.client = client

        def invoke(self, messages: list[HumanMessage]) -> str:
            try:
                openai_messages = [{"role": "system", "content": "You are a helpful assistant."}]
                for message in messages:
                    openai_messages.append({"role": "user", "content": message.content})

                completion = self.client.chat.completions.create(
                    model=OPENAI_MODEL_NAME,
                    messages=openai_messages,
                    max_tokens=OPENAI_MODEL_MAX_TOKENS,
                    temperature=OPENAI_MODEL_TEMPERATURE,
                )

                return completion.choices[0].message.content.strip()
            except Exception as e:
                return "Error when generating the response by the model."
