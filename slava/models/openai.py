from openai import OpenAI
from langchain_core.messages import HumanMessage

class ChatGPTModel:
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
                    model="gpt-4o",
                    messages=openai_messages,
                    # max_tokens=150,
                    temperature=0.7
                )

                return completion.choices[0].message.content.strip()
            except Exception as e:
                return "Ошибка при генерации ответа."
