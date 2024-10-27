import anthropic
from langchain_core.messages import HumanMessage

class ClaudeHandler:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

        self.model = self.Model(self.client)

    class Model:
        def __init__(self, client):
            self.client = client

        def invoke(self, messages: list[HumanMessage]) -> str:

            try:
                prompt = messages[0].content if messages else ""
                
                message = self.client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    # max_tokens=150,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                if isinstance(message.content, list):
                    response_text = ''.join(
                        block.text for block in message.content if hasattr(block, 'text')
                    )
                    return response_text.strip()
                else:
                    return message.content.strip()
            except Exception:
                return "Ошибка при генерации ответа."