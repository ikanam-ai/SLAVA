import anthropic
from langchain_core.messages import HumanMessage

from slava.config import CLAUDE_MODEL_MAX_TOKENS, CLAUDE_MODEL_NAME, TEXT_COLUMN


class ClaudeModel:
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
                    model=CLAUDE_MODEL_NAME,
                    max_tokens=CLAUDE_MODEL_MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}],
                )

                if isinstance(message.content, list):
                    response_text = "".join(block.text for block in message.content if hasattr(block, TEXT_COLUMN))
                    return response_text.strip()
                else:
                    return message.content.strip()
            except Exception:
                return "Error when generating the response by the model."
